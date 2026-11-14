"""Field Studio GPU viewport (phase 0, 2026-11-08): the field atlas on the card and the frame passes.

GpuScene holds every volume (a Field Studio object, or a published FV8 store such as her field) in ONE brick pool on
the card, in the FV8 store layout (32 mm tiles, 4 levels, 9^3 fp16 bricks with apron). The CPU core stays the one
authority for the field: it writes bricks, this module mirrors them (whole volumes at load, dirty bricks after an
edit). No mesh anywhere (P14).

Frame = passes: pass 0 traces every pixel of a rectangle (depth, object, level, first light sample), passes 1.. add
light samples (sun soft shadow + sky rays) into a float accumulator; resolve tone-maps to BGRA8. Incremental: a
rectangle can be re-traced and its accumulation restarted without touching the rest of the frame.
"""
import json
import math
import os
import time

import numpy as np

ORTHO = True            # fs-shell 11-11: cam_params carries the orthographic view (P[17], P[18])

HERE = os.path.dirname(os.path.abspath(__file__))
NLEV = 4
TAB_SCATTER = os.environ.get('FSB_TAB_SCATTER', '1') != '0'   # fs-brush 11-12: only the changed table entries uploaded
# fs-brush-4 11-13 (n3): the window's sky visibility kept per surface cell (trace.cu sky_key / sky_slot / sky_invalidate).
# on by default (FSB_SKY_CACHE=0: off, today's per-pixel rays bit for bit); _LOG2N slots (22: 4 M, 64 MB), _BASE the finest cell (m),
# _NMAX rays a cell holds, _FOOT the cell in pixel footprints, _PAD an edit's reach (m)
SKY_CACHE = os.environ.get('FSB_SKY_CACHE', '1') != '0'
SKY_LOG2N = int(os.environ.get('FSB_SKY_LOG2N', '22'))
SKY_BASE = float(os.environ.get('FSB_SKY_BASE', '3.125e-5'))
SKY_NMAX = float(os.environ.get('FSB_SKY_NMAX', '64'))
SKY_FOOT = float(os.environ.get('FSB_SKY_FOOT', '1.5'))
SKY_PAD = float(os.environ.get('FSB_SKY_PAD', '0.020'))
SKY_PRIOR = float(os.environ.get('FSB_SKY_PRIOR', '4'))   # the cell serves camera-move passes only (the first pass after a restart), as a prior of this many rays
FAR_LAZY = os.environ.get('FSB_FAR_LAZY', '1') != '0'   # fs-brush-4 11-13: _vol_arrays keeps the far grid as a uint8 view (no copy per admission)
TAB_ROOM = float(os.environ.get('FSB_TAB_ROOM', '2.0'))      # fs-brush 11-12: a tile table's room on the card (x its length)
SUN_RADIUS_DEG = 0.267        # the sun disc's angular radius (degrees)
VF, VI = 48, 20
VF_VIS = 46                   # trace.cu VF_VIS (1 visible); picks pass through volumes with 0
NOON_GRASS_DIR = os.path.join(os.environ.get('FIELDSTUDIO_HDRI_DIR') or os.path.join(os.path.expanduser('~'), 'FieldStudio', 'hdri'), 'noon_grass')   # optional: a CC0 HDRI folder


def _default_env():
    """fs-shell 11-13 (Tier 0 part 3: a grass and trees scene as the default distracts and is not needed
    there): the default environment is the neutral studio (gpu/studio_env.py: a formula, made once into
    ~/FieldStudio/env); noon_grass and every other HDRI folder stay choosable (the Light panel). FS_ENV=noon_grass (or a
    folder) picks another default; a failure to make the studio falls back to noon_grass."""
    e = os.environ.get('FS_ENV', '')
    if e == 'noon_grass':
        return NOON_GRASS_DIR
    if e and os.path.isfile(os.path.join(e, 'env.json')):
        return os.path.normpath(e)
    try:
        from .studio_env import ensure
        return os.path.normpath(ensure())
    except Exception as ex:
        print('fieldstudio: the neutral studio could not be made (%r): noon_grass is the environment' % (ex,), flush=True)
        return NOON_GRASS_DIR


HDRI_DIR = _default_env()


POOL_LOG = []        # (reason, MB held before, MB after, ms) of each release (tests read it)


def to_dev(cp, arr):
    """fs-brush 11-11 (twice tonight her store failed to load on the card under RAM pressure, cudaHostAlloc /
    cudaErrorAlreadyMapped on a pinned host buffer): cp.asarray(numpy) stages the copy through CuPy's PINNED pool (a
    page-locked buffer the size of the array: ~48 MB a brick chunk here). This copies with ndarray.set() from the
    pageable array instead (a synchronous cudaMemcpy; no pinned allocation), so loading her store never depends on one."""
    a = np.ascontiguousarray(arr)
    d = cp.empty(a.shape, a.dtype)
    if a.size:
        d.set(a)
    return d


def release_cached(reason=''):
    """11-09 (the renderer and the liquids lane share the card): give CuPy's cached blocks back to the
    driver (the pool keeps every freed block up to its cap). Blocks in use are untouched. -> MB released"""
    import time as _t
    try:
        import cupy as cp
    except Exception:
        return 0.0
    t0 = _t.perf_counter()
    pool = cp.get_default_memory_pool()
    before = pool.total_bytes()
    pool.free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    after = pool.total_bytes()
    POOL_LOG.append((reason, round(before / 2 ** 20), round(after / 2 ** 20), round((_t.perf_counter() - t0) * 1000, 1)))
    del POOL_LOG[:-50]
    return (before - after) / 2 ** 20


POOL_SLACK_MB = float(os.environ.get('FS_GPU_POOL_SLACK_MB', '512'))   # fs-trust 11-11: unused blocks kept between safe points
POOL_RESERVE = 0.5e9


def trim(reason=''):
    """fs-trust 11-11 (M, R9c on a test project: the card peaked at 6.9 of 8 GB with 2.2 GB of the pool held but unused):
    at a safe point OFF the stroke path (a save; Open and idle keep ui/engine's release_cached) the pool's unused blocks
    go back when they exceed POOL_SLACK_MB, and the pool's limit is re-set under what the card has free right now (less
    POOL_RESERVE, never above the cap chosen at start, never under what is in use + 0.5 GB; FS_GPU_POOL_GB still sets the
    cap outright). R9d (15 min, a project copy): trimming at every stroke's end lowered the card's peak 6.91 -> 6.57 GB and its
    median 5.92 -> 5.44 GB, but freed ~1 GB a stroke that the next stroke allocated again, up to 258 ms on the release
    path: so it is not called there (fs-brush's condition). -> MB released"""
    try:
        import cupy as cp
    except Exception:
        return 0.0
    if not getattr(_cp, 'capped', False):
        return 0.0                                   # the card is not in use in this process
    import time as _t
    t0 = _t.perf_counter()
    pool = cp.get_default_memory_pool()
    mb = 0.0
    if (pool.total_bytes() - pool.used_bytes()) / 2 ** 20 > POOL_SLACK_MB:
        mb = release_cached('trim:' + reason)
    if not os.environ.get('FS_GPU_POOL_GB'):
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            cap0 = getattr(_cp, 'pool_cap', int(7e9))
            lim = min(cap0, pool.total_bytes() + free - POOL_RESERVE)
            lim = max(lim, pool.used_bytes() + 0.5e9)
            pool.set_limit(size=int(lim))
            _cp.limit_now = int(lim)
        except Exception:
            pass
    TRIM_LOG.append((reason, round(mb), round(getattr(_cp, 'limit_now', 0) / 2 ** 20), round((_t.perf_counter() - t0) * 1000, 1)))
    del TRIM_LOG[:-200]
    return mb


TRIM_LOG = []        # (reason, MB released, limit MB after, ms) of each trim (tests and the R9 harness read it)


def _cp():
    import cupy as cp
    if not getattr(_cp, 'capped', False):
        # 11-09 (engine/lessons/cupy_pool_ram.md, the liquids lane): CuPy's pool keeps freed blocks; under WDDM, past
        # the 8 GB card they spill into system RAM. A cap (FS_GPU_POOL_GB, default 7) makes the pool give cached blocks
        # back before it grows past it
        # 11-09 (T8b): the cap stays within what the card has FREE when Field Studio starts using it (the
        # card also drives the display and other lanes' work), less a 0.5 GB reserve, at most 7 GB; FS_GPU_POOL_GB
        # sets it outright. _cp.pool_cap / _cp.free_at_start record what was chosen (tests report them).
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            env = os.environ.get('FS_GPU_POOL_GB')
            cap = float(env) * 1e9 if env else min(7e9, max(1e9, free - 0.5e9))
            cp.get_default_memory_pool().set_limit(size=int(cap))
            _cp.pool_cap, _cp.free_at_start, _cp.card_total = int(cap), int(free), int(total)
        except Exception:
            pass
        _cp.capped = True
    return cp


class Volume:
    """One volume's tables, host side. Local frame: local = s * R^T (world - t)."""

    def __init__(self, name, origin, tile_m, dims, levels, lo, hi, albedo=(0.66, 0.56, 0.49)):
        self.name = name
        self.origin = np.asarray(origin, float)
        self.tile_m = float(tile_m)
        self.dims = tuple(int(x) for x in dims)
        self.levels = levels          # list of NLEV dicts or None: {h, band, bpt, tile_table (S, bpt^3) int32, n_bricks}
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.albedo = tuple(albedo)
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.s = 1.0
        self.visible = True
        self.opacity = 1.0            # fs-shell 11-13: view only (the X-ray composite); never in the scene or saved
        self.brick_off = [0] * NLEV   # pool row of brick id 0, per level
        self.grid = None              # (nx, ny, nz, NLEV) int32

    def set_transform(self, R=None, t=None, s=None):
        if R is not None:
            self.R = np.asarray(R, float)
        if t is not None:
            self.t = np.asarray(t, float)
        if s is not None:
            self.s = float(s)

    def ftab(self):
        f = np.zeros(VF, np.float32)
        Rt = self.R.T * self.s
        M = np.concatenate([Rt, (-Rt @ self.t)[:, None]], 1)       # world -> local
        MI = np.concatenate([self.R / self.s, self.t[:, None]], 1)  # local -> world
        f[0:12] = M.reshape(-1)
        f[12:24] = MI.reshape(-1)
        f[24] = self.s
        f[25:28] = self.origin
        f[28] = self.tile_m
        for l in range(NLEV):
            lv = self.levels[l]
            f[29 + l] = lv['h'] if lv else 1.0
            f[33 + l] = lv['band'] if lv else 0.0
        f[37:40] = self.lo
        f[40:43] = self.hi
        f[43:46] = self.albedo
        f[46] = 1.0 if self.visible else 0.0
        f[47] = float(getattr(self, 'opacity', 1.0))   # fs-shell 11-13: the X-ray's per-object opacity (gpu/viewmodes.py)
        return f


class GpuScene:
    def __init__(self, cap_bricks=1 << 16):
        cp = _cp()
        self.cp = cp
        self.vols = []
        self.cap = int(cap_bricks)
        self.sdf = cp.empty((self.cap, 729), cp.float16)
        self.used = 0
        self.grid_parts, self.ttab_parts = [], []
        self._dirty_tables = True
        src = open(os.path.join(HERE, 'trace.cu'), encoding='utf-8').read()
        t = time.perf_counter()
        # 11-09: FS_TRACE_DBG=1 builds the B2 diagnosis marker (trace.cu TRACE_DBG); never set in the app
        self.mod = cp.RawModule(code=src, options=('--use_fast_math', '-std=c++14')
                                + (('-DTRACE_DBG',) if os.environ.get('FS_TRACE_DBG') == '1' else ()))
        self.k_trace = self.mod.get_function('trace_gbuf')
        self.k_light = self.mod.get_function('light_pass')
        self.k_resolve = self.mod.get_function('resolve')
        self.k_scatter = self.mod.get_function('scatter_bricks')
        self.k_restart = self.mod.get_function('restart')
        self.k_pick = self.mod.get_function('pick_rays')
        try:                                       # fs-brush 11-13: the move's live view (trace.cu); absent in older text
            self.k_mvl = self.mod.get_function('mv_lgrid')
            self.k_mvd = self.mod.get_function('mv_ldil')
        except Exception:
            self.k_mvl = self.k_mvd = None
        self._mv = None
        self.compile_s = time.perf_counter() - t
        # fs-shell 11-13: every pixel pass goes through the launch guard (gpu/launch_guard.py): no single launch long
        # enough to reset the display driver (TDR ~2 s); a launch past 1 s stops the call (GuardStop), nothing relaunched
        from .launch_guard import LaunchGuard
        self.guard = LaunchGuard(cp)
        self._k_light0 = self.k_light
        self._load_env()

    # ---- environment (the neutral studio by default since fs-shell 11-13; noon_grass and other HDRIs on request) --
    def _load_env(self, folder=None):
        """folder: an HDRI folder (env_2k.npy + env.json; fs-light's rig switches it, 11-11); None = HDRI_DIR"""
        cp = self.cp
        folder = HDRI_DIR if folder is None else folder
        env = np.load(os.path.join(folder, 'env_2k.npy'))
        self.env_h, self.env_w = env.shape[:2]
        self.d_env = cp.asarray(np.ascontiguousarray(env.astype(np.float16)))
        ej = json.load(open(os.path.join(folder, 'env.json')))
        self.hdri_dir = folder
        self._env_gen = getattr(self, '_env_gen', 0) + 1
        self.envj = ej
        sb = np.array(ej['sun_dir_hdri'], float)
        self.sun_rgb = np.array(ej['sun_irradiance_rgb'], float)
        self.sun_b = sb / np.linalg.norm(sb)
        self.env_rot_deg = 0.0
        self._env_np = env.astype(np.float32)
        self._sh_rot = None

    def sky_sh(self):
        """9 SH irradiance coefficients per channel (world frame, y up), cosine-convolved (Ramamoorthi-Hanrahan),
        projected from the environment with the kernel's own direction mapping and rotation."""
        if self._sh_rot == self.env_rot_deg:
            return self._sh
        env = self._env_np
        eh0, ew0 = env.shape[:2]
        env = env.reshape(eh0 // 4, 4, ew0 // 4, 4, 3).mean((1, 3))     # block mean: enough for 9 SH terms
        eh, ew = env.shape[:2]
        v = (np.arange(eh) + 0.5) / eh * np.pi
        u = (np.arange(ew) + 0.5) / ew
        V, U = np.meshgrid(v, u, indexing='ij')
        bz = np.cos(V)
        phi = (U - 0.5) * 2 * np.pi
        rx, ry = np.sin(V) * np.cos(phi), np.sin(V) * np.sin(phi)
        r = math.radians(self.env_rot_deg)
        cr, sr = math.cos(r), math.sin(r)
        bx, by = cr * rx + sr * ry, -sr * rx + cr * ry          # inverse of the kernel's rotation
        x, y, z = bx, bz, -by                                    # Blender -> world (y up)
        dO = (2 * np.pi / ew) * (np.pi / eh) * np.sin(V)
        Y = np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y,
                      1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], 0)
        A = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4,
                      np.pi / 4])
        sh = np.zeros((3, 9), np.float32)
        for c in range(3):
            sh[c] = A * (Y * (env[..., c] * dO)[None]).reshape(9, -1).sum(1)
        self._sh = sh.reshape(-1)
        self._sh_rot = self.env_rot_deg
        self.d_sh = self.cp.asarray(self._sh)
        return self._sh

    def _sky_gain(self, g):
        """fs-light 11-11: the SH sky on the card scaled by a rig's environment.sky_gain (a dark studio). 1 = the SH
        sky_sh uploaded (bit for bit: nothing is touched unless a gain other than 1 was applied)."""
        key = (self._sh_rot, float(g), getattr(self, '_env_gen', 0))
        prev = getattr(self, '_sh_gain_key', None)
        if key == prev:
            return
        self._sh_gain_key = key
        if g == 1.0 and (prev is None or prev[1] == 1.0):
            return
        self.d_sh = self.cp.asarray(self._sh * np.float32(g))

    def sun_world(self):
        r = math.radians(self.env_rot_deg)
        cr, sr = math.cos(r), math.sin(r)
        b = self.sun_b
        bx, by, bz = cr * b[0] + sr * b[1], -sr * b[0] + cr * b[1], b[2]     # inverse of the kernel's rotation
        w = np.array([bx, bz, -by])                                          # Blender (x, y, z) -> world (x, z, -y)
        return w / np.linalg.norm(w), cr, sr

    # ---- pool -----------------------------------------------------------------------------------------
    def _reserve(self, n):
        cp = self.cp
        if self.used + n <= self.cap:
            return
        cap = max(self.cap * 2, self.used + n + 1024)
        new = cp.empty((cap, 729), cp.float16)
        new[:self.used] = self.sdf[:self.used]
        self.sdf = new
        self.cap = cap

    def alloc_rows(self, n):
        """n pool rows for a mirrored object (free list first, then the bump end)."""
        fl = getattr(self, '_free', None)
        if fl is None:
            fl = self._free = []
        k = min(n, len(fl))
        out = np.empty(n, np.int64)
        if k:
            out[:k] = fl[-k:]
            del fl[-k:]
        if n - k:
            self._reserve(n - k)
            out[k:] = np.arange(self.used, self.used + n - k)
            self.used += n - k
        return out

    def free_rows(self, rows):
        if not hasattr(self, '_free'):
            self._free = []
        rows = [int(r) for r in rows]
        self._free.extend(rows)
        # fs-mouth 11-12: an object's drop (a large block; a stroke frees single rows) also gives
        # the atlas's free TAIL back: undoing the mouth set had kept its ~2 GB of bricks on the card
        if len(rows) >= 4096:
            self._trim_tail()

    def _trim_tail(self, min_mb=256.0):
        """the free rows at the atlas's end released (used lowered); the atlas copied to a smaller array (as _reserve
        copies to a bigger one) once the unused capacity passes min_mb, and the pool's blocks returned to the driver"""
        fl = self._free
        if not fl:
            return 0.0
        fs = set(fl)
        u = self.used
        while u > 0 and (u - 1) in fs:
            u -= 1
        if u == self.used:
            return 0.0
        self._free = [r for r in fl if r < u]
        self.used = u
        cap = max(u + 1024, 1 << 16)
        if cap < self.cap and (self.cap - cap) * 729 * 2 / 2 ** 20 >= min_mb:
            new = self.cp.empty((cap, 729), self.cp.float16)
            if u:
                new[:u] = self.sdf[:u]
            self.sdf = new
            self.cap = cap
            self._dirty_tables = True
            return release_cached('atlas_trim')
        return 0.0

    def add_volume(self, vol, bricks_per_level):
        """bricks_per_level[l]: (N, 9, 9, 9) fp16 array-like (numpy or memmap) or None. Copied to the pool in chunks
        (host RAM stays at one chunk)."""
        cp = self.cp
        tot = sum(len(b) for b in bricks_per_level if b is not None)
        self._reserve(tot)
        # fs-brush 11-11: copied with ndarray.set() (no pinned staging buffer; see to_dev) in 12 MB chunks; a failed upload
        # gives its rows back (before, a half-added volume kept them and a retry appended another copy)
        CH = 8192
        used0 = self.used
        try:
            for l in range(NLEV):
                b = bricks_per_level[l]
                vol.brick_off[l] = self.used
                if b is None:
                    continue
                for i in range(0, len(b), CH):
                    part = np.ascontiguousarray(np.asarray(b[i:i + CH]), dtype=np.float16).reshape(-1, 729)
                    self.sdf[self.used + i:self.used + i + len(part)].set(part)
                self.used += len(b)
        except Exception:
            self.used = used0
            raise
        self.vols.append(vol)
        self._dirty_tables = True
        return len(self.vols) - 1

    def write_bricks(self, rows, vals):
        """Incremental upload: pool rows <- vals (n, 729) fp16 (host). Returns seconds."""
        cp = self.cp
        n = len(rows)
        if n == 0:
            return 0.0
        d_rows = cp.asarray(np.asarray(rows, np.int32))
        d_vals = cp.asarray(np.ascontiguousarray(vals, dtype=np.float16).reshape(n, 729))
        th = 256
        self.k_scatter(((n * 729 + th - 1) // th,), (th,), (self.sdf, d_rows, d_vals, np.int32(n)))
        return n

    def touch_tables(self, v, grid_idx=None, tt_idx=None):
        """a volume's grid / tile tables / far field changed: only its segments are uploaded again (when the layout
        holds).
        fs-brush 11-12 (brush-study's stage map on her, STUDY 1e: every admitting batch re-uploaded her volume's whole
        5 M-entry grid and tile tables, 20-36 ms a call, 50-186 ms per lid stroke, 0.5-1.4 s per body stroke): grid_idx
        (flat indices into v.grid) and tt_idx ({level: flat indices into that level's tile_table}) name the ONLY
        entries that changed; _tables then writes just those entries on the card. Called without them, anything may
        have changed: the volume's whole segments, as before. FSB_TAB_SCATTER=0: always whole segments."""
        v.tab_version = getattr(v, 'tab_version', 0) + 1
        self._dirty_tables = True
        if not TAB_SCATTER or (grid_idx is None and tt_idx is None) or getattr(v, '_tab_full', True):
            v._tab_full = True                   # the whole segments at the next upload (a new volume starts so)
            v._tab_dg, v._tab_dt = [], {}
            return
        if grid_idx is not None and len(grid_idx):
            v._tab_dg.append(np.asarray(grid_idx, np.int64).ravel())
        for l, ix in (tt_idx or {}).items():
            if ix is not None and len(ix):
                v._tab_dt.setdefault(int(l), []).append(np.asarray(ix, np.int64).ravel())

    def touch_box(self):
        """fs-brush 11-12: a volume's box (its ftab row) changed but no table entry did: the small per-volume rows are
        uploaded at the next _tables, no segment"""
        self._dirty_tables = True

    def _scatter_entries(self, dst, off, arr, chunks):
        """the changed entries of one segment (flat indices into arr) written into the card's buffer at off + index"""
        idx = np.unique(np.concatenate(chunks)) if len(chunks) > 1 else np.unique(chunks[0])
        if not len(idx):
            return 0
        cp = self.cp
        vals = np.ascontiguousarray(arr[idx], dtype=dst.dtype)
        dst[to_dev(cp, idx + off)] = to_dev(cp, vals)
        return len(idx)

    def _vol_arrays(self, v):
        key = getattr(v, 'tab_version', 0)
        c = getattr(v, '_tab_cache', None)
        if c is not None and c[0] == key:
            return c
        # views, not copies, when the arrays are int32 already (her object's 0.25 mm grid is 5 M entries)
        g = np.ascontiguousarray(v.grid, np.int32).reshape(-1)
        tts = [None if v.levels[l] is None else np.ascontiguousarray(v.levels[l]['tile_table'], np.int32).reshape(-1)
               for l in range(NLEV)]
        far = getattr(v, 'far', None)
        if FAR_LAZY:
            # fs-brush-4 11-13 (fs-store-6's OK; M, fs-brush/speed4/p3: 3.1-3.8 ms of every brush batch that admits bricks went
            # to rebuilding these arrays, the far grid's uint8 -> int32 copy each time the tile tables changed, while the
            # partial upload never sends the far segment): the far grid as a uint8 view. Every reader converts it as it goes
            # (the full upload's concatenate promotes it to int32; a segment write converts to the card buffer's dtype), so the
            # card holds the same bytes
            fa = None if far is None else far.reshape(-1)
        else:
            fa = None if far is None else far.astype(np.int32).reshape(-1)
        c = (key, g, tts, fa)
        v._tab_cache = c
        return c

    def _tables(self):
        if not self._dirty_tables:
            return
        cp = self.cp
        _t0 = time.perf_counter()
        # Grab fps (11-08 night): per-volume arrays cached by table version; when every segment keeps its size, only
        # the volumes whose version changed are copied into the card's buffers (before: everything, every admission).
        # fs-brush 11-12 (tl1, her body strokes: each time her object's tile table grew, 4,096 -> 190,272 ints over a
        # few strokes, the layout changed and EVERY volume's grid and tile tables went to the card again, 20.5 M ints
        # incl. her 17.4 M-entry grid, 25-565 ms a time): each tile table holds a ROOM on the card (its length x
        # TAB_ROOM, the rest -1), so a table that grows within its room moves nothing and only its written entries go
        # (touch_tables' indices; new rows are -1 there already); the grids have their own layout and are re-sent only
        # when theirs changes. FSB_TAB_SCATTER=0: rooms = lengths, whole segments, as before.
        arrs = [self._vol_arrays(v) for v in self.vols]
        ids = [id(v) for v in self.vols]
        same_ids = getattr(self, '_up_ids', None) == ids
        old_caps = getattr(self, '_tab_caps', None) if same_ids else None
        caps = []
        for i, (v, a) in enumerate(zip(self.vols, arrs)):
            row = []
            for l in range(NLEV):
                t = a[2][l]
                if t is None:
                    row.append(-1)
                    continue
                n = len(t)
                c = old_caps[i][l] if old_caps is not None else -1
                if c < n:
                    c = int(n * TAB_ROOM) + 262144 if (TAB_SCATTER and n) else n
                row.append(c)
            caps.append(row)
        glay = tuple((len(a[1]), -1 if a[3] is None else len(a[3])) for a in arrs)
        tlay = tuple(tuple(r) for r in caps)
        have_up = getattr(self, '_up_versions', None) is not None and len(self._up_versions) == len(arrs)
        g_partial = same_ids and have_up and getattr(self, '_glay', None) == glay and getattr(self, 'd_grid', None) is not None
        t_partial = same_ids and have_up and getattr(self, '_tlay', None) == tlay and getattr(self, 'd_ttab', None) is not None
        vit, vft = [], []
        goff = toff = 0
        segs = []                                    # per volume: [(kind, off, arr, level or None)]
        for i, (v, a) in enumerate(zip(self.vols, arrs)):
            vi = np.zeros(VI, np.int32)
            vi[0] = goff
            vi[1:4] = v.dims
            g = a[1]
            seg = [('g', goff, g, None)]
            goff += len(g)
            for l in range(NLEV):
                lv = v.levels[l]
                if lv is None:
                    continue
                tt = a[2][l]
                seg.append(('t', toff, tt, l))
                vi[4 + l] = toff
                vi[8 + l] = lv['bpt']
                vi[12 + l] = v.brick_off[l]
                toff += max(caps[i][l], 0)
            vi[16] = -1
            mv = getattr(v, 'mask_vol', None)        # milestone: an editable object's edited bricks mask this one
            vi[18] = self.vols.index(mv) if mv is not None and mv in self.vols else -1
            # 11-09: this volume masks another (an editable store object's edited bricks): a brick it lacks may hold her
            # surface (drawn by the other volume), so the band is no lower bound there (trace.cu VI_MASKER)
            vi[19] = 1 if any(getattr(w, 'mask_vol', None) is v for w in self.vols) and not os.environ.get('FS_NO_MASKER') else 0
            if a[3] is not None:
                vi[16] = goff
                vi[17] = v.far_level
                seg.append(('f', goff, a[3], None))
                goff += a[3].size
            vit.append(vi)
            vft.append(v.ftab())
            segs.append(seg)
        if os.environ.get('FS_VALIDATE'):
            bad = self.validate()
            if bad:
                raise RuntimeError('GPU tables invalid: ' + '; '.join(bad[:12]))
        vers = [a[0] for a in arrs]
        n_up = n_sc = 0
        if not g_partial:
            grid = []
            for seg in segs:
                grid += [arr for kind, off, arr, l in seg if kind in ('g', 'f')]
            self.d_grid = to_dev(cp, np.concatenate(grid) if grid else np.zeros(1, np.int32))   # fs-brush 11-11: no pinned
            n_up += int(goff)
        if not t_partial:
            tt_all = []
            for i, seg in enumerate(segs):
                for kind, off, arr, l in seg:
                    if kind == 't':
                        tt_all.append(np.ascontiguousarray(arr, np.int32))
                        pad = caps[i][l] - len(arr)
                        if pad > 0:
                            tt_all.append(np.full(pad, -1, np.int32))
            self.d_ttab = to_dev(cp, np.concatenate(tt_all) if tt_all else np.zeros(1, np.int32))   # staging (to_dev)
            n_up += int(toff)
        if g_partial or t_partial:
            for i, seg in enumerate(segs):
                if vers[i] == self._up_versions[i]:
                    continue
                v = self.vols[i]
                scat = TAB_SCATTER and not getattr(v, '_tab_full', True)
                dg, dt = (v._tab_dg, v._tab_dt) if scat else ([], {})
                for kind, off, arr, l in seg:
                    if arr is None or not len(arr):
                        continue
                    if kind in ('g', 'f'):
                        if not g_partial:
                            continue                 # sent whole above
                        if scat and kind == 'g':
                            if dg:
                                n_sc += self._scatter_entries(self.d_grid, off, arr, dg)
                            continue
                        if scat and kind == 'f':
                            continue                 # the far field changes only through a whole-volume touch
                        dst = self.d_grid
                    else:
                        if not t_partial:
                            continue
                        if scat:
                            if dt.get(l):
                                n_sc += self._scatter_entries(self.d_ttab, off, arr, dt[l])
                            continue
                        dst = self.d_ttab
                    dst[off:off + len(arr)].set(np.ascontiguousarray(arr, dtype=dst.dtype))   # fs-brush 11-11: no pinned staging
                    n_up += len(arr)
        for v in self.vols:                          # fs-brush 11-12: what was named is on the card now
            v._tab_full, v._tab_dg, v._tab_dt = False, [], {}
        self.table_stats = {'partial': bool(g_partial and t_partial), 'grid_partial': bool(g_partial),
                            'ttab_partial': bool(t_partial), 'ints': n_up, 'scattered': n_sc}
        _t1 = time.perf_counter()
        if os.environ.get('FSB_TAB_LOG'):            # fs-brush 11-12: one record per upload (which path, how much, why)
            log_ = getattr(self, 'table_log', None)
            if log_ is None:
                log_ = self.table_log = []
            why = []
            if getattr(self, '_glay', None) != glay:
                why.append('grid layout')
            if getattr(self, '_tlay', None) != tlay:
                why.append('tile-table rooms')
            if not same_ids:
                why.append('volumes')
            log_.append(dict(self.table_stats, t=round(_t0, 3), upload_ms=round((_t1 - _t0) * 1000, 1), why=why,
                             sizes=[(len(a[1]), sum(len(t) for t in a[2] if t is not None)) for a in arrs]))
        self._glay, self._tlay, self._tab_caps = glay, tlay, caps
        self._layout, self._up_versions, self._up_ids = (glay, tlay), vers, ids
        self.d_vit = cp.asarray(np.stack(vit) if vit else np.zeros((1, VI), np.int32))
        self.d_vft = cp.asarray(np.stack(vft) if vft else np.zeros((1, VF), np.float32))
        self.d_wbox = cp.asarray(self._wboxes())
        for nm, arr in (('g_vft', self.d_vft), ('g_vit', self.d_vit)):
            gp = self.mod.get_global(nm)
            cp.ndarray((1,), cp.uint64, gp)[...] = arr.data.ptr
        self._dirty_tables = False
        if SKY_CACHE:
            # n3: a moved, added, removed or hidden volume changes every cell near it: the cache starts over
            tk_ = tuple((id(v), v.ftab()[:29].tobytes(), bool(v.visible)) for v in self.vols)
            if tk_ != getattr(self, '_sky_tkey', None):
                self._sky_tkey = tk_
                self.sky_clear()

    def validate_device(self):
        """r46 debug: the uploaded tables equal what the host volumes hold now (a host change without the dirty flag
        leaves the card reading stale indices) -> list of problems"""
        if self._dirty_tables:
            return []
        keep = (self.d_grid, self.d_ttab, self.d_vit, self.d_vft, self.d_wbox)
        self._dirty_tables = True
        self._tables()
        new = (self.d_grid, self.d_ttab, self.d_vit, self.d_vft, self.d_wbox)
        bad = []
        for nm, a, b in zip(('grid', 'ttab', 'vit', 'vft', 'wbox'), keep, new):
            if a.shape != b.shape:
                bad.append('%s shape %s on the card, %s on the host' % (nm, a.shape, b.shape))
            elif not bool((a == b).all()):
                bad.append('%s differs at %d entries' % (nm, int((a != b).sum())))
        return bad

    def validate(self):
        """r46 debug (FS_VALIDATE=1): every index the kernels follow stays inside its array -> list of problems"""
        bad = []
        rows = int(self.sdf.shape[0]) if getattr(self, 'sdf', None) is not None else 0
        for vi_, v in enumerate(self.vols):
            nm = '%d:%s' % (vi_, getattr(v, 'name', '?'))
            if tuple(v.grid.shape[:3]) != tuple(int(x) for x in v.dims):
                bad.append('%s grid %s vs dims %s' % (nm, v.grid.shape, tuple(v.dims)))
            for l in range(NLEV):
                lv = v.levels[l]
                g = v.grid[..., l]
                if lv is None:
                    if np.any(g >= 0):
                        bad.append('%s level %d absent but grid holds slots' % (nm, l))
                    continue
                bpt = int(lv['bpt'])
                tt = lv['tile_table']
                ntile = tt.shape[0] if tt.ndim > 1 else len(tt) // bpt ** 3
                if tt.size != ntile * bpt ** 3:
                    bad.append('%s L%d tile_table size %d not tiles x bpt^3 (%d)' % (nm, l, tt.size, bpt))
                if g.max(initial=-1) >= ntile:
                    bad.append('%s L%d grid slot %d >= tiles %d' % (nm, l, int(g.max()), ntile))
                ids = tt[tt >= 0]
                if len(ids):
                    top = int(ids.max()) + int(v.brick_off[l])
                    if top >= rows:
                        bad.append('%s L%d brick row %d >= pool rows %d (off %d)' % (nm, l, top, rows, int(v.brick_off[l])))
                    if int(ids.min()) + int(v.brick_off[l]) < 0:
                        bad.append('%s L%d negative brick row' % (nm, l))
            far = getattr(v, 'far', None)
            if far is not None:
                fl = v.levels[v.far_level]
                want = tuple(int(d) * int(fl['bpt']) for d in v.dims) if fl is not None else None
                if want is None or tuple(far.shape) != want:
                    bad.append('%s far %s vs %s (level %d)' % (nm, far.shape, want, v.far_level))
        return bad

    @staticmethod
    def far_field(vol, level, ijk):
        """r10: the volume's far field from its brick cells (ijk at `level`, cells spanning the tile grid): floor of the
        Euclidean distance (in cells, capped 255) from each cell centre to the nearest cell holding a brick."""
        from scipy.ndimage import distance_transform_edt
        n = np.array(vol.dims) * vol.levels[level]['bpt']
        ijk = np.asarray(ijk, np.int64).reshape(-1, 3)
        ijk = ijk[np.all((ijk >= 0) & (ijk < n), 1)]
        if len(ijk) == 0:
            vol.far = None
            return
        occ = np.ones(tuple(n), bool)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = False
        vol.far = np.minimum(np.floor(distance_transform_edt(occ)), 255).astype(np.uint8)
        vol.far_level = level

    def _wboxes(self):
        out = np.zeros((max(len(self.vols), 1), 6), np.float32)
        for i, v in enumerate(self.vols):
            c = np.array([[x, y, z] for x in (v.lo[0], v.hi[0]) for y in (v.lo[1], v.hi[1]) for z in (v.lo[2], v.hi[2])])
            w = (v.R @ c.T).T / v.s + v.t
            out[i, :3] = w.min(0)
            out[i, 3:] = w.max(0)
        return out.reshape(-1)

    def refresh_transforms(self):
        """Transforms / visibility only (no table rebuild)."""
        if self._dirty_tables:
            return self._tables()
        # 11-10: an empty scene (File > New, the last object deleted) kept the card's view only until this refresh, which
        # failed on zero volumes and sent the session to CPU drawing; the placeholder _tables() uses
        self.d_vft = self.cp.asarray(np.stack([v.ftab() for v in self.vols]) if self.vols else np.zeros((1, VF), np.float32))
        self.d_wbox = self.cp.asarray(self._wboxes())

    # ---- FV8 store (her published field) ------------------------------------------------------------------
    def add_fv8_store(self, path, name='fv8', levels=(0, 1, 2, 3), albedo=(0.66, 0.56, 0.49)):
        """11-09 night (a run lost the card at 22:37: OutOfMemory doubling the pool): her store's bricks go onto the card
        ONCE per store. Every later volume of the same store (her object rebuilt after a duplicate, an undone delete,
        View > Published field) is a new Volume sharing those rows, with its own transform and visibility. Before,
        each one appended another 421k rows (0.6 GB) that were never returned (phase0/pool_growth_probe.py)."""
        import copy
        cache = self.__dict__.setdefault('_store_once', {})
        key = (os.path.normcase(os.path.abspath(path)), tuple(levels))
        tpl = cache.get(key)
        if tpl is not None:
            v = copy.copy(tpl)                      # shares grid, levels, brick_off, far (never written for a store)
            v.name = name
            v.R, v.t, v.s, v.visible, v.albedo = np.eye(3), np.zeros(3), 1.0, True, tuple(albedo)
            for k in ('mask_vol', 'oid', 'src'):
                v.__dict__.pop(k, None)
            self.vols.append(v)
            self._dirty_tables = True
            return v
        v = self._load_fv8_store(path, name, levels, albedo)
        cache[key] = copy.copy(v)
        return v

    def _load_fv8_store(self, path, name, levels, albedo):
        meta = json.load(open(os.path.join(path, 'meta.json')))
        tg = np.load(os.path.join(path, 'tile_grid.npy'))           # (nx, ny, nz, 4)
        levs, bricks = [None] * NLEV, [None] * NLEV
        grid = np.full(tg.shape, -1, np.int32)
        for l in levels:
            key = 'L%d' % l
            lm = meta['levels'][key]
            tt = np.load(os.path.join(path, '%s_tile_table.npy' % key))
            levs[l] = {'h': float(lm['h_m']), 'band': float(lm['band_m']), 'bpt': int(lm['bricks_per_tile_axis']),
                       'tile_table': tt}
            bricks[l] = np.load(os.path.join(path, '%s_sdf.npy' % key), mmap_mode='r')
            grid[..., l] = tg[..., l]
        o = np.array(meta['tile_grid_origin'], float)
        dims = tg.shape[:3]
        tm = float(meta['tile_m'])
        lo = o.copy()
        hi = o + np.array(dims) * tm
        v = Volume(name, o, tm, dims, levs, lo, hi, albedo)
        v.grid = grid
        v.store_path = path
        lc = max(levels)
        self.far_field(v, lc, np.load(os.path.join(path, 'L%d_brick_ijk.npy' % lc)))
        self.add_volume(v, bricks)
        return v

    # ---- frames -------------------------------------------------------------------------------------------
    def frame_buffers(self, W, H):
        cp = self.cp
        if getattr(self, '_fb_wh', None) != (W, H):
            self.d_accum = cp.zeros((H, W, 4), cp.float32)
            self.d_depth = cp.zeros((H, W), cp.float32)
            self.d_obj = cp.zeros((H, W), cp.int32)
            self.d_lev = cp.zeros((H, W), cp.int32)
            self.d_fhit = cp.zeros((H, W), cp.float32)
            self.d_out = cp.zeros((H, W), cp.uint32)
            self.d_gp = cp.zeros((H, W, 4), cp.float32)
            self.d_gn = cp.zeros((H, W, 4), cp.float32)
            self.d_gvis = cp.zeros((H, W), cp.float32)
            self.d_skyc = cp.zeros((H, W, 4), cp.float32)
            self.d_aob = cp.zeros((H, W, 2), cp.float32)
            self.d_rmask = cp.zeros((H, W), cp.uint8)
            self.d_pcount = cp.zeros((H, W), cp.int32)
            self.h_out = None
            self._fb_wh = (W, H)

    TARGET = 16            # passes per pixel at convergence
    AA_PASSES = 4          # passes 0-3 re-trace at the 4 rotated-grid sub-pixel positions; later passes only light

    def cam_params(self, cam, W, H, max_steps=320, lod_bias=1.0):
        f, r, u = cam.basis()
        P = np.zeros(24, np.float32)
        P[0:3] = cam.pos
        P[3:6] = f
        P[6:9] = r
        P[9:12] = u
        P[12] = math.tan(math.radians(cam.fov) * 0.5)
        P[13] = W / float(H)
        P[14] = max_steps
        P[15] = 100.0                 # tmax (m)
        P[16] = lod_bias
        # fs-shell 11-11: the orthographic view (render/camera.py): P[17] = the distance to the target, P[18] = how far
        # before the camera the parallel rays start; 0 = perspective (trace.cu's render kernel)
        ov = cam.ortho_vec() if hasattr(cam, 'ortho_vec') else (0.0, 0.0, 0.0)
        if ov[0] > 0.5:
            P[17], P[18] = ov[1], ov[2]
        # fs-shell 11-11 (design 4.4): the section plane (render/camera.py Camera.section): P[19..21] = n, P[22] = d,
        # P[23] = 1; zeros without one (trace.cu's render kernel then runs as before)
        if getattr(cam, 'section', None) is not None and hasattr(cam, 'section_vec'):
            P[19:24] = cam.section_vec()
        # fs-shell 11-11 (design 4.8): hidden boxes (cam.clip_planes, set on the frame's camera copy by gpu/viewport.py
        # from the scene's view state): P[23] bit 2, P[24] = the count, then 24 floats each; none: 24 values as before
        cp_ = getattr(cam, 'clip_planes', None)
        if cp_ is not None and len(cp_):
            B_ = np.asarray(cp_, np.float32).reshape(-1, 24)
            P = np.concatenate([P, np.array([len(B_)], np.float32), B_.ravel()])
            P[23] = float(int(round(float(P[23]))) | 2)
        return P

    def light_params(self, cam, sky_n=2, exposure=None, sun_dir=None, sun_rgb=None, sun_radius_deg=SUN_RADIUS_DEG,
                     headlight=0.0, cavity=1.0, ev=0.0, rig=None):
        """headlight / cavity / ev: the sculpting visibility aid (11-08 night); the defaults are the review light.
        rig (fs-light L1, 11-11): an engine/light/rig.py Rig; when given, the L array comes from the rig (its default rig
        gives this function's own array bit for bit: engine/light/rig_test.py) and the other arguments are not read."""
        if rig is not None:
            import importlib.util
            import sys
            R = getattr(GpuScene, '_rig_mod', None)
            if R is None:
                spec = importlib.util.spec_from_file_location(
                    'fs_light_rig', os.path.normpath(os.path.join(HERE, '..', '..', '..', 'light', 'rig.py')))
                R = importlib.util.module_from_spec(spec)
                sys.modules['fs_light_rig'] = R            # its dataclasses look their module up there
                spec.loader.exec_module(R)
                GpuScene._rig_mod = R
            hd = getattr(rig.environment, 'hdri', None)        # fs-light 11-11 ~22:20: the rig's map, loaded when it differs
            if hd and os.path.normpath(hd) != os.path.normpath(getattr(self, 'hdri_dir', HDRI_DIR)) and os.path.isfile(os.path.join(hd, 'env.json')):
                self._load_env(hd)
            self.env_rot_deg = float(rig.environment.rotation_deg)
            self.sky_sh()
            self._sky_gain(float(getattr(rig.environment, 'sky_gain', 1.0)))   # a dark studio: the sky's light scaled
            if getattr(self, '_rig_env', None) is None or os.path.normpath(self._rig_env.folder) != os.path.normpath(getattr(self, 'hdri_dir', HDRI_DIR)):
                self._rig_env = R.Env(getattr(self, 'hdri_dir', HDRI_DIR))
            return R.fieldstudio_lp(rig, cam.pos, self._rig_env)
        if getattr(self, 'hdri_dir', HDRI_DIR) != HDRI_DIR:   # fs-light: the view's own key takes the default map's sun and sky
            self._load_env()
        L_, cr, sr = self.sun_world()
        if sun_dir is not None:
            L_ = np.asarray(sun_dir, float) / np.linalg.norm(sun_dir)
        srgb = self.sun_rgb if sun_rgb is None else np.asarray(sun_rgb, float)
        self.sky_sh()
        self._sky_gain(1.0)                                # fs-light: no gain outside a rig (a no-op unless one was applied)
        E = float(np.mean(srgb)) + float(np.mean(self.envj['sky_E_horizontal_rgb']))
        expo = exposure if exposure is not None else math.pi / max(E, 1e-6)   # FV8's rule: k = pi / E (a card's irradiance)
        expo *= 2.0 ** float(ev)
        Lp = np.zeros(24, np.float32)
        Lp[0:3] = L_
        Lp[3:6] = srgb
        Lp[6], Lp[7] = cr, sr
        Lp[8] = expo
        Lp[9] = 1.0                   # background = the environment as seen
        Lp[10] = 0.15                 # wrap
        Lp[11] = 10.0                 # sun shadow reach (m), as the bake's sun_vis
        # penumbra from the light's real angular size: res = k f / t with k = 1 / tan(angular radius). r8: k = 24 was a
        # 2.4 deg light; its penumbra reached past the stored band (8 mm at L3) and was cut where bricks end (the
        # blocky edges of her head's shadow on her chest). The sun's 0.267 deg keeps it inside the band to ~1.7 m.
        Lp[12] = 1.0 / math.tan(math.radians(max(sun_radius_deg, 1e-3)))
        Lp[13] = 0.15                 # sky ray reach (m), as live light p1
        Lp[14] = sky_n
        Lp[15] = 0.06                 # specular level
        Lp[16] = 24.0                 # specular exponent
        Lp[17:20] = cam.pos
        Lp[20] = 64                   # r38: sky-ray step cap (a runtime value: see trace.cu)
        Lp[21] = 256                  # r38: sun-march step cap
        Lp[22] = float(headlight) * float(np.mean(srgb))   # headlight irradiance: a fraction of the sun's
        return Lp

    def render_rect(self, cam, W, H, rect=None, passes=1, first_pass=0, sky_n=2, sync=True, light=None, relight=None,
                    **kw):
        """Passes first_pass .. first_pass + passes - 1 over rect = (x0, y0, w, h) (None: the whole frame).
        Pass 0 traces at the pixel centre (depth, object, level written), passes 1-3 re-trace at sub-pixel positions,
        every pass lights the G-buffer and accumulates (pass 0 restarts the accumulation)."""
        cp = self.cp
        self._tables()
        self.frame_buffers(W, H)
        x0, y0, rw, rh = rect if rect is not None else (0, 0, W, H)
        P = cp.asarray(self.cam_params(cam, W, H, **kw))
        Lp = cp.asarray(self.light_params(cam, sky_n=sky_n, **(light or {})))
        self.sky_setup(cam, H)
        bs = (16, 8, 1)
        gs = ((rw + 15) // 16, (rh + 7) // 8, 1)
        geo = (np.int32(len(self.vols)), self.d_vft, self.d_vit, self.d_grid, self.d_ttab, self.sdf)
        box = (np.int32(W), np.int32(H), np.int32(x0), np.int32(y0), np.int32(rw), np.int32(rh))
        # relight = (trace rect, [(lo, hi) world boxes]): inside `rect` re-trace and re-light only the pixels the boxes
        # can change (in the trace rect, within the sky reach, or a sun ray through a box); the set is decided at the
        # job's pass 0 and kept for its later passes (r7: re-deciding per pass dropped jittered samples at edges).
        # A relight job's passes must start at 0.
        if relight is not None:
            q, bxs = relight
            qx0, qy0, qx1, qy1 = q[0], q[1], q[0] + q[2], q[1] + q[3]
            bb = np.array([np.concatenate([lo, hi]) for lo, hi in bxs], np.float32).reshape(-1)
            d_b = cp.asarray(bb)
            rel = (np.int32(len(bxs)), d_b, np.int32(qx0), np.int32(qy0), np.int32(qx1), np.int32(qy1))
            if first_pass == 0:
                self.sky_invalidate(bxs)          # n3: the change's cells re-converge
        else:
            rel = (np.int32(0), self.d_sh, np.int32(0), np.int32(0), np.int32(0), np.int32(0))
        k_trace, k_light = self.k_trace, self.k_light

        def issue(tx, ty, tw, th):        # fs-shell 11-13: one tile's passes (the launch guard sizes and times the tiles)
            gs_ = ((tw + 15) // 16, (th + 7) // 8, 1)
            box_ = (np.int32(W), np.int32(H), np.int32(tx), np.int32(ty), np.int32(tw), np.int32(th))
            for p in range(first_pass, first_pass + passes):
                if p < self.AA_PASSES:
                    k_trace(gs_, bs, geo + (P,) + box_ + (np.int32(p), self.d_gp, self.d_gn, self.d_depth, self.d_obj,
                                                          self.d_lev, self.d_fhit, Lp) + rel +
                            (self.d_rmask, self.d_pcount, np.int32(self.AA_PASSES)))
                k_light(gs_, bs, geo + (self.d_env, np.int32(self.env_w), np.int32(self.env_h), Lp, self.d_sh) + box_ +
                        (np.int32(p), self.d_gp, self.d_gn, self.d_accum, self.d_gvis, np.int32(self.AA_PASSES), self.d_wbox,
                         self.d_skyc, self.d_aob, self.d_rmask, self.d_pcount, np.int32(self.TARGET)) + rel)
        self.guard.run(self._guard_kind('rect%d' % int(first_pass < self.AA_PASSES)), W, H, (x0, y0, rw, rh), issue,
                       units=passes, gen=self._guard_gen())
        self.k_resolve(gs, bs, (self.d_accum, self.d_out, np.int32(W), np.int32(H), np.int32(x0), np.int32(y0),
                                np.int32(rw), np.int32(rh), self.d_skyc, self.d_aob, self.d_depth, self.d_gn,
                                np.float32((light or {}).get('cavity', 1.0))))
        if sync:
            cp.cuda.Stream.null.synchronize()

    # ---- n3: the sky cells (fs-brush-4 11-13) ----------------------------------------------------------------------
    def sky_setup(self, cam, H):
        """the cell table on the card (once) and the per-view pixel angle; no-op when FSB_SKY_CACHE is off"""
        if not SKY_CACHE:
            return
        cp = self.cp
        if getattr(self, 'd_skyk', None) is None:
            n = 1 << SKY_LOG2N
            self.d_skyk = cp.zeros(n, cp.uint64)
            self.d_skyv = cp.zeros(2 * n, cp.float32)
            for nm, arr in (('g_sky_k', self.d_skyk), ('g_sky_v', self.d_skyv)):
                gp = self.mod.get_global(nm)
                cp.ndarray((1,), cp.uint64, gp)[...] = arr.data.ptr
            gm = self.mod.get_global('g_sky_mask')
            cp.ndarray((1,), cp.uint32, gm)[...] = n - 1
            self.k_sky_inv = self.mod.get_function('sky_invalidate')
            self._sky_view = None
        pa = 2.0 * math.tan(math.radians(float(cam.fov)) * 0.5) / max(float(H), 1.0)
        key = (round(pa, 12), SKY_BASE, SKY_NMAX, SKY_FOOT, SKY_PAD, SKY_PRIOR)
        if key != self._sky_view:
            self._sky_view = key
            gpar = self.mod.get_global('g_sky_par')
            cp.ndarray((8,), cp.float32, gpar)[...] = cp.asarray(np.array([SKY_BASE, SKY_NMAX, SKY_FOOT, pa, SKY_PAD, 0, 0, SKY_PRIOR], np.float32))
        n_ = min(len(self.vols), 64)
        if n_ > getattr(self, '_sky_norg', 0):
            # n3b: each volume's key origin = its world box centre when first seen after a clear (kept fixed: a key never moves
            # while its cells are held; a moved volume clears the cache in _tables)
            wb = self._wboxes().reshape(-1, 6)
            org = getattr(self, '_sky_orgs', np.zeros((64, 3), np.float32))
            org[getattr(self, '_sky_norg', 0):n_] = 0.5 * (wb[getattr(self, '_sky_norg', 0):n_, :3] + wb[getattr(self, '_sky_norg', 0):n_, 3:])
            self._sky_orgs = org; self._sky_norg = n_
            cp.ndarray((192,), cp.float32, self.mod.get_global('g_sky_org'))[...] = cp.asarray(org.reshape(-1).astype(np.float32))

    def sky_clear(self):
        if getattr(self, 'd_skyk', None) is not None:
            self.d_skyk.fill(0); self.d_skyv.fill(0)
        self._sky_norg = 0                       # n3b: the origins are taken again at the next frame

    def sky_invalidate(self, boxes):
        """an edit's world boxes [(lo, hi)]: the cells within them + the pad re-converge"""
        if not SKY_CACHE or getattr(self, 'd_skyk', None) is None or not boxes:
            return
        cp = self.cp
        bb = cp.asarray(np.array([np.concatenate([lo, hi]) for lo, hi in boxes], np.float32).reshape(-1))
        self.k_sky_inv((512,), (256,), (np.int32(len(boxes)), bb, np.float32(SKY_PAD)))

    def _guard_gen(self):
        """fs-shell 11-13: the volume set and what is shown (a change makes the launch guard measure again)"""
        return hash(tuple((id(v), bool(v.visible)) for v in self.vols))

    def _guard_kind(self, base):
        """fs-shell 11-13: the launch guard's cost-map kind: the call and which light kernel holds the slot (the review
        light, a view mode, or another lane's drop-in), as their costs a pixel differ by 10x and more"""
        k = self.k_light
        vm = getattr(self, 'view_modes', None)
        if k is self._k_light0:
            lk = 'review'
        elif vm is not None and k is vm.k:
            lk = 'mode_%s' % vm.mode
        else:
            lk = 'other'
        return '%s:%s' % (base, lk)

    # ---- per-pixel progressive (the app's path, r9) ----------------------------------------------------------------
    def restart(self, cam, W, H, rect=None, boxes=None, trace_rect=None, light=None):
        """Pass count -> 0 over rect (None: all), or only for the pixels the world boxes can change."""
        cp = self.cp
        self._fresh = True                # fs-shell 11-13: the next progress() is a first pass (the guard's kind)
        self._tables()
        self.frame_buffers(W, H)
        self.sky_setup(cam, H)
        if boxes:
            self.sky_invalidate(boxes)
        x0, y0, rw, rh = rect if rect is not None else (0, 0, W, H)
        if rw <= 0 or rh <= 0:
            return
        Lp = cp.asarray(self.light_params(cam, **(light or {})))
        if boxes:
            bb = cp.asarray(np.array([np.concatenate([lo, hi]) for lo, hi in boxes], np.float32).reshape(-1))
            q = trace_rect or (0, 0, 0, 0)
            nb = len(boxes)
        else:
            bb, q, nb = self.d_sh, (0, 0, 0, 0), 0
        gs = ((rw + 15) // 16, (rh + 7) // 8, 1)
        self.k_restart(gs, (16, 8, 1), (np.int32(W), np.int32(H), np.int32(x0), np.int32(y0), np.int32(rw), np.int32(rh),
                                        self.d_gp, self.d_gn, Lp, np.int32(nb), bb, np.int32(q[0]), np.int32(q[1]),
                                        np.int32(q[0] + q[2]), np.int32(q[1] + q[3]), self.d_pcount))

    def progress(self, cam, W, H, rect=None, passes=1, light=None, sync=True, **kw):
        """`passes` per-pixel passes over rect: each pixel below TARGET does its next pass (traced while < AA_PASSES).
        -> the smallest pass count left in the rect (TARGET = converged)."""
        cp = self.cp
        self._tables()
        self.frame_buffers(W, H)
        x0, y0, rw, rh = rect if rect is not None else (0, 0, W, H)
        if rw <= 0 or rh <= 0:
            return self.TARGET
        P = cp.asarray(self.cam_params(cam, W, H, **kw))
        Lp = cp.asarray(self.light_params(cam, **(light or {})))
        bs, gs = (16, 8, 1), ((rw + 15) // 16, (rh + 7) // 8, 1)
        geo = (np.int32(len(self.vols)), self.d_vft, self.d_vit, self.d_grid, self.d_ttab, self.sdf)
        box = (np.int32(W), np.int32(H), np.int32(x0), np.int32(y0), np.int32(rw), np.int32(rh))
        rel = (np.int32(0), self.d_sh, np.int32(0), np.int32(0), np.int32(0), np.int32(0))
        self.sky_setup(cam, H)
        prof = getattr(self, 'prof', None)
        ev = [cp.cuda.Event() for _ in range(4)] if prof is not None else None
        k_trace, k_light = self.k_trace, self.k_light

        def issue(tx, ty, tw, th):        # fs-shell 11-13: one tile's passes (the launch guard sizes and times the tiles)
            gs_ = ((tw + 15) // 16, (th + 7) // 8, 1)
            box_ = (np.int32(W), np.int32(H), np.int32(tx), np.int32(ty), np.int32(tw), np.int32(th))
            for _ in range(passes):
                k_trace(gs_, bs, geo + (P,) + box_ + (np.int32(-1), self.d_gp, self.d_gn, self.d_depth, self.d_obj,
                                                      self.d_lev, self.d_fhit, Lp) + rel +
                        (self.d_rmask, self.d_pcount, np.int32(self.AA_PASSES)))
                k_light(gs_, bs, geo + (self.d_env, np.int32(self.env_w), np.int32(self.env_h), Lp, self.d_sh) + box_ +
                        (np.int32(-1), self.d_gp, self.d_gn, self.d_accum, self.d_gvis, np.int32(self.AA_PASSES),
                         self.d_wbox, self.d_skyc, self.d_aob, self.d_rmask, self.d_pcount, np.int32(self.TARGET)) + rel)
        if ev:
            ev[0].record()
        # the pass state matters for the cost (a pass right after a restart traces every pixel; later ones light only):
        # the kind separates the map of a frame's first pass from its refine passes
        st = 'p0' if getattr(self, '_fresh', False) else 'pn'
        self._fresh = False
        self.guard.run(self._guard_kind('prog_' + st), W, H, (x0, y0, rw, rh), issue, units=passes, gen=self._guard_gen())
        if ev:
            ev[1].record(); ev[2].record()
        self.k_resolve(gs, bs, (self.d_accum, self.d_out, np.int32(W), np.int32(H), np.int32(x0), np.int32(y0),
                                np.int32(rw), np.int32(rh), self.d_skyc, self.d_aob, self.d_depth, self.d_gn,
                                np.float32((light or {}).get('cavity', 1.0))))
        if ev:
            ev[3].record(); ev[3].synchronize()
            prof.append((cp.cuda.get_elapsed_time(ev[0], ev[1]), cp.cuda.get_elapsed_time(ev[1], ev[2]),
                         cp.cuda.get_elapsed_time(ev[2], ev[3]), rw * rh))
        m = int(self.d_pcount[y0:y0 + rh, x0:x0 + rw].min().get())
        if sync:
            cp.cuda.Stream.null.synchronize()
        return m

    def pick(self, orig, dirs, exclude=(), only=None):
        """r12: (n, 3) world rays -> t (n,) (-1 miss), volume index (n,), normal (n, 3), on the card's field.
        exclude: object ids the rays pass through (11-08 night: brushes pick through locked protected parts)"""
        cp = self.cp
        self._tables()
        vft = self.d_vft
        ex = [i for i, v in enumerate(self.vols) if getattr(v, 'oid', None) in set(exclude)]
        if ex:
            vft = self.d_vft.copy()
            vft[np.array(ex), VF_VIS] = 0.0           # d_vft is (nvol, VF)
        if only is not None:                          # 11-09: a pick of these objects alone (shown or not)
            vft = vft.copy() if vft is self.d_vft else vft
            sel = np.array([getattr(v, 'oid', None) in set(only) for v in self.vols])
            vft[:, VF_VIS] = 0.0
            if sel.any():
                vft[np.nonzero(sel)[0], VF_VIS] = 1.0
        o = cp.asarray(np.ascontiguousarray(orig, np.float32).reshape(-1))
        d = cp.asarray(np.ascontiguousarray(dirs, np.float32).reshape(-1))
        n = len(o) // 3
        t = cp.empty(n, cp.float32)
        v = cp.empty(n, cp.int32)
        nn = cp.zeros(3 * n, cp.float32)
        self.k_pick(((n + 63) // 64,), (64,), (np.int32(len(self.vols)), vft, self.d_vit, self.d_grid, self.d_ttab,
                                               self.sdf, o, d, np.int32(n), t, v, nn))
        return t.get(), v.get(), nn.get().reshape(n, 3)

    # ---- fs-brush 11-13: the move's live view (trace.cu 'THE MOVE'S LIVE VIEW'; gpu/sculpt.py MOVE_VIEW) ----------------
    MV_H = 64
    MV_MAXC = 4
    MV_NL = int(os.environ.get('FSB_MV_NL', '24'))           # bound-grid cells per axis per copy
    MV_MARGIN = float(os.environ.get('FSB_MV_MARGIN', '1.3'))   # the bound's margin over the sampled stretch

    def _mv_set_int(self, name, v):
        cp = self.cp
        cp.ndarray((1,), cp.int32, self.mod.get_global(name))[...] = int(v)

    def _mv_set_ptr(self, name, ptrs):
        cp = self.cp
        a = np.asarray(ptrs, np.uint64).reshape(-1)
        cp.ndarray((len(a),), cp.uint64, self.mod.get_global(name))[...] = cp.asarray(a)

    def mv_set(self, vidx, copies, lut):
        """An anchored move shown through the tracer: copies = [(bp float32[51], scope grid (uint8 3-D) or None), ...] in
        the order the dabs would be written, on volume vidx; lut = the stroke's falloff table. The volume's box (VF_LO /
        VF_HI and its world box) grows to hold every copy's ball, in place on the card. -> the copies' balls (local)"""
        cp = self.cp
        n = len(copies)
        if n == 0 or n > self.MV_MAXC or self.k_mvl is None:
            raise RuntimeError('move view: %d copies' % n)
        self._tables()
        NL = self.MV_NL
        H = np.zeros((self.MV_MAXC, self.MV_H), np.float32)
        mgs = [0] * self.MV_MAXC
        keep = []
        balls = []
        for c, (bp, mg) in enumerate(copies):
            bp = np.asarray(bp, np.float32)
            H[c, :len(bp)] = bp
            ctr, r = bp[1:4].astype(np.float64), float(bp[7])
            balls.append((ctr - r, ctr + r))
            cs = 2.0 * r / NL
            H[c, 52:55] = ctr - r
            H[c, 55] = cs
            H[c, 56] = NL
            H[c, 57] = c * NL ** 3
            if mg is not None:
                d_mg = cp.asarray(np.ascontiguousarray(mg, np.uint8).reshape(-1))
                keep.append(d_mg)
                mgs[c] = d_mg.data.ptr
                H[c, 58:61] = mg.shape
        st = self._mv if self._mv is not None else {}
        if st.get('L') is None or st['L'].size < n * NL ** 3:
            st['L'] = cp.empty(self.MV_MAXC * NL ** 3, cp.float32)
            st['raw'] = cp.empty(NL ** 3, cp.float32)
        st['hdr'] = cp.asarray(H.reshape(-1))
        st['lut'] = cp.asarray(np.asarray(lut, np.float32))
        st['mg'] = keep
        st['vidx'] = int(vidx)
        st['n'] = n
        self._mv = st
        self._mv_set_ptr('g_mv_hdr', [st['hdr'].data.ptr])
        self._mv_set_ptr('g_mv_L', [st['L'].data.ptr])
        self._mv_set_ptr('g_mv_lut', [st['lut'].data.ptr])
        self._mv_set_int('g_mv_nlut', len(lut))
        self._mv_set_ptr('g_mv_mg', mgs)
        gk = (int(vidx), self.d_vit.data.ptr)          # the tables' grid offset of the volume (one read per upload)
        if st.get('goff_key') != gk:
            st['goff_key'], st['goff'] = gk, int(self.d_vit[vidx, 0].get())
        self._mv_set_int('g_mv_goff', st['goff'])
        nat = [i for i, w in enumerate(self.vols) if getattr(w, 'mask_vol', None) is self.vols[vidx]]
        if nat:                                    # her store object: the native store it masks
            nk = (nat[0], self.d_vit.data.ptr)
            if st.get('nat_key') != nk:
                st['nat_key'], st['nat_goff'] = nk, int(self.d_vit[nat[0], 0].get())
            self._mv_set_int('g_mv_nat', nat[0])
            self._mv_set_int('g_mv_nat_goff', st['nat_goff'])
        else:
            self._mv_set_int('g_mv_nat', -1)
        th = 128
        nb = (NL ** 3 + th - 1) // th
        for c in range(n):
            self.k_mvl((nb,), (th,), (np.int32(c), st['raw']))
            self.k_mvd((nb,), (th,), (np.int32(c), st['raw'], st['L'], np.float32(self.MV_MARGIN)))
        # the volume's box holds the moved surface (in place: the tables' pointers, g_vft included, stay valid)
        v = self.vols[vidx]
        lo = np.minimum.reduce([v.lo] + [b[0] for b in balls])
        hi = np.maximum.reduce([v.hi] + [b[1] for b in balls])
        f = v.ftab()
        f[37:40], f[40:43] = lo, hi
        self.d_vft[vidx] = cp.asarray(f)
        c8 = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = (v.R @ c8.T).T / v.s + v.t
        self.d_wbox[6 * vidx:6 * vidx + 6] = cp.asarray(np.concatenate([w.min(0), w.max(0)]).astype(np.float32))
        self._mv_set_int('g_mv_n', n)
        return balls

    def mv_probe(self, vidx, pw):
        """diagnosis: the tracer's read of volume vidx at world points pw (n, 3) -> (n, 10): f, level, mapped, pre-image
        (local), bound, the stroke-start read at the pre-image, its level, its empty flag"""
        cp = self.cp
        self._tables()
        pw = np.ascontiguousarray(pw, np.float32).reshape(-1, 3)
        n = len(pw)
        out = cp.zeros(10 * n, cp.float32)
        k = self.mod.get_function('mv_probe')
        k(((n + 63) // 64,), (64,), (np.int32(vidx), self.d_vft, self.d_vit, self.d_grid, self.d_ttab, self.sdf,
                                    cp.asarray(pw.reshape(-1)), np.int32(n), out))
        return out.get().reshape(n, 10)

    def mv_clear(self):
        """The move view off (g_mv_n = 0) and the moved volume's box back to its own."""
        st = self._mv
        if st is None:
            return
        self._mv = None
        try:
            self._mv_set_int('g_mv_n', 0)
            vidx = st.get('vidx')
            if vidx is not None and vidx < len(self.vols) and not self._dirty_tables:
                v = self.vols[vidx]
                self.d_vft[vidx] = self.cp.asarray(v.ftab())
                self.d_wbox[...] = self.cp.asarray(self._wboxes())
        except Exception:
            self._dirty_tables = True

    def to_host(self):
        """BGRA8 frame to the host (pinned), (H, W) uint32."""
        cp = self.cp
        if self.h_out is None or self.h_out.shape != self.d_out.shape:
            mem = cp.cuda.alloc_pinned_memory(self.d_out.nbytes)
            self.h_out = np.frombuffer(mem, np.uint32, self.d_out.size).reshape(self.d_out.shape)
        self.d_out.get(out=self.h_out)
        return self.h_out

    def gpu_bytes(self):
        cp = self.cp
        return cp.get_default_memory_pool().total_bytes()
