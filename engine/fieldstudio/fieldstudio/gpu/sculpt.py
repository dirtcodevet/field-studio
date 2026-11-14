"""Field Studio GPU sculpt (integration option A, 2026-11-08): brush dabs on the card's brick atlas.

The CPU core stays the authority for save, hash and undo: after each batch of dabs the CHANGE the GPU made is written
into the object's sculpt layer, delta += F_after - F_before at the lattice nodes the GPU bricks and the sculpt bricks
share (GPU nodes 0..7 of brick g are sculpt brick g - boff's nodes), through the same StrokeRecord the CPU path fills
(snapshots before the first write, new bricks listed), so undo / redo / save work unchanged. Wherever no dab acted
the change is exactly 0, so untouched sculpt values never move. The mirror is then told those sculpt bricks are
current on the card (no re-sampling).

Stays on the CPU path (accepts() says no): masks in use, inverted masks, region (sampled) objects, mask painting.

Determinism: the GPU applies brush.py's rules in fp32 on the atlas field (fp16 storage); a CPU replay of the op log
gives slightly different numbers, so Verify replay / the undo history rebuilt on Open will not match bit for bit for
GPU strokes (the saved state and its hash are what the GPU produced). GPU strokes run no CPU re-distance on release.
"""
import math
import os
import time

import numpy as np

from ..core import brush as BR
from .mirror import LEVEL, BPT, _ukeys, _eval_nodes9

HERE = os.path.dirname(os.path.abspath(__file__))


class GridOverflow(RuntimeError):
    """r40: a dab reaches past its object's GPU volume grid; .oid names the object (only it is rebuilt)."""
    def __init__(self, oid, msg, box=None):
        super().__init__(msg)
        self.oid = oid
        self.box = box          # (lo, hi) local: what the rebuilt grid must cover


PROF = bool(os.environ.get('FSB_PROF'))     # fs-brush 11-11: per-stage card timing (synchronising; benches only)
SCRATCH = os.environ.get('FSB_SCRATCH', '1') != '0'   # fs-brush 11-11: persistent per-role card buffers (_scr)
SCRATCH_KEEP_MB = float(os.environ.get('FSB_SCRATCH_KEEP_MB', '64'))   # fs-brush 11-12 (F5): larger ones go at a stroke's end
TOPO_CHECK = os.environ.get('FSB_TOPO_CHECK', '0') != '0'   # fs-brush 11-11 (design D4): the stroke-end topology check
TOPO_MAX_NODES = int(float(os.environ.get('FSB_TOPO_MAX', '4e6')))   # a box larger than this is not checked (logged)
RD_RELEASE = os.environ.get('FSB_RD_RELEASE', '0') != '0'   # fs-brush 11-11: the surface layer re-distanced once at release
RD_REACH = os.environ.get('FSB_RD_REACH', '1') != '0'   # p50: bit-exact, rows -30 %, rd -25 % (Grab r 80)   # fs-brush 11-11: no-skip brushes re-distance only reachable rows
RD_MODE = os.environ.get('FSB_RD', 'eik')   # fs-brush 11-11: live re-distance: 'eik' (seeds + Godunov, default) | 'jfa'
# fs-brush-2 11-13 (the grain on thin parts, every tool; sculpt.cu rd_eik_apply): the live re-distance moves a node only as
# far as the batch changed it (full at a change of RD_CHG_EPS_H x h); FSB_RD_CHG=0: the sphere depth alone, as before
RD_CHG = os.environ.get('FSB_RD_CHG', '0') == '1'   # fs-brush-3 11-13: OFF by default (fs-brush-2's verdict: no rim effect, weakens big strokes)
RD_CHG_EPS_H = float(os.environ.get('FSB_RD_CHG_EPS_H', '0.1'))
# fs-brush-3 11-13 (the grain): the live re-distance's rebuilt layer seeded from the surface layer's KEPT values
# (sculpt.cu rd_eik_init seed_kept); FSB_RD_SEED=est: the |F| / |grad F| estimate, as before
RD_SEED_KEPT = os.environ.get('FSB_RD_SEED', 'kept') == 'kept'
# fs-brush-3 11-13 (fs-brush-2's defect 2: a fresh object's first big anchored stroke grew the card grid mid-drag, 142-174 ms):
# the grid must hold the MOVED SURFACE, not the brush's ball: the form's own box within the copies' reach, swept by the move
# (|v|) and a band; the ball's empty parts beyond the grid only ever hold far values. FSB_GROW_SURF=0: the balls, as before
GROW_SURF = os.environ.get('FSB_GROW_SURF', '1') != '0'


def _swept_surface_box(obj, lo, hi, move, h):
    """fs-brush-3 11-13: (lo, hi) of the form's box within the box lo..hi, swept by the move and 2 h, or None when the
    form does not reach into it (nothing can move there). move: a vector (a Grab / Stretch moves its points along the
    drag only: the box is swept along it; 11-13, block-out S3's 107 mm pull regrew the grid mid-drag, 918 ms,
    because the box was dilated by the pull in every direction) or a length (every direction)"""
    flo, fhi = obj.local_box()
    s_lo, s_hi = np.maximum(np.asarray(flo, float), lo), np.minimum(np.asarray(fhi, float), hi)
    if np.any(s_lo > s_hi):
        return None
    e = 2.0 * float(h)
    if np.ndim(move) == 0:
        return s_lo - (float(move) + e), s_hi + (float(move) + e)
    v = np.asarray(move, float)
    return s_lo + np.minimum(v, 0.0) - e, s_hi + np.maximum(v, 0.0) + e
BAND_ADMIT = os.environ.get('FSB_BAND_ADMIT', '1') != '0'   # fs-brush 11-11: local brushes admit band cells only
STORE_BAND_ADMIT = os.environ.get('FSB_STORE_BAND_ADMIT', '1') != '0'   # fs-brush 11-12 (F5): her store object too
BATCH_ROWS = int(os.environ.get('FSB_BATCH_ROWS', '40000'))   # fs-brush 11-12 (probe5b): a card batch's estimated band rows
GATHER_ROWS = int(os.environ.get('FSB_GATHER_ROWS', '32768'))   # fs-brush 11-12 (Flatten 40): rows a gather chunk takes (48 MB fp16)
LIGHT_SYNC = os.environ.get('FSB_LIGHT_SYNC', '1') != '0'   # fs-brush 11-12: the release syncs only what the stroke wrote
SMOOTH_RING_REACH = float(os.environ.get('FSB_SMOOTH_RING_REACH', '1'))   # fs-brush 11-12: Smooth holds its rings' reach (x QSM_R2 r)
SHEET_FRONT = os.environ.get('FSB_SHEET_FRONT', '1') != '0'   # fs-brush-4 11-13: the sheet walk in ONE launch (sh_walk_1b); FSB_SHEET_FRONT=0: a launch a step, as before
SHEET_HOST = os.environ.get('FSB_SHEET_HOST', '0') != '0'   # fs-brush 11-12: the sheet labelled ON THE CARD (cs4, L2); FSB_SHEET_HOST=1: on the host (slower, the same rule)
SKIP_SIDE = os.environ.get('FSB_SKIP_SIDE', '0') != '0'   # fs-brush 11-12: the card's node skip one-sided (OFF until proven)
ADMIT_GROW = os.environ.get('SC_ADMIT_GROW', '0') != '0'   # sculpt-contact 11-12 (lane copy C): the band admission (OFF until its window evidence lands)
#                                                            counts the stroke's earlier growth over each cell
# fs-core 11-14 (a sculpting investigation; fs-core/fh.py, the tracer's own reads on a fresh sphere): THE TRUNCATED
# DISTANCE BAND. The card tracer steps by the value it reads and the dabs, the node skip, the row filter and Smooth's rings
# all take a stored value for a distance; nothing kept it one. A dab changes only the nodes in its ball (and skips those
# beyond its reach), the live re-distance rewrote only the dab sphere's own nodes in <= 40 steps, and every other held node
# kept the field from before: after one Clay drag the tracer's read overestimated the distance to its own surface by up to
# 4.5 mm, after a crossing stroke 9 mm, after 10 strokes 18 mm, with steps of 5 mm between rewritten and stale rows at
# brick faces; the next stroke skipped / filtered by those values and the surface stopped at them (walls, terraces,
# fragments; L 3.6 -> 69). Now every batch keeps the band a distance: every held node within 2 W of a dab ball takes part,
# the surface's own nodes stay as they are (frozen seeds: the zero set does not move), a node whose nearest surface may
# have moved (nearer to a ball than min(|F|, W)) takes the converged eikonal distance, and every node is clamped to +-W
# (a stored value never promises more room than there is). FSB_TSDF=0: as before.
TSDF = os.environ.get('FSB_TSDF', '1') != '0'
TSDF_W_H = float(os.environ.get('FSB_TSDF_W', '12'))     # the band W in lattice spacings
# fs-core-2 11-14 (strokes piled on one spot: the surface steepened stroke by stroke, then walls from stroke 6): under the band
# a surface node keeps its value only while it can be a distance: slope <= TSDF_GK and |F| <= TSDF_GK x the axis crossing
# beside it (sculpt.cu seed_fix). FSB_TSDF_GK=0: every surface node kept, as before
TSDF_GK = float(os.environ.get('FSB_TSDF_GK', '1.25'))
# and a node off the surface keeps no more than the distance rebuilt from those nodes (sculpt.cu rd_tsdf_apply), beyond a
# rounding tolerance of TSDF_NB_TOL h. FSB_TSDF_NB_TOL=-1: off
TSDF_NB_TOL = float(os.environ.get('FSB_TSDF_NB_TOL', '0.05'))
SHEET_REACH_H = float(os.environ.get('FSB_SHEET_REACH_H', '4'))   # fs-core 11-14: the sheet walk's reach past r (h); 2 before


def _true_move(p):
    """sculpt-contact 11-12: a dab's largest surface move: its amount, times Swell to contact's gain where it applies"""
    a = abs(float(p['bp'][13]))
    if len(p['bp']) > BR.NBP_SWC + 1 and p['bp'][BR.NBP_SWC] > 0.5:
        a *= max(float(p['bp'][BR.NBP_SWC + 1]), 1.0)
    return a


def _grown(prev, ctr, bm):
    """sculpt-contact 11-12: per cell centre, the summed true move of the stroke's earlier dabs whose ball (+ the cell's
    half-diagonal) reaches it; prev rows (cx, cy, cz, r, move) in the object's local frame"""
    if prev is None or not len(prev) or not len(ctr):
        return 0.0
    lo, hi = ctr.min(0), ctr.max(0)
    rr = prev[:, 3] + 0.8661 * bm
    near = np.all((prev[:, :3] >= lo - rr[:, None]) & (prev[:, :3] <= hi + rr[:, None]), axis=1)
    pv = prev[near]
    if not len(pv):
        return 0.0
    out = np.zeros(len(ctr))
    for i in range(0, len(pv), 256):
        q = pv[i:i + 256]
        d2 = ((ctr[:, None, :] - q[None, :, :3]) ** 2).sum(2)
        out += ((d2 <= (q[None, :, 3] + 0.8661 * bm) ** 2) * q[None, :, 4]).sum(1)
    return out


SMOOTH_RING_BAND = os.environ.get('FSB_SMOOTH_RING_BAND', '0') != '0'   # fs-brush 11-12: her store's band widened for Smooth's rings (to measure; off)
WB_BAND_H = float(os.environ.get('FSB_WB_BAND_H', '0'))   # fs-brush 11-11: write-back band (0 = every changed brick).
# OFF (17:40): a skipped deep brick left the CPU behind the card; the next stroke to flip a sign there wrote back
# after - before (before from the card) onto the stale CPU value, so whole bricks came out with the wrong sign
# (M, head test h13 / h14: brick-shaped flakes and blocks under the chin, 87 -> 13 416 speck voxels)
# fs-brush 11-12 speed (prof1: the anchored Grab's restore was 51 % of its card time, its ball ~7x a live
# Grab dab's bricks): FSB_FLOW_SUPPORT=0 = the flow dab over its bounding ball as before (r + 1.5 L); FSB_FLOW_DEV=0 = the
# restore from host values as before
FLOW_SUPPORT = os.environ.get('FSB_FLOW_SUPPORT', '1') != '0'
FLOW_DEV = os.environ.get('FSB_FLOW_DEV', '1') != '0'
FLOW_ONEBOX = os.environ.get('FSB_FLOW_ONEBOX', '1') != '0'
# fs-brush 11-13 (a block-out at 60-150 mm brushes ran 5-9 fps: an anchored move rewrites the moved VOLUME of the 1 mm
# lattice every batch, (r/h)^3): THE MOVE'S LIVE VIEW. While an anchored Grab / Stretch is dragged, nothing is written to
# the lattice: the tracer reads the moved volume through the move's inverse (trace.cu, GpuScene.mv_set), so a frame costs
# its pixels whatever the radius; the release writes the lattice once (the old path's last batch, the same total).
# FSB_MOVE_VIEW=0: the old per-batch write. FSB_MOVE_VIEW_RH: the smallest brush (in lattice spacings) shown this way.
MOVE_VIEW = os.environ.get('FSB_MOVE_VIEW', '1') != '0'
MOVE_VIEW_RH = float(os.environ.get('FSB_MOVE_VIEW_RH', '0'))
MOVE_VIEW_STORE = os.environ.get('FSB_MOVE_VIEW_STORE', '1') != '0'   # her store object too (trace.cu g_mv_nat)
# fs-brush-2 11-13 (design: changes appear in real time and
# the release is never a pause the user waits through): a stroke shown through the tracer prepares its write while
# it is dragged (MV_PREP: the cells it will write admitted and their first-touch values taken, in time-boxed chunks; the
# stroke-start field does not change during the drag) and WRITES BEHIND THE RELEASE (PW): the stroke stays in the view as
# its formula while the one fine dab and the CPU write-back run as slices between frames (gpu/hooks.py end -> pw_begin;
# ui/engine.py runs pw_step each turn, finishes it before any job that needs the written field).
MV_PREP = os.environ.get('FSB_MV_PREP', '1') != '0'
MV_PREP_MS = float(os.environ.get('FSB_MV_PREP_MS', '5'))       # per drag frame
PW = os.environ.get('FSB_PW', '1') != '0'
PW_SLICE_MS = float(os.environ.get('FSB_PW_SLICE_MS', '10'))    # per engine turn after the release
PW_WB_ROWS = int(os.environ.get('FSB_PW_WB_ROWS', '1500'))      # write-back rows per slice
PREP_CELLS = int(os.environ.get('FSB_PREP_CELLS', '500'))       # cells admitted per chunk (mv8e: 1500 ran 26-46 ms)
PREP_ROWS = int(os.environ.get('FSB_PREP_ROWS', '1500'))        # rows tracked per chunk
PW_DEV_ROWS = int(os.environ.get('FSB_PW_DEV_ROWS', '100000'))  # a stroke's written rows copied on the card up to this many
PW_SWITCH_DIRTY = os.environ.get('FSB_PW_SWITCH_DIRTY', '0') != '0'   # repaint at the switch (the lattice draws what the
                                                                    # formula drew, to ~0.01 mm: mv5; off = no 150-200 ms refine)   # one dirty box per anchored batch (else 3: the restore's twice)
ROW_FILT = os.environ.get('FSB_ROW_FILT', '1') != '0'   # fs-brush 11-11: skip-band brushes leave far bricks untracked
BATCH_GUARD = os.environ.get('FSB_BATCH_GUARD', '1') != '0'   # fs-brush 11-11: the guard once per batch (net change)
TOPO_SWEEPS = 16     # 11-09 night: colour sweeps per dab before the still-refused nodes are clamped (dab_topo_commit)


class GpuSculpt:
    pw_history = []                                # fs-brush-2 11-13: the finished writes' records (tests read them)
    def __init__(self, view):
        self.view = view                 # gpu.viewport.GpuView (its gpu scene and mirror)
        cp = view.gpu.cp
        self.cp = cp
        src = open(os.path.join(HERE, 'sculpt.cu'), encoding='utf-8').read()
        opts = ('--use_fast_math', '-std=c++14') + (('-DQSM_E=%sf' % os.environ['FSB_QSM_E'],) if os.environ.get('FSB_QSM_E') else ()) + ('-DQSM_HV0=%.6ff' % BR.QSM_HV[0], '-DQSM_HV1=%.6ff' % BR.QSM_HV[1], '-DSH_BEHIND=%.6ff' % float(os.environ.get('FSB_SHEET_BEHIND_H', '0.5')))   # fs-brush-3 11-13: Smooth's height-field fade (core/brush.py QSM_HV)
        # fs-tools 11-14: the approved brush tools' rules (toolset/ftools.cuh, warp/warp.cuh) and their codes (brush.T)
        # the included headers' text in the options too: the compile cache is keyed by sculpt.cu's text and the options
        # only, so a change to an included header was not compiled (fs-tools s2/reg2b: the feather missing on the card)
        import hashlib as _hl
        _ih = _hl.sha1(b''.join(open(os.path.join(os.path.dirname(HERE), *f_), 'rb').read() for f_ in
                                (('warp', 'warp.cuh'), ('toolset', 'ftools.cuh')))).hexdigest()[:12]
        opts = opts + ('-I' + os.path.dirname(HERE), '-DFS_INC_SHA=0x%s' % _ih, '-DFT_TYP_SWELL=%d' % BR.T['ft_swell'],
                       '-DFT_TYP_SETTLE=%d' % BR.T['ft_settle'], '-DFT_TYP_FLATTEN=%d' % BR.T['ft_flatten'],
                       '-DFT_TYP_PENCIL=%d' % BR.T['ft_pencil'])     # fs-tools patch 4: Pencil
        self.mod = cp.RawModule(code=src, options=opts)
        self.k_dab = self.mod.get_function('dab_compute')
        self.k_ft = self.mod.get_function('ft_dab')              # fs-tools 11-14: the approved brush tools (BR.FT_TOOLS)
        self.ft_typs = tuple(BR.T[t] for t in BR.FT_ALL)     # fs-tools patch 4: + Pencil (a line tool)
        self.k_area = self.mod.get_function('area_plane')      # fs-brush 11-11
        self.k_minabs = self.mod.get_function('row_minabs')    # fs-brush 11-11
        self.k_minmax = self.mod.get_function('row_minmax')    # fs-brush 11-11 (RD_REACH)
        self.k_restore = self.mod.get_function('rows_restore')  # fs-brush 11-11 (batch guard)
        self.k_rrm = self.mod.get_function('rows_restore_masked')  # fs-brush 11-12 (the sheet under the brush)
        self.k_sh_cells = self.mod.get_function('sh_cells')        # fs-brush 11-12: the sheet labelled on the card
        self.k_sh_step = self.mod.get_function('sh_step')
        self.k_sh_label = self.mod.get_function('sh_label')
        self.k_sh_restore = self.mod.get_function('sh_restore')
        self.k_sh_restore_min = self.mod.get_function('sh_restore_min')   # fs-core 11-14 (the truncated band)
        self.k_sh_seed = self.mod.get_function('sh_seed')
        self.k_sh_fill0 = self.mod.get_function('sh_fill0')
        self.k_sh_seed_set = self.mod.get_function('sh_seed_set')
        self.k_sh_seed_set_f = self.mod.get_function('sh_seed_set_f')
        self.k_sh_walk_1b = self.mod.get_function('sh_walk_1b')      # fs-brush-4 11-13: the whole walk in one launch
        self.k_apron_nb = self.mod.get_function('apron_fill_nb')  # fs-brush 11-11
        self.k_commit = self.mod.get_function('dab_commit')
        self.k_topo = self.mod.get_function('dab_topo_commit')   # 11-09 night: surfaces kept apart (sculpt.cu)
        self.k_topo_list = self.mod.get_function('dab_topo_list')   # 11-10: the sweeps over the flipping nodes only
        self.k_split = self.mod.get_function('dab_commit_split')     # 11-10: commit + list the flips in one pass
        self.k_gather = self.mod.get_function('gather_rows')
        self.k_collide = self.mod.get_function('collide_clamp')
        self.k_apron = self.mod.get_function('apron_fill')
        self.k_redist = self.mod.get_function('redist')
        # r23: the Newton re-distance measured worse (OFF); r41: a jump-flood closest-point transform ('jfa'), off until
        # measured: FIELDSTUDIO_GPU_REDIST = 'newton' | 'jfa' | anything else = off
        self.redist_mode = os.environ.get('FIELDSTUDIO_GPU_REDIST', '0')
        self.redistance = self.redist_mode in ('1', 'newton', 'jfa', 'scale', 'scale2')
        self.k_grad_mag = self.mod.get_function('grad_mag')
        self.k_scale_smooth = self.mod.get_function('scale_smooth')
        self.k_redist_scale = self.mod.get_function('redist_scale')
        for nm in ('jfa_seed', 'jfa_step', 'jfa_apply', 'rd_seed', 'rd_apply', 'rd_eik_init', 'rd_eik_step', 'rd_eik_apply',
                   'rd_tsdf_apply'):
            setattr(self, 'k_' + nm, self.mod.get_function(nm))
        self.stats = {}
        self.history = []
        view.mirror.live_cells = self._live_cells
        self.enabled = True

    # ---- what goes to the GPU ------------------------------------------------------------------------------------
    def accepts(self, scene, oid, settings):
        if not self.enabled:
            return False
        tool = settings.get('tool')
        if tool not in BR.T or tool in ('mask', 'settle'):   # 11-09: settle runs on the CPU path (swell: r2 on the card)
            return False
        if tool == 'swell' and settings.get('swell_toward', True) and not settings.get('invert', False)                 and os.environ.get('SC_SWELL_CPU') == '1':
            return False                           # sculpt-contact: SC_SWELL_CPU=1 keeps Swell to contact on the CPU path
        # fs-brush 11-11 (design 4.8): the Region brush (ids only, no field change) runs on the CPU path; strokes gated by
        # regions run here, the gate folded into the stroke's mask rows (_mask_fill)
        if tool == 'region':
            return False
        o = scene.get(oid)
        if any(n.kind == 'sampled' for n in o.base.walk()):
            return False
        # fs-brush 11-11: masks are applied on the card (mpool, _mask_fill); a mask in the scene no longer sends every
        # stroke to the CPU path (M, fs-trust R9: 100 of 109 strokes on the CPU after one Ctrl+Mask, 0-2 fps)
        return True

    def _mask_in_use(self, scene, oi, settings):
        """the object's mask matters for this stroke: use_mask on and (Mask > Invert, or any non-zero mask node); or the
        stroke is gated by regions (fs-brush 11-11, design 4.8: its gate rides in the same rows)"""
        if settings.get('skip_regions') or settings.get('automask_region') is not None:
            return True
        if not settings.get('use_mask', True):
            return False
        ob = scene.objects[oi]
        if getattr(ob, 'mask_inv', False):
            return True
        key = (id(scene.store.mask), scene.version, scene.store.used)
        c = getattr(self, '_mask_any', {})
        if c.get(oi, (None,))[0] != key:
            sl = list(ob.sculpt.slots.values())
            c[oi] = (key, bool(sl) and scene.store.has('mask') and bool(np.any(scene.store.mask[np.array(sl)] != 0)))
            self._mask_any = c
        return c[oi][1]

    def _mask_fill(self, scene, oi, ov, rows, keys, settings=None):
        """fs-brush 11-11: the card's mask for these rows (first touched in this stroke): each row's 9^3 nodes from the
        CPU mask (its own 8^3 nodes and the neighbours' first planes, as program.grid_sample reads them), 1 - m under
        Mask > Invert, written into mpool (grown with the pool)"""
        cp = self.cp
        o = scene.objects[oi]
        g = o.sculpt
        st = scene.store
        # fs-brush 11-11 (fs-trust: the first version sized the buffer to the whole atlas, 2.7 GB on a test project, and
        # strokes failed out of memory): a row -> slot map (4 B a pool row) and a compact buffer of only this stroke's
        # rows (1,458 B a row), both grown by 1.5x; the map's entries are cleared at the next stroke's start
        need = int(self.view.gpu.used) + 1
        ms = getattr(self, 'mslot', None)
        if ms is None or ms.shape[0] < need:
            m2 = cp.full(max(need + need // 2, 65536), -1, cp.int32)
            if ms is not None:
                m2[:ms.shape[0]] = ms
            self.mslot = ms = m2
        n0 = int(getattr(self, 'mcount', 0))
        n1 = n0 + len(rows)
        mp = getattr(self, 'mpool', None)
        if mp is None or mp.shape[0] < n1:
            m2 = cp.zeros((max(n1 + n1 // 2, 1024), 729), cp.float16)
            if mp is not None and n0:
                m2[:n0] = mp[:n0]
            self.mpool = mp = m2
        self.mcount = n1
        self.mrows = getattr(self, 'mrows', []) + [np.asarray(rows, np.int64)]
        sk = np.asarray(keys, np.int64).reshape(-1, 3) - ov.boff
        M = np.zeros((len(sk), 9, 9, 9), np.float32)
        settings = settings or {}
        skip = settings.get('skip_regions') or ()
        only = settings.get('automask_region')
        gate = bool(skip) or only is not None                 # fs-brush 11-11 (design 4.8): brush.region_gate on the card
        RID = np.zeros((len(sk), 9, 9, 9), np.uint16) if gate else None
        if g.index.size:
            org = np.asarray(g.origin, np.int64)
            shp = np.array(g.index.shape)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        rel = sk + np.array([dx, dy, dz]) - org
                        ok = np.all((rel >= 0) & (rel < shp), 1)
                        sl = np.full(len(sk), -1, np.int64)
                        sl[ok] = g.index[rel[ok, 0], rel[ok, 1], rel[ok, 2]]
                        hv = sl >= 0
                        if not hv.any():
                            continue
                        src = st.mask[sl[hv]]                     # (n, 8, 8, 8)
                        xi = slice(0, 8) if dx == 0 else slice(0, 1)
                        yi = slice(0, 8) if dy == 0 else slice(0, 1)
                        zi = slice(0, 8) if dz == 0 else slice(0, 1)
                        xo = slice(0, 8) if dx == 0 else slice(8, 9)
                        yo = slice(0, 8) if dy == 0 else slice(8, 9)
                        zo = slice(0, 8) if dz == 0 else slice(8, 9)
                        M[np.nonzero(hv)[0], xo, yo, zo] = src[:, xi, yi, zi]
                        if gate:
                            RID[np.nonzero(hv)[0], xo, yo, zo] = st.rid[sl[hv]][:, xi, yi, zi]
        if getattr(o, 'mask_inv', False):
            M = 1.0 - np.clip(M, 0.0, 1.0)
        if gate:
            if not settings.get('use_mask', True):
                M[:] = 0.0
            blocked = np.zeros(RID.shape, bool)
            if skip:
                blocked |= np.isin(RID, np.asarray(list(skip), np.uint16))
            if only is not None:
                blocked |= RID != np.uint16(int(only))
            M = np.maximum(M, blocked.astype(np.float32))
        mp[n0:n1] = cp.asarray(M.reshape(len(rows), 729).astype(np.float16))
        ms[cp.asarray(np.asarray(rows, np.int64))] = cp.arange(n0, n1, dtype=cp.int32)

    def _mask_reset(self):
        """a new stroke: the previous stroke's row -> slot entries cleared (its rows may hold other bricks now)"""
        if getattr(self, 'mrows', None):
            self.mslot[self.cp.asarray(np.concatenate(self.mrows))] = -1
        self.mrows = []
        self.mcount = 0

    # ---- a batch of samples (the engine's extend_stroke) ---------------------------------------------------------
    def apply(self, scene, oid, settings, samples, rec, flush=False):
        """fs-brush 11-12 (fs-bench-full probe5b, p71: on her every body brush of r >= 30 mm ran the card out of memory: one
        batch took the whole queued drag, dozens of dabs x ~25 k rows each, x 2 under Mirror X, and the keep-apart guard's
        index alone asked 2.15 GB). The samples go to the card in sub-batches whose estimated band rows stay under
        BATCH_ROWS, as if more frames had come; each sub-batch is _apply_batch as before. -> world dirty boxes"""
        anch_ = ((settings.get('tool') == 'grab' and settings.get('grab_flow', False) and not settings.get('snake', False))
                 or (settings.get('tool') == 'stretch' and settings.get('stretch_flow', False)))
        if self._pw_other(rec) and not (anch_ and not flush and self._mv_usable(scene, oid, settings)):
            # fs-brush-2 11-13: only a stroke shown through the tracer (it writes nothing until its release) may run over
            # an earlier stroke's write behind its release; anything that writes the lattice lands it first
            self.pw_complete()
        if ((settings.get('tool') == 'grab' and settings.get('grab_flow', False)) or
                (settings.get('tool') == 'stretch' and settings.get('stretch_flow', False))):
            return self._apply_flow(scene, oid, settings, samples, rec, flush=flush)
        if settings.get('tool') == 'grab' and settings.get('snake', False) and BR.SNAKE_PATH and BR.SNAKE_COMPOSE:
            return self._apply_snakepath(scene, oid, settings, samples, rec)
        if BATCH_ROWS <= 0 or len(samples) <= 1:
            return self._apply_batch(scene, oid, settings, samples, rec)
        k = max(1, int(BATCH_ROWS // max(self._rows_per_sample(scene, oid, settings), 1.0)))
        if k >= len(samples):
            return self._apply_batch(scene, oid, settings, samples, rec)
        boxes = []
        for i in range(0, len(samples), k):
            boxes += list(self._apply_batch(scene, oid, settings, samples[i:i + k], rec) or [])
        return boxes

    def _apply_flow(self, scene, oid, settings, samples, rec, flush=False):
        """fs-brush 11-12 (toolkit review 4.2): the anchored flow Grab on the card: every row this stroke has touched gets
        its first-touch value back (the stroke-start field), then ONE flow dab with the drag's total so far (sculpt.cu
        typ 14): the grabbed point follows the drag exactly, the result depends only on the total (a wiggle back is the
        start), and a flow never folds. -> world dirty boxes"""
        cp = self.cp
        live = self._live
        if live is None or live.get('rec') is not rec:
            fl = {'c0': None, 'v': np.zeros(3), 'n0': None}
        else:
            fl = live.setdefault('flow', {'c0': None, 'v': np.zeros(3), 'n0': None})
        for smp in samples:
            if fl['c0'] is None:
                fl['c0'] = np.asarray(smp['p'], float)
                fl['n0'] = list(smp.get('n', (0.0, 1.0, 0.0)))
            fl['v'] = fl['v'] + np.asarray(smp.get('v', (0.0, 0.0, 0.0)), float)
        if live is not None and live.get('rec') is rec and live.get('mv'):
            # fs-brush 11-13: the stroke so far was shown through the tracer; nothing on the lattice to restore
            if not flush and self._mv_usable(scene, oid, settings):
                return self._mv_frame(scene, oid, settings, fl, rec)
            return self._mv_flush(scene, oid, settings, fl, rec)
        if (live is None or live.get('rec') is not rec) and not flush and fl['c0'] is not None \
                and self._mv_usable(scene, oid, settings):
            self._mv_live_init(scene, rec)
            return self._mv_frame(scene, oid, settings, fl, rec)
        boxes = []
        _tf0 = time.perf_counter()
        _nrest = 0
        if live is not None and live.get('rec') is rec:
            th = 256
            for oi, rows in live.get('rows', {}).items():
                if not rows:
                    continue
                db = live.get('dbefore', {}).get(oi) if FLOW_DEV else None
                if db:
                    if len(db) > 1:                  # one block: the restore is one launch, the aprons one call
                        db[:] = [(cp.concatenate([b[0] for b in db]), cp.concatenate([b[1] for b in db]),
                                  np.concatenate([b[2] for b in db]))]
                    d_rw, d_F, ks = db[0]
                    _nrest += len(ks)
                    self.k_restore(((len(ks) * 729 + th - 1) // th,), (th,),
                                   (self.view.gpu.sdf, d_rw, np.int32(len(ks)), d_F))
                    ks = ks.reshape(-1, 3)
                else:
                    rl = list(rows.keys())
                    _nrest += len(rl)
                    F0 = self._ft_rows(live, oi, rl).astype(np.float16).reshape(-1)   # fs-brush-3: dict or blocks
                    self.k_restore(((len(rl) * 729 + th - 1) // th,), (th,),
                                   (self.view.gpu.sdf, cp.asarray(np.array(rl, np.int32)), np.int32(len(rl)), cp.asarray(F0)))
                    ks = np.array([rows[r_] for r_ in rl], np.int64).reshape(-1, 3)
                obj = scene.objects[oi]
                ov = self.view.mirror.objs[obj.id]
                self._aprons_card(ov, ks)
                b_ = self._world_box(obj, ov, ks, float(obj.sculpt.h))
                if not FLOW_ONEBOX:
                    self.view.note_dirty([b_])
                boxes.append(b_)
        if fl['c0'] is None:
            return boxes
        got = BR.anchored_sample(scene, oid, settings, fl['c0'], fl['n0'], fl['v'])
        if got is None:
            if self._live is not None:
                self._live['flow'] = fl
            return boxes
        if PROF:
            cp.cuda.Stream.null.synchronize()
        _tf1 = time.perf_counter()
        boxes += list(self._apply_batch(scene, oid, got[0], [got[1]], rec) or [])
        self._live['flow'] = fl
        self.stats['flow_restore_ms'] = round((_tf1 - _tf0) * 1000, 2)   # fs-brush 11-12 speed: the restore's share
        self.stats['flow_rows_restored'] = _nrest
        self.stats['flow_ball_m'] = float(got[0]['radius'])
        if FLOW_ONEBOX and len(boxes) > 1:
            # fs-brush 11-12 speed (p2: the view re-renders each dirty box on its own, and the restore's box was noted
            # twice: copy Grab 'restart' 25 ms a frame vs live 6): the restored rows and the dab overlap, one box covers both
            boxes = [(np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0))]
        return boxes

    # ---- fs-brush 11-13: the move's live view (MOVE_VIEW above; trace.cu) ---------------------------------------------
    def mv_pending(self, rec):
        """an anchored stroke shown through the tracer and not yet written to the lattice"""
        live = self._live
        return live is not None and live.get('rec') is rec and bool(live.get('mv'))

    def _mv_usable(self, scene, oid, settings):
        if not MOVE_VIEW or getattr(self.view.gpu, 'k_mvl', None) is None:
            return False
        if settings.get('sym_world', False):
            return False
        oi = scene.index_of(oid)
        o = scene.objects[oi]
        ov = self.view.mirror.objs.get(o.id)
        if ov is None or ov.vol is None or (getattr(ov, 'is_store', False) and not MOVE_VIEW_STORE):
            return False
        if self._mask_in_use(scene, oi, settings):
            return False
        r_l = float(settings.get('radius', 0.0)) / max(float(o.transform['scale']), 1e-9)
        return r_l >= MOVE_VIEW_RH * float(o.sculpt.h)

    def _mv_frame(self, scene, oid, settings, fl, rec):
        """one frame of an anchored drag: the move's copies to the tracer, nothing written. -> world dirty boxes (this
        frame's balls and the last frame's)"""
        live = self._live
        live['flow'] = fl
        prev = live.get('mv_boxes') or []
        G = self.view.gpu
        t0 = time.perf_counter()
        # fs-brush-2 11-13: while an earlier stroke's CPU write-back is behind its release, the CPU field there is old: the
        # pivot depth and Grab's scope (CPU reads) take their provisional values until it lands (core/brush.py CPU_STALE)
        BR.CPU_STALE = self._pw_other(rec)
        got = BR.anchored_sample(scene, oid, settings, fl['c0'], fl['n0'], fl['v'])
        if got is None:
            G.mv_clear()
            live['mv_boxes'] = []
            return list(prev)
        t1 = time.perf_counter()
        oi0 = scene.index_of(oid)
        copies = []
        vol = None
        grown = 0.0
        exp_ = list(self._expand(scene, oid, got[0], [got[1]]))
        o0 = scene.objects[oi0]
        ov0 = self.view.mirror.objs.get(o0.id)
        if ov0 is not None and ov0.vol is not None and exp_:
            # the moved surface must lie on the card's grid at the release: grown now, mid-drag, where nothing has been
            # written (the old path overflowed mid-stroke and rebuilt, 0.74-0.8 s; here it would have been at the
            # release), with a brush radius to spare so a longer drag does not grow it again
            v_ = ov0.vol
            bm_ = 8.0 * float(o0.sculpt.h)
            g_lo = np.asarray(v_.origin, float)
            g_hi = g_lo + np.asarray(v_.dims, float) * BPT * bm_
            lo_ = np.min([np.asarray(c_, float) - float(st_['radius_local']) for _, st_, c_, _, _ in exp_], 0)
            hi_ = np.max([np.asarray(c_, float) + float(st_['radius_local']) for _, st_, c_, _, _ in exp_], 0)
            need_ = np.any(lo_ - bm_ < g_lo) or np.any(hi_ + bm_ > g_hi)
            if need_ and GROW_SURF:
                mv_ = max(float(np.linalg.norm(np.asarray(v__, float))) for _, _, _, _, v__ in exp_)
                sbs_ = [_swept_surface_box(o0, lo_ - mv_, hi_ + mv_, np.asarray(v__, float), o0.sculpt.h) for _, _, _, _, v__ in exp_]
                sbs_ = [b_ for b_ in sbs_ if b_ is not None]
                sb_ = (np.min([b_[0] for b_ in sbs_], 0), np.max([b_[1] for b_ in sbs_], 0)) if sbs_ else None
                need_ = sb_ is not None and (np.any(sb_[0] - bm_ < g_lo) or np.any(sb_[1] + bm_ > g_hi))
                if need_:
                    lo_, hi_ = np.minimum(lo_, sb_[0]), np.maximum(hi_, sb_[1])
            if need_:
                if self._pw_other(rec):
                    self.pw_complete()             # the rebuild re-mirrors from the CPU: an earlier write lands first
                _tg = time.perf_counter()
                sp_ = float(settings.get('radius', 0.0)) / max(float(o0.transform['scale']), 1e-9)
                self.view.mirror.grow_hint[o0.id] = (lo_ - sp_, hi_ + sp_)
                ov0.base_sig = None
                G.mv_clear()
                self.view._sync(scene)
                self._mv_prep_reset(live)             # the rebuild re-allocated every row: what was prepared is void
                self.rebuilds = getattr(self, 'rebuilds', 0) + 1
                grown = (time.perf_counter() - _tg) * 1000
                prev = prev + [(np.full(3, -1e3), np.full(3, 1e3))]     # the whole view: the volume changed
        t2 = time.perf_counter()
        for oi, st, c, n, v in exp_:
            if oi != oi0 or st['tool'] not in ('grabflow', 'stretchflow'):
                raise RuntimeError('move view: a copy on another object or tool (%s)' % st['tool'])
            p = self._plan(scene, oi, st, c, n, v, bp_only=True)
            if p is None:
                continue
            vol = p['vol']
            copies.append((p['bp'], p['mg']))
        if not copies:
            G.mv_clear()
            live['mv_boxes'] = []
            return list(prev)
        lut = BR.falloff_lut(settings.get('falloff', 'smooth'), settings.get('curve')).astype(np.float32)
        t_pp = time.perf_counter()
        if MV_PREP and live.get('prep_ok', True):
            try:
                self._mv_prep(scene, live, exp_, time.perf_counter() + MV_PREP_MS / 1000.0)
            except GridOverflow:
                live['prep_key'] = None
            except Exception as e_:                # the preparation must never cost the stroke
                live['prep_ok'] = False
                live['prep_error'] = '%s: %s' % (type(e_).__name__, e_)
                self._mv_prep_reset(live)
        t3 = time.perf_counter()
        balls = G.mv_set(G.vols.index(vol), copies, lut)
        if PROF:
            self.cp.cuda.Stream.null.synchronize()
        t4 = time.perf_counter()
        of = scene.pack().of_tab[oi0]
        boxes = [self.view.mirror._world_box(of, (lo, hi)) for lo, hi in balls]
        live['mv_boxes'] = boxes
        live['mv_n'] = live.get('mv_n', 0) + 1
        self.stats = {'mv': len(copies), 'mv_ms': round((time.perf_counter() - t0) * 1000, 2), 'dabs': 0, 'bricks': 0,
                      'plan_ms': 0.0, 'gpu_ms': 0.0, 'rd_ms': 0.0, 'mv_anch_ms': round((t1 - t0) * 1000, 2),
                      'mv_grow_ms': round(grown, 1), 'mv_plan_ms': round((t_pp - t2) * 1000, 2),
                      'mv_prep_ms': round((t3 - t_pp) * 1000, 2), 'mv_set_ms': round((t4 - t3) * 1000, 2),
                      'mv_prep_rows': sum(len(r_) for r_ in live.get('rows', {}).values())}
        out = list(prev) + boxes
        if FLOW_ONEBOX and len(out) > 1:
            out = [(np.min([b[0] for b in out], 0), np.max([b[1] for b in out], 0))]
        return out

    def _mv_flush(self, scene, oid, settings, fl, rec):
        # fs-brush-3 11-13: GROW_SURF's plan clipping holds only inside the live view's own prep / flush
        self._mv_clip = getattr(self, '_mv_clip', 0) + 1
        try:
            return self._mv_flush_body(scene, oid, settings, fl, rec)
        finally:
            self._mv_clip -= 1

    def _mv_flush_body(self, scene, oid, settings, fl, rec):
        """the release of a stroke shown through the tracer: the lattice written once with the drag's total (as the
        old path's last batch), the tracer back to the lattice. The card state the drag prepared (_mv_live_init /
        _mv_prep: admitted cells, first-touch rows) is the batch's own. -> world dirty boxes"""
        live = self._live
        prev = list(live.get('mv_boxes') or []) if live is not None else []
        self.view.gpu.mv_clear()
        if live is None or live.get('rec') is not rec or not live.get('prepped'):
            self._live = None                     # the batch starts the stroke's card state (key0, masks, rows)
        t0 = time.perf_counter()
        boxes = list(prev)
        if fl.get('c0') is not None:
            got = BR.anchored_sample(scene, oid, settings, fl['c0'], fl['n0'], fl['v'])
            if got is not None:
                boxes += list(self._apply_batch(scene, oid, got[0], [got[1]], rec) or [])
        if self._live is not None:
            self._live['flow'] = fl
        self.stats['mv_flush_ms'] = round((time.perf_counter() - t0) * 1000, 2)
        if len(boxes) > 1:
            boxes = [(np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0))]
        return boxes

    # ---- fs-brush-2 11-13: preparing the write during the drag, writing behind the release (MV_PREP / PW above) -------
    def _mv_live_init(self, scene, rec):
        """a stroke shown through the tracer gets the card state _apply_batch would give its first batch (its scene key,
        masks reset, its result), so the cells and first-touch rows prepared during the drag are the release batch's own"""
        kn_ = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        if (self.view.changed and kn_ != getattr(self.view, '_sync_key', None)
                and not getattr(self.view, 'hold_sync', False)):
            self.view._sync(scene)                # the atlas holds the stroke-start field before the view reads it
        live = self._live = {'rec': rec, 'rows': {}, 'before': {}, 'tool': 'mvflush', 'mv': True, 'prepped': True}
        live['key0'] = kn_ if kn_ == getattr(self.view, '_sync_key', None) else None
        self._mask_reset()
        if getattr(self, '_result_rec', None) is not rec:
            self._result = {}
            self._result_rec = rec
        return live

    def _mv_prep_reset(self, live):
        for k in ('rows', 'before'):
            live[k] = {}
        for k in ('seen', 'blk', 'ft_index', 'dbefore', 'ftb', 'prep_cells', 'prep_key', 'prep_done'):   # fs-brush-3: + the blocks' index
            live.pop(k, None)

    def _mv_prep(self, scene, live, exp_, deadline):
        # fs-brush-3 11-13: GROW_SURF's plan clipping holds only inside the live view's own prep / flush
        self._mv_clip = getattr(self, '_mv_clip', 0) + 1
        try:
            return self._mv_prep_body(scene, live, exp_, deadline)
        finally:
            self._mv_clip -= 1

    def _mv_prep_body(self, scene, live, exp_, deadline):
        """admit the cells the stroke's write will touch and take their first-touch values (the stroke-start field: nothing
        is written during the drag), in chunks until the deadline. -> True when all of them are done"""
        L_ = 0.0
        for _, st_, c_, _, v_ in exp_:
            L_ = max(L_, float(np.linalg.norm(v_)) / max(float(st_['radius_local']), 1e-9))
        key = tuple(np.round(np.concatenate([np.r_[np.asarray(c_, float), float(st_['radius_local'])]
                                             for _, st_, c_, _, _ in exp_]) * 200.0).astype(np.int64).tolist())
        if live.get('prep_key') is None or (live.get('prep_key') != key and live.get('prep_n', 0) % 4 == 0):
            plans = [self._plan(scene, oi, st, c, n, v) for oi, st, c, n, v in exp_]
            cells = {}
            for p in plans:
                if p is not None and len(p['cells']):
                    cells.setdefault(p['oi'], []).append(p['cells'])
            live['prep_cells'] = {oi: _ukeys(np.concatenate(cl)) for oi, cl in cells.items()}
            live['prep_key'] = key
            live['prep_done'] = False
        live['prep_n'] = live.get('prep_n', 0) + 1
        if live.get('prep_done'):
            return True
        admitted = False
        for oi, cells in live['prep_cells'].items():
            ov = self.view.mirror.objs[scene.objects[oi].id]
            need = cells[ov.rowmap[cells[:, 0], cells[:, 1], cells[:, 2]] < 0]
            while len(need):
                self._admit(scene, oi, ov, need[:PREP_CELLS])
                admitted = True
                need = need[PREP_CELLS:]
                if time.perf_counter() >= deadline:
                    break
            if len(need):
                self.view.gpu._tables()
                return False
        if admitted:
            self.view.gpu._tables()
        for oi, cells in live['prep_cells'].items():
            ov = self.view.mirror.objs[scene.objects[oi].id]
            rw = ov.rowmap[cells[:, 0], cells[:, 1], cells[:, 2]]
            rr, kk = rw[rw >= 0].astype(np.int64), cells[rw >= 0]
            if not len(rr):
                continue
            seen = live.setdefault('seen', {}).get(oi)
            need_ = int(rr.max()) + 1
            if seen is None or len(seen) < need_:
                s2 = np.zeros(max(need_, 2 * (len(seen) if seen is not None else 0), 1024), bool)
                if seen is not None:
                    s2[:len(seen)] = seen
                seen = live['seen'][oi] = s2
            nw = ~seen[rr]
            rws, kws = rr[nw], kk[nw]
            for i in range(0, len(rws), PREP_ROWS):
                r_, k_ = rws[i:i + PREP_ROWS], kws[i:i + PREP_ROWS]
                F = self._gather(r_)
                seen[r_] = True
                # (keys as tuples and no per-row 'before' views: 25 k container objects fewer for each gen-2 collection
                # to walk, mv8e saw 56-63 ms ones; the blocks hold the first-touch values, abort reads them)
                live['rows'].setdefault(oi, {}).update(zip(r_.tolist(), map(tuple, k_.tolist())))
                live.setdefault('blk', {}).setdefault(oi, []).append((r_, k_, F))
                if time.perf_counter() >= deadline and i + PREP_ROWS < len(rws):
                    return False
        live['prep_done'] = True
        return True

    def pw_ok(self, rec):
        return PW and self.mv_pending(rec)

    def pw_pending(self):
        return bool(getattr(self, '_pwq', None))

    def _pw_other(self, rec):
        return any(e['cmd'].rec is not rec for e in (getattr(self, '_pwq', None) or []))

    def _pw_flags(self):
        on = bool(getattr(self, '_pwq', None))
        self.view.hold_sync = on                   # the card is ahead of the CPU until the last write-back lands
        BR.CPU_STALE = on

    def pw_begin(self, scene, cmd, on_done):
        """the release of a stroke shown through the tracer: its write joins the queue behind the release (pw_step), the
        stroke staying in the view as its formula until its one fine dab lands. Card phases (preparation, the fine dab, a
        card-side copy of the written rows) run in stroke order; CPU write-backs run in stroke order; a later stroke's card
        phase may run ahead of an earlier one's CPU write-back (which reads its own copy of the rows)"""
        q = self.__dict__.setdefault('_pwq', [])
        e = {'live': self._live, 'result': getattr(self, '_result', {}), 'result_rec': getattr(self, '_result_rec', None),
             'phase': None, 'scene': scene, 'cmd': cmd, 'on_done': on_done, 'chained': bool(q),
             't0': time.perf_counter(), 'slices': 0, 'work_ms': 0.0, 'max_ms': 0.0, 'log': [], 'wb_rows': PW_WB_ROWS}
        e['gen'] = self._pw_gen(scene, cmd, e)
        q.append(e)
        self._live = None                          # the queue entry owns the stroke's card state now
        self._pw_flags()

    def _pw_pick(self):
        q = self._pwq
        for x in q:                                # the first stroke whose fine dab has not landed (in order) ...
            if x['phase'] not in ('switch', 'wb'):
                return x
        return q[0]                                # ... else the oldest one's CPU write-back

    def pw_step(self, budget_ms=None, stop_after=None):
        """slices of the queued writes within budget_ms -> True when none is left"""
        q = getattr(self, '_pwq', None)
        if not q:
            return True
        b = PW_SLICE_MS if budget_ms is None else budget_ms
        t_end = time.perf_counter() + b / 1000.0
        while q:
            e = self._pw_pick()
            # a write-back chunk sized to the turn's budget from the measured time a row (a running mean, 14 us to start)
            e['wb_rows'] = int(min(max(b * 1000.0 / e.get('us_row', 25.0), 200), 6000))
            self._pw_advance(e)
            if stop_after is not None and e['phase'] == stop_after:
                break
            if time.perf_counter() >= t_end:
                break
        self._pw_flags()
        return not q

    def _pw_advance(self, e):
        """one step of entry e's write, its own card state swapped in (a next stroke may be live meanwhile)"""
        cur = (self._live, getattr(self, '_result', {}), getattr(self, '_result_rec', None))
        self._live, self._result, self._result_rec = e['live'], e['result'], e['result_rec']
        ts = time.perf_counter()
        done = False
        try:
            try:
                tag = next(e['gen'])
            except StopIteration:
                tag, done = 'done', True
            except GridOverflow as ex:
                tag, done = 'done', True
                self._pw_overflow(e, ex)
            except Exception:
                self._pw_failed = e
                raise
            dt = (time.perf_counter() - ts) * 1000
            e['phase'] = tag
            e['slices'] += 1
            e['work_ms'] += dt
            e['max_ms'] = max(e['max_ms'], dt)
            if tag == 'wb' and e.get('last_rows'):
                u_ = dt * 1000.0 / e['last_rows']
                e['us_row'] = u_ if 'us_row' not in e else 0.5 * e['us_row'] + 0.5 * u_
            if dt > 20.0:
                wb_ = self.stats.get('wb') if tag == 'wb' else None
                e['log'].append((tag, round(dt, 1), e.get('last_rows') if tag == 'wb' else None,
                                 {k: round(v, 1) for k, v in (wb_ or {}).items() if isinstance(v, float)}))
            if done:
                if e in self._pwq:
                    self._pwq.remove(e)
                self.stats['pw'] = {'total_ms': round((time.perf_counter() - e['t0']) * 1000, 1),
                                    'work_ms': round(e['work_ms'], 1), 'slices': e['slices'],
                                    'max_slice_ms': round(e['max_ms'], 1), 'long_slices': e['log'][:12],
                                    'chained': e['chained']}
                self.pw_history.append(self.stats['pw'])
                e['on_done']()
        finally:
            e['live'], e['result'], e['result_rec'] = self._live, self._result, self._result_rec
            self._live, self._result, self._result_rec = cur
        return done

    def pw_until_switch(self):
        """every queued write up to its fine dab (the lattice then holds those strokes; only CPU write-backs are left), so a
        next stroke shown through the tracer can start over them"""
        while getattr(self, '_pwq', None) and any(x['phase'] not in ('switch', 'wb') for x in self._pwq):
            self.pw_step(1e9, stop_after='switch')

    def pw_complete(self):
        """every queued write, all of it now (before a job that needs the written field)"""
        while getattr(self, '_pwq', None):
            self.pw_step(1e9)

    def pw_abort_failed(self):
        """a write that raised: its stroke's card rows back to their first-touch values, the entry dropped -> rows"""
        e = getattr(self, '_pw_failed', None)
        self._pw_failed = None
        if e is None:
            return 0
        cur = (self._live, getattr(self, '_result', {}), getattr(self, '_result_rec', None))
        self._live = e['live']
        try:
            n = self.abort()
        finally:
            self._live, self._result, self._result_rec = cur
        return n

    def _pw_overflow(self, e, ex):
        """entry e's fine dab reached past the card grid (rare: the drag grows the grid ahead): as gpu/hooks.py does for
        any anchored stroke, the stroke-start rows back, the grid rebuilt around the reach, the whole stroke written at
        once. The rebuild re-mirrors from the CPU, so the writes queued ahead of e land first"""
        for x in list(self._pwq):
            if x is e:
                break
            while x in self._pwq:
                self._pw_advance(x)
        scene, cmd = e['scene'], e['cmd']
        self.abort()
        if getattr(ex, 'oid', None) is not None and getattr(ex, 'box', None) is not None:
            self.view.mirror.grow_hint[ex.oid] = ex.box
        for k_, ov in self.view.mirror.objs.items():
            if getattr(ex, 'oid', None) is None or k_ == ex.oid:
                ov.base_sig = None
        self.view.hold_sync = False
        self.view._sync(scene)
        self.rebuilds = getattr(self, 'rebuilds', 0) + 1
        self.view.note_dirty(self.apply(scene, cmd.oid, cmd.settings, list(cmd.samples), cmd.rec, flush=True))
        self._finish_core(scene, cmd.rec)

    def _pw_gen(self, scene, cmd, e):
        rec = cmd.rec
        live = self._live
        oid, settings = cmd.oid, cmd.settings
        fl = live['flow']
        t0 = time.perf_counter()
        # fs-brush-3 11-13: the stroke's OWN queued write does not make its stroke-start CPU field stale (only an earlier
        # stroke's pending write-back does): _pw_flags set CPU_STALE for any queue, so the release read no scope / pivot
        BR.CPU_STALE = self._pw_other(rec)
        got = BR.anchored_sample(scene, oid, settings, fl['c0'], fl['n0'], fl['v']) if fl.get('c0') is not None else None
        if got is not None and MV_PREP and live.get('prep_ok', True):
            exp_ = list(self._expand(scene, oid, got[0], [got[1]]))
            live['prep_key'] = None              # the final total's cells
            while not self._mv_prep(scene, live, exp_, time.perf_counter() + PW_SLICE_MS / 1000.0):
                yield 'prep'
        t1 = time.perf_counter()
        BR.CPU_STALE = self._pw_other(rec)        # fs-brush-3 11-13: as above (another turn may have set it meanwhile)
        boxes = self._mv_flush(scene, oid, settings, fl, rec)
        self._pw_flags()
        if PW_SWITCH_DIRTY:
            self.view.note_dirty(boxes)
        self.stats['pw_prep_ms'] = round((t1 - t0) * 1000, 1)
        yield 'flushed'
        snap = self._pw_snapshot(scene, rec)
        yield 'switch'
        # the sculpt index grown once over the stroke's whole box (chunk by chunk it grew piecewise: the first write-back
        # chunk took 91-143 ms in mv8c)
        n_all = 0
        for oi, rows, ks, Fb, vals in snap:
            o_ = scene.objects[oi]
            sk_ = ks - self.view.mirror.objs[o_.id].boff
            n_all += len(sk_)
            if len(sk_) and o_.sculpt.ensure_index(sk_.min(0), sk_.max(0)):
                scene.touch(index=True)
        st_ = scene.store
        if n_all and hasattr(st_, '_grow'):
            st_._grow(st_.used + n_all)            # the store's room once (mv8e: the first chunk's alloc 55 ms)
        yield 'wb'
        yield from self._finish_gen(scene, rec, e, snap)

    def _pw_snapshot(self, scene, rec):
        """the stroke's written rows and their values copied on the card (or gathered to the host when very many), so its
        CPU write-back reads them while later strokes may already write the lattice -> [(oi, rows, keys, Fb, values)]"""
        cp = self.cp
        live = self._live
        out = []
        if live is None or live.get('rec') is not rec:
            return out
        mirror = self.view.mirror
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            ov = mirror.objs[scene.objects[oi].id]
            blk = live.get('blk', {}).get(oi)
            if blk and sum(len(b[0]) for b in blk) == len(rk):
                rows = np.concatenate([b[0] for b in blk])
                ks = np.concatenate([b[1] for b in blk]).reshape(-1, 3)
                Fb = np.concatenate([b[2] for b in blk])
            else:
                rows = np.array(list(rk.keys()), np.int64)
                ks = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
                Fb = np.stack([live['before'][oi][r] for r in rows.tolist()])
            now = ov.rowmap[ks[:, 0], ks[:, 1], ks[:, 2]]
            if np.any(now != rows):
                self.stats['row_alias'] = int(np.sum(now != rows))
                keep = now == rows
                rows, ks, Fb = rows[keep], ks[keep], Fb[keep]
            n = len(rows)
            if n <= PW_DEV_ROWS:
                vals = cp.empty(n * 729, cp.float16)
                th = 256
                if n:
                    self.k_gather(((n * 729 + th - 1) // th,), (th,),
                                  (self.view.gpu.sdf, cp.asarray(rows.astype(np.int32)), np.int32(n), vals))
            else:
                vals = self._gather(rows)
            out.append((oi, rows, ks, Fb, vals))
        return out

    def _finish_gen(self, scene, rec, e=None, snap=None):
        """_finish_core in chunks (the release's write-back into the CPU sculpt layer), from the rows' card-side copy"""
        live = self._live
        if (live is None or live['rec'] is not rec or (RD_RELEASE and self.live_redist) or self.redistance
                or TOPO_CHECK):
            # (the collider's re-application in _finish_core runs only under self.redistance, which this path excludes)
            self.finish(scene, rec)
            return
        t0 = time.perf_counter()
        if not hasattr(self, '_result'):
            self._result = {}
        mirror = self.view.mirror
        kn_ = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        # a stroke that ran over an earlier one's write: its scene key was taken mid-write; the earlier one's light sync
        # leaves the view synced up to now when nothing else changed the CPU (any such job lands the queue first)
        light = (LIGHT_SYNC and getattr(self.view, '_sync_key', None) == kn_
                 and (live.get('key0') == kn_ or bool(e and e.get('chained'))))
        wrote = False
        for oi, rows, ks, Fb, vals in (snap if snap is not None else self._pw_snapshot(scene, rec)):
            i = 0
            n = len(rows)
            oid_ = scene.objects[oi].id
            parts = []                             # each chunk's result (disjoint keys), merged once at the end
            while i < n:
                m = int(e['wb_rows']) if e else PW_WB_ROWS
                r_, k_, fb_ = rows[i:i + m], ks[i:i + m], Fb[i:i + m]
                if isinstance(vals, np.ndarray):
                    fa_ = vals[i:i + m]
                else:
                    fa_ = vals[i * 729:(i + len(r_)) * 729].get().astype(np.float32).reshape(len(r_), 729)
                res_all = self._result
                self._result = {}
                try:
                    wrote = (self._write_back(scene, oi, r_, k_, fb_, fa_, rec) is not None) or wrote
                    if oid_ in self._result:
                        parts.append(self._result[oid_])
                finally:
                    self._result = res_all
                i += len(r_)
                if e is not None:
                    e['last_rows'] = len(r_)
                yield 'wb'
            if parts:
                nk = np.concatenate([p_[0] for p_ in parts])
                na = np.concatenate([p_[1] for p_ in parts])
                prev = self._result.get(oid_)
                if prev is not None and len(prev[0]):
                    def _code(k):
                        k = np.asarray(k, np.int64) + (1 << 20)
                        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
                    keep = ~np.isin(_code(prev[0]), _code(nk))
                    nk, na = np.concatenate([prev[0][keep], nk]), np.concatenate([prev[1][keep], na])
                self._result[oid_] = (nk, na)
            mirror.coarse_pending.append((mirror.objs[oid_].oid, ks.copy()))
        if self._live is live:
            self._live = None
        if light and wrote:
            self.view._sync(scene, known_current=True)
        self.stats['light_sync'] = bool(light and wrote)
        self.stats['finish_ms'] = (time.perf_counter() - t0) * 1000
        self.history.append(dict(self.stats, rows=sum(len(r) for r in live['rows'].values()),
                                 rebuilds=getattr(self, 'rebuilds', 0)))
        self.stats.pop('row_alias', None)
        self._trim_scratch()

    def _restore_rows(self, scene, live):
        """fs-brush 11-12: every row this stroke has touched gets its first-touch value back (the stroke-start field) and
        its aprons refreshed: the card's values from FLOW_DEV's blocks (one launch), else from the host. -> world boxes"""
        cp = self.cp
        boxes = []
        th = 256
        for oi, rows in live.get('rows', {}).items():
            if not rows:
                continue
            db = live.get('dbefore', {}).get(oi) if FLOW_DEV else None
            if db:
                if len(db) > 1:
                    db[:] = [(cp.concatenate([b[0] for b in db]), cp.concatenate([b[1] for b in db]),
                              np.concatenate([b[2] for b in db]))]
                d_rw, d_F, ks = db[0]
                self.k_restore(((len(ks) * 729 + th - 1) // th,), (th,), (self.view.gpu.sdf, d_rw, np.int32(len(ks)), d_F))
                ks = ks.reshape(-1, 3)
            else:
                rl = list(rows.keys())
                F0 = self._ft_rows(live, oi, rl).astype(np.float16).reshape(-1)   # fs-brush-3: dict or blocks
                self.k_restore(((len(rl) * 729 + th - 1) // th,), (th,),
                               (self.view.gpu.sdf, cp.asarray(np.array(rl, np.int32)), np.int32(len(rl)), cp.asarray(F0)))
                ks = np.array([rows[r_] for r_ in rl], np.int64).reshape(-1, 3)
            obj = scene.objects[oi]
            ov = self.view.mirror.objs[obj.id]
            self._aprons_card(ov, ks)
            boxes.append(self._world_box(obj, ov, ks, float(obj.sculpt.h)))
        return boxes

    def _apply_snakepath(self, scene, oid, settings, samples, rec):
        """fs-brush 11-12 (tk3 / s5 at 1 mm: a path tool re-reading its own field each step diffused its form): Snake
        Hook's whole path composed on the card: the touched rows back to the stroke-start field, then per mirror copy ONE
        dab (sculpt.cu typ 17) walking every node back through all the path's steps (brush.snake_path_back). -> boxes"""
        live = self._live
        same = live is not None and live.get('rec') is rec
        allS = (list(live.get('snake_all', [])) if same else []) + list(samples)
        boxes = self._restore_rows(scene, live) if same else []
        oi = scene.index_of(oid)
        obj = scene.objects[oi]
        from ..tools.protected_merge import _to_world
        s_ = max(float(obj.transform['scale']), 1e-6)
        for steps in BR.snake_path_steps(obj, allS, settings):
            if not len(steps):
                continue
            c_l, r_l = BR.snake_path_ball(steps)
            p_w = _to_world(obj, np.asarray(c_l, float)[None, :])[0]
            st = dict(settings, tool='snakepath', radius=r_l * s_, snake_steps_local=steps, autosmooth=0.0,
                      symmetry=[False, False, False], snake=False, sym_world=False)
            boxes += list(self._apply_batch(scene, oid, st, [{'p': np.asarray(p_w).tolist(), 'n': [0.0, 0.0, 1.0],
                                                              'v': [0.0, 0.0, 0.0]}], rec) or [])
        if self._live is not None:
            self._live['snake_all'] = allS
        if len(boxes) > 1:
            boxes = [(np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0))]
        return boxes

    def _rows_per_sample(self, scene, oid, settings):
        """fs-brush 11-12: a rough count of the rows one sample's dabs track: the surface patch under the brush (pi r^2
        in bricks) times the band's thickness in bricks, times the mirror copies"""
        o = scene.get(oid)
        h = float(o.sculpt.h)
        bm = 8.0 * h
        r = float(settings.get('radius', 0.01)) / max(float(o.transform['scale']), 1e-9)
        band = 1.6 * 0.05 * r + BR.SKIP_BAND_H * h
        nsym = 2 ** sum(1 for v in (settings.get('symmetry') or [False] * 3) if v)
        return math.pi * (r / bm + 1.0) ** 2 * (2.0 * band / bm + 2.0) * nsym

    def _apply_batch(self, scene, oid, settings, samples, rec):
        """Apply world samples like brush.apply_stroke, on the GPU. r16: the CPU sculpt layer is written once, at the
        stroke's end (finish): during the drag the card holds the current field (the viewport and picks read it).
        Order per batch: every dab's cells and admissions first (a new brick's own nodes were never changed on the
        card, so the CPU field holds their current values), the before-images of bricks new to this stroke, the dabs
        in order, then the aprons refreshed from their owners. -> world dirty boxes."""
        cp = self.cp
        t0 = time.perf_counter()
        if self.view.changed:
            # the atlas must hold the current field before dabs read it. fs-brush 11-11: only when the CPU scene changed
            # (the key GpuView.frame checks): during a card stroke nothing changes there until the release, and the full
            # mirror sync (a checksum of every sculpt brick + a dict over every slot) ran every batch, growing with the
            # session's brick count (M: the head scene's 5-10 fps whatever the brush size)
            key = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
            if key != getattr(self.view, '_sync_key', None) and not getattr(self.view, 'hold_sync', False):
                self.view._sync(scene)
        live = self._live
        if live is None or live['rec'] is not rec:
            live = self._live = {'rec': rec, 'rows': {}, 'before': {}, 'tool': settings.get('tool')}
            # fs-brush 11-12: the scene key this stroke's card state starts from (the light sync at its release needs
            # nothing else on the CPU to have changed since)
            kn_ = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
            live['key0'] = kn_ if kn_ == getattr(self.view, '_sync_key', None) else None
            self._mask_reset()
            if getattr(self, '_result_rec', None) is not rec:
                self._result = {}                  # r42: a new stroke's result (kept across a mid-stroke write-back)
                self._result_rec = rec
        self._scene_objs = [o.id for o in scene.objects]
        lut = BR.falloff_lut(settings.get('falloff', 'smooth'), settings.get('curve')).astype(np.float32)
        d_lut = cp.asarray(lut)
        plans = [self._plan(scene, oi, st, c, n, v) for oi, st, c, n, v in self._expand(scene, oid, settings, samples)]
        plans = [p for p in plans if p is not None]
        t_pl = time.perf_counter()
        for p in plans:
            live.setdefault('dabs', {}).setdefault(p['oi'], []).append(np.r_[p['bp'][1:4], p['bp'][7]])
        for oi in sorted({p['oi'] for p in plans}):
            adm = [p['admit'] for p in plans if p['oi'] == oi and len(p['admit'])]
            if adm:
                ov = self.view.mirror.objs[scene.objects[oi].id]
                cells = _ukeys(np.concatenate(adm))
                if len(cells):
                    cells = cells[ov.rowmap[cells[:, 0], cells[:, 1], cells[:, 2]] < 0]
                pl_ = [p for p in plans if p['oi'] == oi]
                # (not Clay / Smooth: their area plane and ring samples read the field up to the form's own departure
                # from a plane under the brush, beyond the band)
                band_tools = (BR.T['draw'], BR.T['carve'], BR.T['inflate'], BR.T['swell'])
                if getattr(ov, 'is_store', False) and STORE_BAND_ADMIT:
                    # fs-brush 11-12 (probe5b / p71: a Smooth r 60 and an Inflate r 45 under Mirror X admitted +378 k rows
                    # of her into the atlas and ran the card out): on her store object Smooth and Clay change only band
                    # nodes too (their skip band), and what their rings / plane read beyond it comes from her store
                    # volume where no row is held, which is her current field there (a cell holding a change always
                    # has its row): band cells only for them as well
                    band_tools = band_tools + (BR.T['smooth'], BR.T['clay'])
                # fs-brush 11-12 (F5): her store object too (it admitted EVERY cell of each dab ball: the card grew with each
                # torso stroke). A cell left out holds no change (its nodes lie beyond every dab's reach), so her store volume
                # draws it exactly as the CPU field has it
                if (len(cells) and BAND_ADMIT and (STORE_BAND_ADMIT or not getattr(ov, 'is_store', False))
                        and all(p['bp'][BR.NBP_SKIP] > 0 and int(p['bp'][0]) in band_tools for p in pl_)):
                    # fs-brush 11-11 (band-only allocation, step 2; fs-trust's memory finding): a
                    # brush whose dabs skip nodes beyond their reach (bp[28]) never changes a cell lying wholly beyond it,
                    # so such a cell is not brought onto the card at all (the CPU field there is current: the card never
                    # held it). The test: |F| at the cell's centre beyond the batch's largest reach + its half-diagonal
                    # + the batch's summed move (the surface may come that much nearer within the batch).
                    from ..core.program import eval_points_local
                    pk_ = scene.pack()
                    bm_ = pl_[0]['bm']
                    ctr = np.ascontiguousarray(ov.vol.origin + (cells + 0.5) * bm_, np.float64)
                    fc = np.empty(len(ctr))
                    eval_points_local(pk_.code, pk_.prm, pk_.oi_tab, pk_.of_tab, pk_.index_flat, pk_.delta, oi, ctr, fc)
                    lim = max(float(p['bp'][BR.NBP_SKIP]) for p in pl_) + 0.8661 * bm_ + sum(abs(float(p['bp'][13])) for p in pl_)
                    if SMOOTH_RING_BAND and any(int(p['bp'][0]) == BR.T['smooth'] for p in pl_):
                        # fs-brush 11-12: Smooth's ring samples lie on the tangent plane, up to half the ring radius off the
                        # surface (its height clamp): the band they read is that wide
                        lim = max(lim, 0.5 * BR.QSM_R2 * max(float(p['bp'][7]) for p in pl_) + 0.8661 * bm_
                                  + sum(abs(float(p['bp'][13])) for p in pl_))
                    if ADMIT_GROW:
                        # sculpt-contact 11-12 (fold15 / fold16: flat brick-aligned faces where Swell to contact's growth
                        # met cells never brought onto the card). The test reads the field from BEFORE the stroke and
                        # one batch's nominal move; a growing surface may already stand nearer by the true move of this
                        # stroke's earlier dabs over the cell (Swell to contact: up to gain x the amount). Each cell's
                        # limit gains the summed true move of the earlier dabs whose ball reaches it.
                        lim = lim - sum(abs(float(p['bp'][13])) for p in pl_) + sum(_true_move(p) for p in pl_)
                        keep_ = np.abs(fc) <= lim + _grown(live.get('sc_grow', {}).get(oi), ctr, bm_)
                    else:
                        keep_ = np.abs(fc) <= lim
                    self.stats_band_skip = int((~keep_).sum())
                    cells = cells[keep_]
                if len(cells):
                    self._admit(scene, oi, ov, cells)
        if ADMIT_GROW:
            for p in plans:
                if int(p['bp'][0]) in (BR.T['draw'], BR.T['carve'], BR.T['inflate'], BR.T['swell']):
                    g_ = live.setdefault('sc_grow', {})
                    row = np.r_[np.asarray(p['bp'][1:4], np.float64), float(p['bp'][7]), _true_move(p)][None]
                    g_[p['oi']] = row if g_.get(p['oi']) is None else np.concatenate([g_[p['oi']], row])
        t_adm = time.perf_counter()
        self.view.gpu._tables()                  # r25: once per batch (tables change only at admission)
        t_tab = time.perf_counter()
        # fs-brush 11-11 (M: 6-30 ms a batch and 20 ms at the release in per-row Python loops at r 80): the rows this
        # stroke touches for the first time found with a per-object 'seen' mask over the pool, their before-images
        # gathered as one block; live['rows'] / live['before'] (row -> key / values, first-touch order) are filled by
        # dict.update, and the blocks ('blk') give finish its arrays in that same order without a per-row loop.
        for p in plans:
            ov = self.view.mirror.objs[scene.objects[p['oi']].id]
            cl = p['cells']
            rw = ov.rowmap[cl[:, 0], cl[:, 1], cl[:, 2]] if len(cl) else np.zeros(0, np.int64)
            p['keys'] = cl[rw >= 0]
            p['rows'] = rw[rw >= 0]
        tsdf_on = TSDF and self.live_redist and settings.get('tool') != 'mask'
        live['tsdf'] = tsdf_on
        if ROW_FILT and not tsdf_on:
            # fs-core 11-14: not with the truncated band (its rows' stored values are what it keeps true; dropping a row by
            # its values left the stale rows the next stroke stopped at)
            # fs-brush 11-11 (M, head test stage 5: a mirrored r 30 Smooth ran 3 000-3 500 rows a batch, 6-8 dabs, 8-13
            # fps; a dab's sphere holds ~4x the bricks its band reaches): when every dab of the object skips nodes beyond
            # its reach (bp[28]), a brick whose nearest node lies beyond the re-distance band (_footprint_redist's jband,
            # >= every reach) is changed by no dab and no re-distance in this batch: a node changes only while its own
            # value is within its dab's reach. Such bricks are dropped from the plans before tracking, so they are not
            # gathered, computed, refreshed or written back. Exact: the same nodes change, to the same values.
            for oi in sorted({p['oi'] for p in plans}):
                pl = [p for p in plans if p['oi'] == oi and len(p['rows'])]
                if not pl or not all(p['bp'][BR.NBP_SKIP] > 0 for p in pl):
                    continue
                h_ = float(scene.objects[oi].sculpt.h)
                jb = max(1.7 * float(max(p['bp'][BR.NBP_SKIP] for p in pl)) + 2 * h_, 8 * h_)
                ru = np.unique(np.concatenate([p['rows'] for p in pl]).astype(np.int64))
                mn = cp.full(len(ru), 1e30, cp.float32)
                self.k_minabs((len(ru),), (256,), (self.view.gpu.sdf, cp.asarray(ru.astype(np.int32)), np.int32(len(ru)), mn))
                near = np.zeros(int(ru.max()) + 1, bool)
                near[ru[mn.get() <= jb]] = True
                for p in pl:
                    k_ = near[p['rows']]
                    p['rows'], p['keys'] = p['rows'][k_], p['keys'][k_]
                self.stats_row_filt = (len(ru), int(near.sum()))
        for oi in sorted({p['oi'] for p in plans}):
            pl = [p for p in plans if p['oi'] == oi and len(p['rows'])]
            if not pl:
                continue
            rr = np.concatenate([p['rows'] for p in pl]).astype(np.int64)
            kk = np.concatenate([p['keys'] for p in pl]).astype(np.int64).reshape(-1, 3)
            rr, ix = np.unique(rr, return_index=True)
            kk = kk[ix]
            seen = live.setdefault('seen', {}).get(oi)
            need = int(rr.max()) + 1
            if seen is None or len(seen) < need:
                s2 = np.zeros(max(need, 2 * (len(seen) if seen is not None else 0), 1024), bool)
                if seen is not None:
                    s2[:len(seen)] = seen
                seen = live['seen'][oi] = s2
            nw = ~seen[rr]
            if not nw.any():
                continue
            rws, kws = rr[nw], kk[nw]
            seen[rws] = True
            if live.setdefault('mask', {}).setdefault(oi, self._mask_in_use(scene, oi, settings)):
                self._mask_fill(scene, oi, self.view.mirror.objs[scene.objects[oi].id], rws, kws, settings)
            F = self._gather(rws)
            if live.get('tool') in BR.FT_ALL:               # fs-tools patch 4: the Pencil's tip reads the press surface too
                self._ft_start_add(live, oi, rws, F)     # fs-tools patch 3: the press-time field for Follow's anchors
            if FLOW_DEV and live.get('tool') in ('grabflow', 'stretchflow', 'snakepath'):
                # fs-brush 11-12 speed (c): the anchored tools restore every touched row each batch; their first-touch
                # values go to the card once here (fp16, exactly the gathered values) instead of a per-row stack + upload
                live.setdefault('dbefore', {}).setdefault(oi, []).append(
                    (cp.asarray(rws.astype(np.int32)), cp.asarray(F.astype(np.float16).reshape(-1)), np.asarray(kws, np.int64)))
            live['rows'].setdefault(oi, {}).update(zip(rws.tolist(), kws.tolist()))
            live['before'].setdefault(oi, {}).update(zip(rws.tolist(), F))
            live.setdefault('blk', {}).setdefault(oi, []).append((rws, kws, F))
        t1 = time.perf_counter()
        nb = 0
        if PROF:
            cp.cuda.Stream.null.synchronize()
            self.prof = {'_t': time.perf_counter(), 'plan_ms': (time.perf_counter() - t0) * 1000,
                         'plans_ms': (t_pl - t0) * 1000, 'admit_ms': (t_adm - t_pl) * 1000, 'tables_ms': (t_tab - t_adm) * 1000,
                         'track_ms': (t1 - t_tab) * 1000}
        # fs-brush 11-11: the keep-apart guard once per batch on its net change (BATCH_GUARD; see sculpt.cu rows_restore)
        bguard = {}
        if BATCH_GUARD:
            for oi in sorted({p['oi'] for p in plans if p['bp'][BR.NBP_TOPO] > 0.5}):
                pl = [p for p in plans if p['oi'] == oi and len(p['rows'])]
                if not pl:
                    continue
                ru = np.concatenate([p['rows'] for p in pl]).astype(np.int64)
                ku = np.concatenate([p['keys'] for p in pl]).astype(np.int64).reshape(-1, 3)
                ru, ix = np.unique(ru, return_index=True)
                ku = ku[ix]
                d_ru = cp.asarray(ru.astype(np.int32))
                S16 = self._scr('g_s16_%d' % oi, len(ru) * 729, np.float16)   # kept until this object's guard
                th_ = 256
                self.k_gather(((len(ru) * 729 + th_ - 1) // th_,), (th_,), (self.view.gpu.sdf, d_ru, np.int32(len(ru)), S16))
                mv = 0.0
                for p in pl:
                    mv += float(np.linalg.norm(p['bp'][9:12])) if int(p['bp'][0]) in (BR.T['grab'], BR.T['stretch'], BR.T['grabflow'], BR.T['stretchflow'], BR.T['snakeflow']) else abs(float(p['bp'][13]))
                    p['bp'] = p['bp'].copy()
                    p['bp'][BR.NBP_TOPO] = 0.0               # the dabs commit directly; the guard runs below
                bguard[oi] = (ru, ku, d_ru, S16, mv, pl[0]['vol'], pl[0]['bm'])
        for p in plans:
            p['mask'] = bool(live.get('mask', {}).get(p['oi'], False))
        t_sh = time.perf_counter()
        if BR.sheet_on(settings):
            for oi_ in sorted({p['oi'] for p in plans}):
                if SHEET_HOST:
                    self._sheet_masks(scene, [p for p in plans if p['oi'] == oi_], live, settings)
                else:
                    self._sheet_masks_card(scene, [p for p in plans if p['oi'] == oi_], live, settings)
        self.stats_sheet_ms = (time.perf_counter() - t_sh) * 1000
        for p in plans:
            if p['bp'][BR.NBP_SWC] > 0.5:             # sculpt-contact: the contacts' distance at the plan's rows
                from . import collide_card as _CC
                p['T'] = _CC.contact_rows(self, scene, settings, p, live)
        self._rd_pre = None
        if RD_CHG and self.live_redist and RD_MODE == 'eik' and settings.get('tool') != 'mask':
            # the batch's rows as they are before its dabs (on the card): the re-distance's measure of what changed
            rr_ = [p['rows'] for p in plans if len(p['rows'])]
            if rr_:
                ru_ = np.unique(np.concatenate(rr_).astype(np.int64))
                G_ = self.view.gpu
                pre_ = self._scr('rd_pre', len(ru_) * 729, np.float16)
                th_ = 256
                self.k_gather(((len(ru_) * 729 + th_ - 1) // th_,), (th_,),
                              (G_.sdf, cp.asarray(ru_.astype(np.int32)), np.int32(len(ru_)), pre_))
                pidx_ = cp.full(G_.used + 1, -1, cp.int32)
                pidx_[cp.asarray(ru_)] = cp.arange(len(ru_), dtype=cp.int32)
                self._rd_pre = (pre_, pidx_)
        for p in plans:
            nb += self._run(p, d_lut, len(lut))
        for oi, (ru, ku, d_ru, S16, mv, vol_, bm_) in bguard.items():
            self._guard_batch(ru, ku, d_ru, S16, mv, vol_, bm_)
        if (settings.get('collide', False) or settings.get('tool') == 'swell') and settings.get('tool') != 'mask':
            live['collide'] = dict(settings, collide=True)
            self.stats_collide = self._collide(scene, settings, plans, live)
        t_rd = time.perf_counter()
        changed = {}
        if self.live_redist and settings.get('tool') != 'mask':
            changed = self._footprint_redist(scene, plans) or {}
        if BR.sheet_on(settings):
            if SHEET_HOST:
                self._sheet_restore(plans, live)
            else:
                self._sheet_restore_card(live)
        if PROF:
            cp.cuda.Stream.null.synchronize()
        self.stats_rd_ms = (time.perf_counter() - t_rd) * 1000
        boxes = []
        for oi in sorted({p['oi'] for p in plans}):
            ov = self.view.mirror.objs[scene.objects[oi].id]
            ks = np.concatenate([p['keys'] for p in plans if p['oi'] == oi]) if plans else np.zeros((0, 3), np.int64)
            if len(ks) == 0:
                continue
            ks = _ukeys(ks)
            ch_ = changed.get(oi)
            if ROW_FILT:
                # every brick a node of which changed is in the (filtered) plans; the aprons of all of them are refreshed
                # (a far brick's apron nodes were computed by its own dab before; now they come only from their owner)
                ch_ = None
            self._aprons_card(ov, ks if ch_ is None else ch_)
            bm = 8 * float(scene.objects[oi].sculpt.h)
            # fs-brush 11-11: one box per cluster of overlapping dabs, not the union of the batch (under Mirror X the
            # union spanned both sides of the head and the frame restarted nearly the whole view: M, 7-14 ms a frame)
            cl = []
            for p in plans:
                if p['oi'] != oi or not len(p['keys']):
                    continue
                kb_ = p['dcells'] if p.get('dcells') is not None and len(p['dcells']) else p['keys']
                a, b = kb_.min(0), kb_.max(0) + 1
                for q in cl:
                    if np.all(a <= q[1]) and np.all(b >= q[0]):
                        q[0], q[1] = np.minimum(q[0], a), np.maximum(q[1], b)
                        break
                else:
                    cl.append([a, b])
            of = scene.pack().of_tab[oi]
            for a, b in cl:
                boxes.append(self.view.mirror._world_box(of, (ov.vol.origin + a * bm, ov.vol.origin + b * bm)))
        cp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()
        self.stats = {'dabs': len(plans), 'bricks': nb, 'plan_ms': (t1 - t0) * 1000, 'gpu_ms': (t2 - t1) * 1000,
                      'rd_ms': getattr(self, 'stats_rd_ms', 0.0)}
        if PROF:
            self.stats['aprons_ms'] = (t2 - t_rd) * 1000 - self.stats['rd_ms']
            self.stats.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.prof.items() if k != '_t'})
        return boxes

    # fs-brush 11-11: the live re-distance of the batch's dab footprint (sculpt.cu rd_seed / jfa_step / rd_apply; the
    # measured cause and the scheme are written there). On by default; FS_LIVE_REDIST=0 turns it off (then the release
    # re-distance runs as before).
    live_redist = os.environ.get('FS_LIVE_REDIST', '1') != '0'

    def _footprint_redist(self, scene, plans, dabs_over=None, iface_rw=False):
        """-> {oi: keys of the rows whose values may have changed in this batch} (the apron refresh's set), or None for
        an object whose every row may have changed. dabs_over {oi: (n, 4)}: the footprint's spheres instead of the
        plans' dabs; iface_rw: the surface layer re-distanced too (_release_norm only)"""
        cp = self.cp
        G = self.view.gpu
        changed = {}
        for oi in sorted({p['oi'] for p in plans}):
            pl = [p for p in plans if p['oi'] == oi and len(p['rows'])]
            if not pl:
                continue
            o = scene.objects[oi]
            ov = self.view.mirror.objs[o.id]
            rows = np.concatenate([p['rows'] for p in pl]).astype(np.int64)
            keys = np.concatenate([p['keys'] for p in pl]).astype(np.int64).reshape(-1, 3)
            rows, ix = np.unique(rows, return_index=True)
            keys = keys[ix]
            dabs = np.array([np.r_[p['bp'][1:4], p['bp'][7]] for p in pl], np.float32).reshape(-1, 4)
            if dabs_over is not None and oi in dabs_over:
                dabs = np.asarray(dabs_over[oi], np.float32).reshape(-1, 4)
            h = float(o.sculpt.h)
            if TSDF and not iface_rw and RD_MODE == 'eik':
                # fs-core 11-14: THE TRUNCATED BAND (see TSDF above). Every held row within 2 W of the batch's balls
                # (_plan); the eikonal from the surface's own nodes (kept values) to convergence within W; then
                # rd_tsdf_apply: a node nearer a ball than min(|F|, W) takes min(U, W), every node |F| <= W
                Wt = TSDF_W_H * h
                nb = len(rows)
                d_rowmap, shp, kO = self._rowmap_slice(ov, keys)
                regmap = cp.full(G.used + 1, -1, cp.int32)
                regmap[cp.asarray(rows)] = cp.arange(nb, dtype=cp.int32)
                d_rows = cp.asarray(rows.astype(np.int32))
                d_keys = cp.asarray((keys - kO).astype(np.int32).reshape(-1))
                th = 256
                nt = (nb * 512 + th - 1) // th
                dims = (np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2]))
                org0 = cp.asarray(np.asarray(np.asarray(ov.vol.origin, np.float64) + kO * 8 * h, np.float32))
                U0 = self._scr('rd_u0', nb * 512, np.float32)
                U1 = self._scr('rd_u1', nb * 512, np.float32)
                iface = self._scr('rd_iface', nb * 512, np.uint8)
                GLb = self._scr('rd_gl', nb * 512, np.float32)
                self.k_rd_eik_init((nt,), (th,), (G.sdf, d_rowmap) + dims + (d_rows, d_keys, np.int32(nb), np.float32(h), U0, iface, GLb, np.int32(1), np.float32(TSDF_GK)))
                it_max = min(int(math.ceil(1.7321 * Wt / h)) + 3, 64)
                chg = cp.zeros(1, cp.int32)
                for it in range(it_max):
                    if it and it % 8 == 0:
                        if int(chg.get()[0]) == 0:
                            break
                        chg[...] = 0
                    self.k_rd_eik_step((nt,), (th,), (d_rowmap,) + dims + (regmap, d_keys, np.int32(nb), np.float32(h), U0, U1, iface, chg))
                    U0, U1 = U1, U0
                stt = cp.zeros(2, cp.int32)
                self.k_rd_tsdf_apply((nt,), (th,), (G.sdf, d_rows, d_keys, np.int32(nb), np.float32(h), org0,
                                                    cp.asarray(dabs.reshape(-1)), np.int32(len(dabs)), U0, iface,
                                                    np.float32(Wt), stt,
                                                    np.float32(TSDF_NB_TOL * h if TSDF_GK > 0 and TSDF_NB_TOL >= 0 else -1.0)))
                self.stats_rd = {'bricks': nb, 'iters': it + 1, 'tsdf': True}
                changed[oi] = None
                continue
            # fs-brush 11-11: only the bricks near the surface. Every dab of the batch skipped nodes beyond its reach
            # (bp[28]); a brick whose nearest node lies beyond the largest reach + 2 h was not changed by any dab and
            # is too far for the re-distance to matter to the surface, so it keeps its values (as before the batch)
            if all(p['bp'][BR.NBP_SKIP] > 0 for p in pl):
                jband = max(1.7 * float(max(p['bp'][BR.NBP_SKIP] for p in pl)) + 2 * h, 8 * h)
                d_r = cp.asarray(rows.astype(np.int32))
                mn = cp.full(len(rows), 1e30, cp.float32)
                th_ = 256
                self.k_minabs((len(rows),), (th_,), (G.sdf, d_r, np.int32(len(rows)), mn))
                near = mn.get() <= jband
                rows, keys = rows[near], keys[near]
                changed[oi] = keys
                if not len(rows):
                    continue
            else:
                changed[oi] = None
                if RD_REACH and RD_MODE == 'eik' and len(rows) > 64:
                    # fs-brush 11-11 (M, p43: a Grab batch at r 80 re-distanced all 5 673 rows of its ball, 18 Jacobi steps
                    # each, 15 ms): a step moves a distance one node, so after `it` steps only nodes within `it` nodes of
                    # a surface node can hold one. Every surface node lies in the 9^3 of a brick holding both signs, and a
                    # brick d bricks (along some axis) from that one starts 8 d - 8 nodes beyond its apron, so bricks more
                    # than it // 8 + 1 bricks from every such brick are out of reach and keep their values either way.
                    # Exact: the same nodes, the same values, the same steps.
                    it_ = min(int(math.ceil(16 * h / h)) + 2, 40)
                    D = it_ // 8 + 1
                    d_r = cp.asarray(rows.astype(np.int32))
                    lo_ = cp.empty(len(rows), cp.float32)
                    hi_ = cp.empty(len(rows), cp.float32)
                    self.k_minmax((len(rows),), (256,), (G.sdf, d_r, np.int32(len(rows)), lo_, hi_))
                    mixed = (lo_.get() <= 0.0) & (hi_.get() >= 0.0)   # a 0 node is a surface node too (rd_eik_init)
                    k0 = keys.min(0)
                    kk = keys - k0
                    g_ = np.zeros(tuple(kk.max(0) + 1), bool)
                    g_[kk[mixed, 0], kk[mixed, 1], kk[mixed, 2]] = True
                    for ax in range(3):                  # a box dilation by D bricks, axis by axis
                        g0 = g_.copy()
                        for s_ in range(1, D + 1):
                            if s_ >= g_.shape[ax]:
                                break
                            sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
                            sl_a[ax] = slice(s_, None); sl_b[ax] = slice(None, -s_)
                            g_[tuple(sl_a)] |= g0[tuple(sl_b)]
                            g_[tuple(sl_b)] |= g0[tuple(sl_a)]
                    keep = g_[kk[:, 0], kk[:, 1], kk[:, 2]]
                    self.stats_rd_reach = (len(rows), int(keep.sum()))
                    rows, keys = rows[keep], keys[keep]
                    if not len(rows):
                        continue
            nb = len(rows)
            d_rowmap, shp, kO = self._rowmap_slice(ov, keys)   # fs-brush 11-12 (F5): the batch's slice of the map (keys +-1 cell)
            regmap = cp.full(G.used + 1, -1, cp.int32)
            regmap[cp.asarray(rows)] = cp.arange(nb, dtype=cp.int32)
            d_rows = cp.asarray(rows.astype(np.int32))
            d_keys = cp.asarray((keys - kO).astype(np.int32).reshape(-1))   # relative to the slice
            th = 256
            nt = (nb * 512 + th - 1) // th
            dims = (np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2]))
            org0 = cp.asarray(np.asarray(np.asarray(ov.vol.origin, np.float64) + kO * 8 * h, np.float32))   # the slice's
            if RD_MODE == 'eik':
                # fs-brush 11-11: the CPU scheme (seeds + Godunov sweeps), interface nodes untouched (sculpt.cu rd_eik_*)
                U0 = self._scr('rd_u0', nb * 512, np.float32)
                U1 = self._scr('rd_u1', nb * 512, np.float32)
                iface = self._scr('rd_iface', nb * 512, np.uint8)
                GLb = self._scr('rd_gl', nb * 512, np.float32)
                self.k_rd_eik_init((nt,), (th,), (G.sdf, d_rowmap) + dims + (d_rows, d_keys, np.int32(nb), np.float32(h), U0, iface, GLb, np.int32(1 if RD_SEED_KEPT else 0), np.float32(0.0)))
                band_n = int(math.ceil((jband if changed.get(oi) is not None else 16 * h) / h)) + 2
                chg = cp.zeros(1, cp.int32)
                for it in range(min(band_n, 40)):
                    if it and it % 8 == 0:
                        if int(chg.get()[0]) == 0:
                            break
                        chg[...] = 0
                    self.k_rd_eik_step((nt,), (th,), (d_rowmap,) + dims + (regmap, d_keys, np.int32(nb), np.float32(h), U0, U1, iface, chg))
                    U0, U1 = U1, U0
                stt = cp.zeros(2, cp.int32)
                pre_ = getattr(self, '_rd_pre', None) if not iface_rw else None
                if pre_ is None:
                    if getattr(self, '_rd_pre_dummy', None) is None:
                        self._rd_pre_dummy = (cp.zeros(729, cp.float16), cp.full(1, -1, cp.int32))
                    pa_ = self._rd_pre_dummy + (np.int32(0),)
                else:
                    pa_ = pre_ + (np.int32(1),)
                self.k_rd_eik_apply((nt,), (th,), (G.sdf, d_rows, d_keys, np.int32(nb), np.float32(h), org0,
                                                   cp.asarray(dabs.reshape(-1)), np.int32(len(dabs)), U0, iface, stt)
                                    + self._margs(bool(pl[0].get('mask'))) + (GLb, np.int32(1 if iface_rw else 0))
                                    + pa_ + (np.float32(RD_CHG_EPS_H * h),))
                self.stats_rd = {'bricks': nb, 'iters': it + 1}
                continue
            sa = cp.empty((nb * 512, 4), cp.float32)
            sb = cp.empty((nb * 512, 4), cp.float32)
            self.k_rd_seed((nt,), (th,), (G.sdf, d_rowmap) + dims + (d_rows, d_keys, np.int32(nb), sa))
            ext = int((keys.max(0) - keys.min(0)).max() + 1) * 8
            if changed.get(oi) is not None:      # band rows: every node lies within jband + a brick of a seed
                ext = min(ext, int(math.ceil(jband / h)) + 16)
            step = 1
            while step * 2 < min(ext, 128):
                step *= 2
            steps = []
            while step >= 1:
                steps.append(step)
                step //= 2
            for st_ in steps + [1]:
                self.k_jfa_step((nt,), (th,), (d_rowmap,) + dims + (regmap, d_keys, np.int32(nb), np.int32(st_), sa, sb))
                sa, sb = sb, sa
            stt = cp.zeros(2, cp.int32)
            org0 = cp.asarray(np.asarray(np.asarray(ov.vol.origin, np.float64) + kO * 8 * h, np.float32))   # the slice's
            self.k_rd_apply((nt,), (th,), (G.sdf, d_rows, d_keys, np.int32(nb), np.float32(h), org0,
                                           cp.asarray(dabs.reshape(-1)), np.int32(len(dabs)), sa, stt)
                            + self._margs(bool(pl[0].get('mask'))))
            self.stats_rd = {'bricks': nb, 'steps': len(steps) + 1}
        return changed

    _live = None

    def _collide(self, scene, settings, plans, live):
        """the collider step after a batch's dabs: gpu/collide_card.py (sculpt-contact's module since 11-12)"""
        from . import collide_card as CC
        return CC.collide(self, scene, settings, plans, live)

    def _rows_far(self, obj, vol, keys, parts, live, oi, gap):
        from . import collide_card as CC
        return CC.rows_far(self, obj, vol, keys, parts, live, oi, gap)

    def _collide_rows(self, scene, obj, oi, parts, rk, vol, live, gap, pk):
        """the collider rule for given rows (the release): gpu/collide_card.py"""
        from . import collide_card as CC
        return CC.collide_rows(self, scene, obj, oi, parts, rk, vol, live, gap, pk)

    def _live_cells(self, oid):
        live = self._live
        if live is None:
            return None
        out = set()
        for oi, rk in live['rows'].items():
            if oi < len(self._scene_objs) and self._scene_objs[oi] == oid:
                out.update(tuple(k) for k in rk.values())
        return out

    _scene_objs = []

    def _rowmap_slice(self, ov, keys):
        """fs-brush 11-12 (F5: does the map kept on the card add to the resident set? yes: 1 063 MB on her (p67), and a body
        Smooth then ran the card out of memory): -> (map, shape, lo): the object's cell -> row map over the box of keys
        (cells) +-1 cell, int32 on the card, a few MB per batch and nothing resident. The kernels read the map only at a
        key's own cell and its +-1 neighbours (node_row / apron_fill_nb: -1 outside, as outside the whole map; JFA's longer
        jumps reach only rows of this batch, through regmap) and place nodes at org0 + I h: keys relative to lo and org0
        shifted by lo give the same nodes. Before e42472b8 the WHOLE map was uploaded twice a batch (her 0.4-0.85 s each)."""
        cp = self.cp
        k = np.asarray(keys, np.int64).reshape(-1, 3)
        shp = np.array(ov.rowmap.shape, np.int64)
        lo = np.maximum(k.min(0) - 1, 0)
        hi = np.minimum(k.max(0) + 2, shp)
        sub = np.ascontiguousarray(ov.rowmap[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.int32)
        d = cp.empty(sub.shape, cp.int32)
        d.set(sub)                              # a plain copy (no pinned staging buffer)
        return d, sub.shape, lo

    def _aprons_card(self, ov, ks):
        """fs-brush 11-11: _aprons with the neighbour bricks found on the card (sculpt.cu apron_fill_nb)"""
        cp = self.cp
        if len(ks) == 0:
            return
        d_rm, shp, kO = self._rowmap_slice(ov, ks)   # fs-brush 11-12 (F5): the batch's slice (apron reach: +-1 cell)
        d_k = cp.asarray(np.ascontiguousarray(np.asarray(ks, np.int64) - kO, np.int32).reshape(-1))
        n = len(ks) * 8 * 217
        th = 256
        self.k_apron_nb(((n + th - 1) // th,), (th,), (self.view.gpu.sdf, d_rm, np.int32(shp[0]), np.int32(shp[1]),
                                                         np.int32(shp[2]), d_k, np.int32(len(ks))))

    def _aprons(self, ov, ks):
        """r16: refresh the aprons of the touched bricks and of their lower neighbours (whose aprons are copies of the
        touched bricks' own nodes)."""
        cp = self.cp
        G = self.view.gpu
        if len(ks) == 0:
            return
        allk = _ukeys((ks[:, None, :] + np.array([[a, b, c] for a in (0, -1) for b in (0, -1) for c in (0, -1)])[None]).reshape(-1, 3))
        n = np.array(ov.rowmap.shape)
        allk = allk[np.all((allk >= 0) & (allk < n), 1)]
        rw = ov.rowmap[allk[:, 0], allk[:, 1], allk[:, 2]]
        allk, rw = allk[rw >= 0], rw[rw >= 0]
        if len(rw) == 0:
            return
        vidx = G.vols.index(ov.vol)
        th = 256
        self.k_apron(((len(rw) * 729 + th - 1) // th,), (th,),
                     (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL),
                      cp.asarray(rw.astype(np.int32)), cp.asarray(allk.astype(np.int32).reshape(-1)), np.int32(len(rw))))

    def full_check(self, scene):
        """debug: every GPU brick of every object against the CPU field at its own nodes (|F| < band) -> per object
        (bricks over 50 um, max um, a few keys)."""
        out = {}
        mirror = self.view.mirror
        for oi, o in enumerate(scene.objects):
            ov = mirror.objs.get(o.id)
            if ov is None or ov.vol is None or not ov.key_to_row:
                continue
            keys = np.array(list(ov.key_to_row.keys()), np.int64)
            rows = np.array([ov.key_to_row[tuple(k)] for k in keys.tolist()], np.int64)
            Fa = self._gather(rows).reshape(-1, 9, 9, 9)[:, :8, :8, :8].reshape(len(rows), -1)
            mirror._field(scene, oi)
            pk = mirror._pk
            h = float(o.sculpt.h)
            cc = np.empty((len(keys), 729))
            _eval_nodes9(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi,
                         np.ascontiguousarray(ov.vol.origin + keys * 8 * h, np.float64), h, cc)
            cc = cc.reshape(-1, 9, 9, 9)[:, :8, :8, :8].reshape(len(keys), -1)
            band = ov.vol.levels[LEVEL]['band']
            d = np.abs(Fa - cc).max(1)                         # every node (r20b: the band-only check missed it)
            # fp16 storage: a stored value is within half an ulp of the CPU's (relative 2^-11)
            d = np.maximum(d - np.abs(cc).max(1) * 2.0 ** -10, 0.0)
            bad = np.nonzero(d > 50e-6)[0]
            if len(bad):
                out[o.name] = {'bad': int(len(bad)), 'of': int(len(keys)), 'max_um': round(float(d.max()) * 1e6, 1),
                               'keys': keys[bad[:4]].tolist()}
        return out

    def _redistance(self, scene, live):
        if self.redist_mode == 'jfa':
            return self._redistance_jfa(scene, live)
        if self.redist_mode == 'scale':
            return self._redistance_scale(scene, live)
        if self.redist_mode == 'scale2':
            return self._redistance_scale2(scene, live)
        return self._redistance_newton(scene, live)

    def _redistance_jfa(self, scene, live):
        """r41: jump-flood closest-point transform over the stroke's bricks (sculpt.cu jfa_*), then the aprons."""
        cp = self.cp
        G = self.view.gpu
        out_st = {}
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            o = scene.objects[oi]
            ov = self.view.mirror.objs[o.id]
            rows = np.array(list(rk.keys()), np.int64)
            keys = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
            dabs = np.array(live.get('dabs', {}).get(oi, []), np.float32).reshape(-1, 4)
            if len(dabs) > 512:
                dabs = dabs[np.linspace(0, len(dabs) - 1, 512).astype(int)]
            if len(dabs) == 0:
                continue
            h = float(o.sculpt.h)
            nb = len(rows)
            d_rowmap, shp, kO = self._rowmap_slice(ov, keys)   # fs-brush 11-12 (F5): the batch's slice of the map (keys +-1 cell)
            regmap = cp.full(G.used + 1, -1, cp.int32)
            regmap[cp.asarray(rows)] = cp.arange(nb, dtype=cp.int32)
            d_rows = cp.asarray(rows.astype(np.int32))
            d_keys = cp.asarray((keys - kO).astype(np.int32).reshape(-1))   # relative to the slice
            sa = cp.zeros((nb * 512, 4), cp.float32)
            sb = cp.zeros((nb * 512, 4), cp.float32)
            th = 256
            nt = (nb * 512 + th - 1) // th
            dims = (np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2]))
            self.k_jfa_seed((nt,), (th,), (G.sdf, d_rowmap) + dims + (d_rows, d_keys, np.int32(nb), np.float32(h), sa))
            for step in (16, 8, 4, 2, 1, 1):
                self.k_jfa_step((nt,), (th,), (d_rowmap,) + dims + (regmap, d_keys, np.int32(nb), np.int32(step), sa, sb))
                sa, sb = sb, sa
            st = cp.zeros(2, cp.int32)
            org0 = cp.asarray(np.asarray(np.asarray(ov.vol.origin, np.float64) + kO * 8 * h, np.float32))   # the slice's
            self.k_jfa_apply((nt,), (th,), (G.sdf, d_rows, d_keys, np.int32(nb), np.float32(h), org0,
                                            cp.asarray(dabs.reshape(-1)), np.int32(len(dabs)), sa, st))
            self._aprons(ov, keys)
            s_ = st.get()
            out_st[o.name] = {'bricks': int(nb), 'no_seed': int(s_[0])}
        return out_st

    def _redistance_scale(self, scene, live):
        """11-09 T4 candidate: sculpt.cu redist_scale over the stroke's bricks, then their aprons"""
        cp = self.cp
        G = self.view.gpu
        n = 0
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            o = scene.objects[oi]
            ov = self.view.mirror.objs[o.id]
            rows = np.array(list(rk.keys()), np.int64)
            keys = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
            h = float(o.sculpt.h)
            G._tables()
            vidx = G.vols.index(ov.vol)
            d_rows = cp.asarray(rows.astype(np.int32))
            d_org = cp.asarray((ov.vol.origin + keys * 8 * h).astype(np.float32).reshape(-1))
            out = cp.empty(len(rows) * 729, cp.float32)
            th = 256
            nt = (len(rows) * 729 + th - 1) // th
            self.k_redist_scale((nt,), (th,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL),
                                               d_rows, d_org, np.int32(len(rows)), out))
            self.k_commit((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), out))
            self._aprons(ov, keys)
            n += len(rows)
        return {'rows': n}

    def _redistance_scale2(self, scene, live):
        """11-09 T4 candidate 'scale2' (sculpt.cu grad_mag + scale_smooth) over the stroke's bricks, then their aprons;
        it measures itself (the r64 gate): the surface's crossings on lattice edges before / after the pass (its own
        shift) and |grad F| at them after, into self.t4_meas"""
        cp = self.cp
        G = self.view.gpu
        meas = {'shift_um': [], 'grad_after': []}
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            o = scene.objects[oi]
            ov = self.view.mirror.objs[o.id]
            rows = np.array(list(rk.keys()), np.int64)
            keys = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
            h = float(o.sculpt.h)
            G._tables()
            vidx = G.vols.index(ov.vol)
            d_rows = cp.asarray(rows.astype(np.int32))
            d_org = cp.asarray((ov.vol.origin + keys * 8 * h).astype(np.float32).reshape(-1))
            th = 256
            nt = (len(rows) * 729 + th - 1) // th
            before = self._gather(rows) if os.environ.get('FS_T4_MEASURE') else None
            gm = cp.empty(len(rows) * 729, cp.float32)
            self.k_grad_mag((nt,), (th,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL),
                                           d_rows, d_org, np.int32(len(rows)), gm))
            out = cp.empty(len(rows) * 729, cp.float32)
            self.k_scale_smooth((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), gm, out))
            self.k_commit((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), out))
            self._aprons(ov, keys)
            if before is not None:
                after = self._gather(rows)
                A = before.reshape(-1, 9, 9, 9); B_ = after.reshape(-1, 9, 9, 9)
                for ax in range(3):
                    a0 = np.take(A, range(0, 8), axis=ax + 1); a1 = np.take(A, range(1, 9), axis=ax + 1)
                    b0 = np.take(B_, range(0, 8), axis=ax + 1); b1 = np.take(B_, range(1, 9), axis=ax + 1)
                    cross = (np.sign(a0) != np.sign(a1)) & (np.sign(b0) != np.sign(b1)) & (a0 != a1) & (b0 != b1)
                    ta = a0[cross] / (a0[cross] - a1[cross]); tb = b0[cross] / (b0[cross] - b1[cross])
                    meas['shift_um'].append(np.abs(ta - tb) * h * 1e6)
                    meas['grad_after'].append(np.abs(b1[cross] - b0[cross]) / h)
        if meas['shift_um']:
            sh = np.concatenate(meas['shift_um']); ga = np.concatenate(meas['grad_after'])
            self.t4_meas = getattr(self, 't4_meas', []) + [(sh, ga)]
        return {'rows': int(sum(len(r) for r in live['rows'].values()))}

    def _redistance_newton(self, scene, live):
        """r23: re-distance the stroke's bricks on the card (sculpt.cu redist), then refresh their aprons."""
        cp = self.cp
        G = self.view.gpu
        out_st = {}
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            o = scene.objects[oi]
            ov = self.view.mirror.objs[o.id]
            rows = np.array(list(rk.keys()), np.int64)
            keys = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
            dabs = np.array(live.get('dabs', {}).get(oi, []), np.float32).reshape(-1, 4)
            if len(dabs) > 512:                     # a long stroke: every n-th dab (consecutive dabs overlap by 3/4)
                dabs = dabs[np.linspace(0, len(dabs) - 1, 512).astype(int)]
            if len(dabs) == 0:
                continue
            h = float(o.sculpt.h)
            G._tables()
            vidx = G.vols.index(ov.vol)
            d_rows = cp.asarray(rows.astype(np.int32))
            d_org = cp.asarray((ov.vol.origin + keys * 8 * h).astype(np.float32).reshape(-1))
            out = cp.empty(len(rows) * 729, cp.float32)
            st = cp.zeros(2, cp.int32)
            th = 256
            nt = (len(rows) * 729 + th - 1) // th
            self.k_redist((nt,), (th,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL), d_rows,
                                         d_org, np.int32(len(rows)), cp.asarray(dabs.reshape(-1)), np.int32(len(dabs)), out, st))
            self.k_commit((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), out))
            self._aprons(ov, keys)
            s = st.get()
            out_st[o.name] = {'bricks': int(len(rows)), 'not_converged': int(s[0]), 'too_far': int(s[1])}
        return out_st

    def _release_norm(self, scene, live):
        """fs-brush 11-11 (M, head test h13: over stage 2's ~90 Grab pulls |grad F| at the surface reached 0.7-1.5+, and
        the next pulls, reading that warped field, left streaks and a faceted flange; the live re-distance keeps the
        surface layer as it is, because rewriting it every batch flaked or grew noise: a loop): once, at the stroke's
        release, the stroke's rows re-distanced with the surface layer too (its seeds), blended by the stroke's footprint
        (its dab spheres, thinned to one per quarter radius), so each stroke starts from a distance field."""
        cp = self.cp
        t0 = time.perf_counter()
        n_rows = 0
        for oi, rk in live['rows'].items():
            dl = live.get('dabs', {}).get(oi) or []
            if not rk or not dl:
                continue
            rows = np.fromiter(rk.keys(), np.int64, len(rk))
            keys = np.array(list(rk.values()), np.int64).reshape(-1, 3)
            kept = []
            for d in np.asarray(dl, np.float32).reshape(-1, 4):
                if not kept or float(np.min(np.linalg.norm(np.asarray(kept)[:, :3] - d[:3], axis=1))) > 0.25 * float(d[3]):
                    kept.append(d)
            bp0 = np.zeros(BR.NBP, np.float32)
            pseudo = {'oi': oi, 'rows': rows, 'keys': keys, 'bp': bp0, 'mask': bool(live.get('mask', {}).get(oi, False))}
            self._footprint_redist(scene, [pseudo], dabs_over={oi: np.asarray(kept)}, iface_rw=True)
            self._aprons_card(self.view.mirror.objs[scene.objects[oi].id], keys)
            n_rows += len(rows)
        cp.cuda.Stream.null.synchronize()
        self.stats['release_norm'] = {'rows': n_rows, 'ms': round((time.perf_counter() - t0) * 1000, 1)}

    def finish(self, scene, rec):
        """the stroke's end (_finish_core), then the scratch arena trimmed (fs-brush 11-12, F5)"""
        try:
            return self._finish_core(scene, rec)
        finally:
            self._trim_scratch()

    def abort(self):
        """fs-brush 11-12 (fs-bench-full probe5b: two strokes whose first batch failed on the card made no history step, yet
        that batch had already changed card rows, and the next stroke started a fresh live state over them: a change on
        the card that no undo, save or write-back knows about). The stroke's every touched row gets its before-image back
        (the 729 values gathered before its first dab: aprons included) and its live state and result are dropped, so a
        stroke that never entered the history leaves nothing behind. -> rows restored"""
        live = self._live
        self._live = None
        self._result = {}
        if live is not None and getattr(self, '_pwq', None):   # fs-brush-2 11-13: this stroke's queued write goes with it
            self._pwq = [x for x in self._pwq if x['cmd'].rec is not live.get('rec')]
        self._pw_flags()
        try:
            self.view.gpu.mv_clear()               # fs-brush 11-13: the move view off
        except Exception:
            pass
        if live is None:
            return 0
        cp = self.cp
        G = self.view.gpu
        n = 0
        done_ = set()
        for oi, bl in (live.get('blk') or {}).items():
            # fs-brush-2 11-13: the first-touch blocks (a stroke shown through the tracer keeps no per-row 'before' dict)
            for rws, kws, F in bl:
                if len(rws):
                    G.sdf[cp.asarray(np.asarray(rws, np.int64))] = cp.asarray(np.asarray(F).astype(np.float16))
                    n += len(rws)
            done_.add(oi)
        for oi, bm in (live.get('before') or {}).items():
            if not bm or oi in done_:
                continue
            rows = np.fromiter(bm.keys(), np.int64, len(bm))
            F = np.stack([bm[r] for r in rows.tolist()]).astype(np.float16)
            G.sdf[cp.asarray(rows)] = cp.asarray(F)
            n += len(rows)
        cp.cuda.Stream.null.synchronize()
        return n

    def _trim_scratch(self):
        """fs-brush 11-12 (F5: card memory grew with each torso-scale stroke and was never returned, then a
        body Smooth ran the card out): the arena's buffers keep the largest batch's size x 1.5 for good. At a stroke's end any
        buffer larger than SCRATCH_KEEP_MB goes back to CuPy's pool (whose idle release returns it to the card); the small
        ones stay, so ordinary strokes still reuse theirs"""
        d = getattr(self, '_scratch', None)
        if not d:
            return
        lim = SCRATCH_KEEP_MB * 2 ** 20
        for k, b in list(d.items()):
            if b is not None and b.nbytes > lim:
                d[k] = None

    def _finish_core(self, scene, rec):
        """The stroke's end: write delta += F_end - F_start of every brick it touched into the CPU sculpt layer,
        through rec (snapshots / new bricks, as brush.apply_sample does). -> ms."""
        live = self._live
        self._live = None
        if live is None or live['rec'] is not rec:
            return 0.0
        t0 = time.perf_counter()
        if not hasattr(self, '_result'):
            self._result = {}
        mirror = self.view.mirror
        kn_ = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        light = (LIGHT_SYNC and live.get('key0') is not None and kn_ == live['key0']
                 and getattr(self.view, '_sync_key', None) == kn_)
        wrote = False
        if RD_RELEASE and self.live_redist and live.get('tool') != 'mask':
            self._release_norm(scene, live)
        if self.redistance:
            self.stats['redist'] = self._redistance(scene, live)
            if live.get('collide'):
                # 11-09: the collider rule again after the re-distance (CPU r2: it moved clamped nodes back 36-62 um)
                from ..core import collide as CL
                stg = live['collide']
                pk = scene.pack()
                for oi, rk in live['rows'].items():
                    obj = scene.objects[oi]
                    parts = CL.colliders(scene, obj.id, stg.get('contact_ids'))
                    if parts and rk:
                        self._collide_rows(scene, obj, oi, parts, rk, mirror.objs[obj.id].vol, live,
                                           float(stg.get('collide_gap', CL.GAP)), pk)
                        self._aprons(mirror.objs[obj.id], _ukeys(np.array(list(rk.values()), np.int64).reshape(-1, 3)))
        for oi, rk in live['rows'].items():
            if not rk:
                continue
            _tg = time.perf_counter()
            ov = mirror.objs[scene.objects[oi].id]
            blk = live.get('blk', {}).get(oi)
            if blk and sum(len(b[0]) for b in blk) == len(rk):
                rows = np.concatenate([b[0] for b in blk])
                ks = np.concatenate([b[1] for b in blk]).reshape(-1, 3)
                Fb = np.concatenate([b[2] for b in blk])
            else:                                  # (a redistance mode may have added rows outside the blocks)
                rows = np.array(list(rk.keys()), np.int64)
                ks = np.array([rk[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
                Fb = np.stack([live['before'][oi][r] for r in rows.tolist()])
            now = ov.rowmap[ks[:, 0], ks[:, 1], ks[:, 2]]
            if np.any(now != rows):               # r19: a row reassigned during the stroke would write garbage
                self.stats['row_alias'] = int(np.sum(now != rows))
                keep = now == rows
                rows, ks, Fb = rows[keep], ks[keep], Fb[keep]
                if not len(rows):
                    continue
            Fa = self._gather(rows)
            self.stats.setdefault('wb', {})['gather_ms'] = (time.perf_counter() - _tg) * 1000
            self.stats['wb']['rows'] = int(len(rows))
            if TOPO_CHECK and live.get('tool') != 'mask':
                try:
                    rows, ks, Fb, Fa = self._stroke_topo(scene, oi, ov, rows, ks, Fb, Fa)
                except Exception as e:                    # the check must never cost the stroke
                    self.topo_event = {'oid': scene.objects[oi].id, 'error': '%s: %s' % (type(e).__name__, e)}
            wrote = (self._write_back(scene, oi, rows, ks, Fb, Fa, rec) is not None) or wrote
            # r34/r37: the coarse level over the stroke's region, from the now-current CPU field - queued for the
            # viewport's next idle frame (it only steers light rays; refreshing it here delayed the next stroke)
            mirror.coarse_pending.append((ov.oid, ks.copy()))
        if light and wrote:
            # fs-brush 11-12 (brush-study STUDY 1e: the next frame's full sync, ~280 ms at 226 k slots): every brick the
            # stroke wrote is marked current and nothing else changed: the mirror's light sync now, the key taken as synced
            self.view._sync(scene, known_current=True)
        self.stats['light_sync'] = bool(light and wrote)
        ms = (time.perf_counter() - t0) * 1000
        self.stats['finish_ms'] = ms
        if os.environ.get('FS_GPU_SCULPT_CHECK'):
            self.stats['full_check'] = self.full_check(scene)
        self.history.append(dict(self.stats, wb_check_max_um=getattr(self, '_wbc', None), rows=sum(len(r) for r in live['rows'].values()),
                                 rebuilds=getattr(self, 'rebuilds', 0)))
        self.stats.pop('row_alias', None)
        return ms

    def _stroke_topo(self, scene, oi, ov, rows, ks, Fb, Fa):
        """fs-brush 11-11 (design D4, the outcome: whatever is sculpted, nothing that existed before a stroke is broken by it;
        no hidden pocket is left that the user did not make; a hole closes or opens only on purpose; no labels). At the
        stroke's end, from its own before / after node values (the lattice's signs, inside 26- / outside 6-connected:
        core/topo.py), over the box of the nodes whose sign flipped + 2 nodes (topology can change only where signs
        flipped; the box's border is the same before and after), the parts / tunnels / cavities before and after.
        - an enclosed pocket the stroke made (outside nodes it sealed off from the outside) is FILLED here, in the
          stroke's own values (its undo and its exact replay carry the fill);
        - a part fused or split, a tunnel closed or opened, an existing cavity lost: reported for the window's prompt
          (fs-shell's ui/topo_ui.py: Keep / Undo); the stroke itself is not changed.
        -> (rows, ks, Fb, Fa), with the filled rows' values (and any untouched rows the pocket reached) added; sets
        self.topo_event (the engine emits topoChanged) when anything changed."""
        from ..core import topo as TP
        from scipy import ndimage as ndi
        t0 = time.perf_counter()
        obj = scene.objects[oi]
        h = float(obj.sculpt.h)
        q8 = np.array([(i * 9 + j) * 9 + k for i in range(8) for j in range(8) for k in range(8)], np.int64)
        ijk = np.array([(i, j, k) for i in range(8) for j in range(8) for k in range(8)], np.int64)
        sb = Fb[:, q8] < 0.0
        sa = Fa[:, q8] < 0.0
        fl = sb != sa
        rf = np.nonzero(fl.any(1))[0]
        self.stats['topo'] = {'flips': int(fl.sum())}
        if not len(rf):
            return rows, ks, Fb, Fa
        P = (ks[rf][:, None, :] * 8 + ijk[None]).reshape(-1, 3)[fl[rf].reshape(-1)]
        lo = P.min(0) - 2
        hi = P.max(0) + 2
        shp = hi - lo + 1
        if int(np.prod(shp)) > TOPO_MAX_NODES:
            self.stats['topo'].update(skipped=int(np.prod(shp)))
            return rows, ks, Fb, Fa
        gb = np.zeros(shp, bool)
        ga = np.zeros(shp, bool)
        known = np.zeros(shp, bool)
        rowpos = {}
        for n in range(len(rows)):                  # the stroke's own bricks: before / after
            o_ = ks[n] * 8 - lo
            a0 = np.maximum(o_, 0); a1 = np.minimum(o_ + 8, shp)
            if np.any(a1 <= a0):
                continue
            s0 = a0 - o_; s1 = a1 - o_
            B = sb[n].reshape(8, 8, 8)[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
            A = sa[n].reshape(8, 8, 8)[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
            gb[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] = B
            ga[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] = A
            known[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] = True
            rowpos[int(rows[n])] = n
        # the rest of the box (unchanged by the stroke): its rows on the card, else the CPU field's sign at the cell
        cl = np.stack(np.meshgrid(*[np.arange(lo[a] >> 3, (hi[a] >> 3) + 1) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
        have = {tuple(k) for k in ks.tolist()}
        miss = np.array([c for c in cl.tolist() if tuple(c) not in have], np.int64).reshape(-1, 3)
        if len(miss):
            inb = np.all((miss >= 0) & (miss < np.array(ov.rowmap.shape)), 1)
            rr = np.full(len(miss), -1, np.int64)
            rr[inb] = ov.rowmap[miss[inb, 0], miss[inb, 1], miss[inb, 2]]
            on = rr >= 0
            vals = self._gather(rr[on]) if on.any() else np.zeros((0, 729), np.float32)
            sg = np.zeros((len(miss), 512), bool)
            sg[on] = vals[:, q8] < 0.0
            if (~on).any():
                from ..core.program import eval_points_local
                pk = scene.pack()
                ctr = np.ascontiguousarray(ov.vol.origin + (miss[~on] + 0.5) * 8 * h, np.float64)
                fc = np.empty(len(ctr))
                eval_points_local(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi, ctr, fc)
                sg[~on] = (fc < 0.0)[:, None]
            for n in range(len(miss)):
                o_ = miss[n] * 8 - lo
                a0 = np.maximum(o_, 0); a1 = np.minimum(o_ + 8, shp)
                if np.any(a1 <= a0):
                    continue
                s0 = a0 - o_; s1 = a1 - o_
                V = sg[n].reshape(8, 8, 8)[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
                gb[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] = V
                ga[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] = V
        RS = np.array(ov.rowmap.shape)

        def _row(k_):
            k_ = np.asarray(k_, np.int64)
            return int(ov.rowmap[k_[0], k_[1], k_[2]]) if np.all(k_ >= 0) and np.all(k_ < RS) else -1
        bb = TP.betti(gb)
        ba = TP.betti(ga)
        ev = None
        filled = 0
        if (bb['b0'], bb['b1'], bb['b2']) != (ba['b0'], ba['b1'], ba['b2']):
            # pockets the stroke made: outside components of 'after' not reaching the box's border whose nodes were
            # connected to the border before (outside then, or inside then and now enclosed air)
            lab_a, na = ndi.label(~np.pad(ga, 1))
            lab_b, nb_ = ndi.label(~np.pad(gb, 1))
            border_b = set(np.unique(np.concatenate([lab_b[0].ravel(), lab_b[-1].ravel(), lab_b[:, 0].ravel(), lab_b[:, -1].ravel(),
                                                     lab_b[:, :, 0].ravel(), lab_b[:, :, -1].ravel()]))) - {0}
            border_a = set(np.unique(np.concatenate([lab_a[0].ravel(), lab_a[-1].ravel(), lab_a[:, 0].ravel(), lab_a[:, -1].ravel(),
                                                     lab_a[:, :, 0].ravel(), lab_a[:, :, -1].ravel()]))) - {0}
            fill = np.zeros(shp, bool)
            for c in range(1, na + 1):
                if c in border_a:
                    continue
                m = lab_a[1:-1, 1:-1, 1:-1] == c
                prev = lab_b[1:-1, 1:-1, 1:-1][m]
                old_cav = set(np.unique(prev[prev > 0]).tolist()) - border_b
                if old_cav:
                    continue                          # it holds (part of) a cavity that existed: kept, never filled
                fill |= m                             # a pocket the stroke made: air it sealed off, or new air inside
            if fill.any():
                # into the stroke's values: those nodes inside, half a cell deep (sign only; the surface around them is
                # gone); rows the pocket reached outside the stroke's bricks are added (their before = now)
                Pf = np.argwhere(fill) + lo
                kb = Pf >> 3
                loc = Pf & 7
                qf = (loc[:, 0] * 9 + loc[:, 1]) * 9 + loc[:, 2]
                ukb = np.unique(kb, axis=0)
                add_r, add_k = [], []
                for k_ in ukb.tolist():
                    r_ = _row(k_)
                    if r_ < 0:
                        continue
                    if r_ not in rowpos:
                        add_r.append(r_); add_k.append(k_)
                if add_r:
                    G_ = self._gather(np.array(add_r, np.int64))
                    for i_, r_ in enumerate(add_r):
                        rowpos[r_] = len(rows) + i_
                    rows = np.concatenate([rows, np.array(add_r, np.int64)])
                    ks = np.concatenate([ks, np.array(add_k, np.int64).reshape(-1, 3)])
                    Fb = np.concatenate([Fb, G_])
                    Fa = np.concatenate([Fa, G_.copy()])
                for (k0, k1, k2), q in zip(kb.tolist(), qf.tolist()):
                    r_ = _row((k0, k1, k2))
                    n = rowpos.get(r_)
                    if n is None:
                        continue
                    Fa[n, q] = min(float(Fa[n, q]), -0.5 * h)
                    filled += 1
                # the card gets the filled values too (rows_restore), then their aprons
                ch = sorted({rowpos[_row(k_)] for k_ in kb.tolist() if _row(k_) in rowpos})
                if ch:
                    cp = self.cp
                    ch = np.array(ch, np.int64)
                    buf = cp.asarray(Fa[ch].astype(np.float16).reshape(-1))
                    th = 256
                    self.k_restore(((len(ch) * 729 + th - 1) // th,), (th,),
                                   (self.view.gpu.sdf, cp.asarray(rows[ch].astype(np.int32)), np.int32(len(ch)), buf))
                    self._aprons_card(ov, ks[ch])
                    self.view.note_dirty([self._world_box(obj, ov, ks[ch], h)])
            from ..tools.protected_merge import _to_world
            org = ov.vol.origin
            Wlo = _to_world(obj, (org + lo * h)[None])[0]; Whi = _to_world(obj, (org + hi * h)[None])[0]
            cav = TP.cavities(ga & ~fill if fill.any() else ga, h, org + lo * h)
            for c in cav:                            # local -> world
                c['centroid_m'] = _to_world(obj, np.asarray(c['centroid_m'])[None])[0].tolist()
                lo_c, hi_c = c['bbox_m']
                c['bbox_m'] = (_to_world(obj, np.asarray(lo_c)[None])[0].tolist(), _to_world(obj, np.asarray(hi_c)[None])[0].tolist())
            ga2 = ga | fill if fill.any() else ga
            b2 = TP.betti(ga2) if fill.any() else ba
            ev = {'oid': obj.id, 'before': bb, 'after': b2, 'after_raw': ba,
                  'box_m': (np.minimum(Wlo, Whi).tolist(), np.maximum(Wlo, Whi).tolist()),
                  'cavities': cav, 'filled': bool(filled), 'filled_nodes': int(filled),
                  'opening_closed': bool(b2['b1'] < bb['b1'] or b2['b0'] != bb['b0'] or b2['b2'] < bb['b2']),
                  'tunnels_new': max(0, b2['b1'] - bb['b1']), 'ms': round((time.perf_counter() - t0) * 1000, 1)}
            self.topo_event = ev
        self.stats['topo'].update(nodes=int(np.prod(shp)), before=bb, after=ba, filled=filled,
                                  ms=round((time.perf_counter() - t0) * 1000, 1))
        return rows, ks, Fb, Fa

    def _world_box(self, obj, ov, keys, h):
        from ..tools.protected_merge import _to_world
        k = np.asarray(keys, np.int64).reshape(-1, 3)
        lo = ov.vol.origin + k.min(0) * 8 * h; hi = ov.vol.origin + (k.max(0) + 1) * 8 * h
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = _to_world(obj, C)
        return (W.min(0), W.max(0))

    def _sheet_masks(self, scene, pl, live, settings):
        """fs-brush 11-12, the sheet under the brush on the card: one batch's dabs on one object labelled together on the
        host from their rows' first-touch values (live['before'], the field at the stroke's start: the CPU field is not
        touched during a card stroke, nor is a row before its first gather) by core/brush.py's _sheet_flood (the CPU
        path's rule), seeded at every dab's press point (and mirror partner): one flood a batch, not one a dab.
        Sets p['smask'] = uint8 (rows * 729), 1 = that dab may change the node; accumulates per row the nodes the reach
        saw that are not on the sheet (live['sheet_keep'], restored after the live re-distance: _sheet_restore)."""
        _tp = [time.perf_counter()]
        pl = [p for p in pl if len(p['rows'])]
        if not pl:
            return
        oi = pl[0]['oi']
        bef = live['before'][oi]
        bm = float(pl[0]['bm'])
        h = bm / 8.0
        org = np.asarray(pl[0]['vol'].origin, np.float64)
        rows = np.concatenate([p['rows'] for p in pl]).astype(np.int64)
        keys = np.concatenate([np.asarray(p['keys'], np.int64).reshape(-1, 3) for p in pl])
        rows, ix = np.unique(rows, return_index=True)
        keys = keys[ix]
        vals = np.stack([bef[int(rw)] for rw in rows])
        # a dab with a skip reach (bp[28]: Smooth's band, the skip tools' move + band) never writes a node beyond it, so
        # a row with no node within that reach needs no label (its mask stays 1: nothing there is written anyway)
        skips = [float(p['bp'][BR.NBP_SKIP]) for p in pl]
        if min(skips) > 0.0:
            near_ = np.abs(vals).min(1) <= max(skips) + 2 * h
            rows, keys, vals = rows[near_], keys[near_], vals[near_]
            if not len(rows):
                return
        n = len(rows)
        kmin = keys.min(0)
        dims = tuple(int(v) for v in (keys.max(0) - kmin) * 8 + 9)
        box = np.full(dims, np.nan, np.float64)
        off = np.ascontiguousarray((keys - kmin) * 8)
        _tp.append(time.perf_counter())
        BR.sheet_box_fill(box, vals, off)
        _tp.append(time.perf_counter())
        sym = settings.get('symmetry') or [False, False, False]
        seeds, norms = [], []
        rmax, amax = 0.0, 0.0
        for p in pl:
            bp = p['bp']
            n0 = np.asarray(bp[4:7], np.float64)
            n0 = n0 / max(float(np.linalg.norm(n0)), 1e-12)
            seeds.append(np.asarray(bp[1:4], np.float64) - org)
            norms.append(n0)
            if bp[BR.NBP_MIRP + 3] > 0.5:
                nm = n0.copy()
                for ax in range(3):
                    if sym[ax]:
                        nm[ax] = -nm[ax]
                seeds.append(np.asarray(bp[BR.NBP_MIRP:BR.NBP_MIRP + 3], np.float64) - org)
                norms.append(nm)
            rmax = max(rmax, float(bp[7]))
            amax = max(amax, abs(float(bp[13])))
        geo, cosc = BR.sheet_reach(settings, rmax, h)
        out = np.zeros(dims, np.uint8)
        i0, j0, k0 = (int(v) for v in kmin * 8)
        BR._sheet_flood(box, h, i0, j0, k0, np.ascontiguousarray(np.array(seeds)), np.ascontiguousarray(np.array(norms)),
                        rmax, geo, BR.SHEET_GMIN, cosc, BR.sheet_band(amax, h, settings), out)
        _tp.append(time.perf_counter())
        on = np.empty((n, 729), np.bool_)
        seen_off = np.empty((n, 729), np.bool_)
        BR.sheet_box_take(out, off, on, seen_off)
        _tp.append(time.perf_counter())
        pos = {int(rw): t for t, rw in enumerate(rows)}
        for p in pl:
            ix_ = np.fromiter((pos.get(int(rw), -1) for rw in p['rows']), np.int64, len(p['rows']))
            m_ = np.ones((len(ix_), 729), np.uint8)
            hit_ = ix_ >= 0
            m_[hit_] = on[ix_[hit_]]
            p['smask'] = m_.reshape(-1)
        _tp.append(time.perf_counter())
        kp = live.setdefault('sheet_keep', {}).setdefault(oi, {})
        on_ = live.setdefault('sheet_on', {}).setdefault(oi, {})
        # only rows that hold another sheet's nodes (now or in an earlier batch) need the per-row bookkeeping
        so_any = seen_off.any(1)
        if kp:
            so_any |= np.fromiter((int(rw) in kp for rw in rows), bool, n)
        for t in np.nonzero(so_any)[0]:
            rw = int(rows[t])
            o1 = on_.get(rw)
            o1 = on[t].copy() if o1 is None else (o1 | on[t])
            on_[rw] = o1
            k1 = kp.get(rw)
            k1 = seen_off[t] if k1 is None else (k1 | seen_off[t])
            kp[rw] = k1 & ~o1
        _tp.append(time.perf_counter())
        live['sheet_prof'] = [round((b - a) * 1000, 1) for a, b in zip(_tp[:-1], _tp[1:])]

    def _sheet_masks_card(self, scene, pl, live, settings):
        """fs-brush 11-12, the sheet under the brush labelled ON THE CARD (sculpt.cu sh_cells / sh_step / sh_label; the
        rule of core/brush.py _sheet_flood, from each row's first-touch values kept on the card for the stroke): one
        batch's dabs on one object together. p['smask'] (the batch's rows x 729, on the card) and p['smidx'] (the dab's
        rows -> batch rows); the restore set for _sheet_restore_card in live['sh_batch']."""
        cp = self.cp
        pl = [p for p in pl if len(p['rows'])]
        if not pl:
            return
        t0 = time.perf_counter()
        oi = pl[0]['oi']
        bm = float(pl[0]['bm'])
        h = bm / 8.0
        org = np.asarray(pl[0]['vol'].origin, np.float64)
        rows = np.concatenate([p['rows'] for p in pl]).astype(np.int64)
        keys = np.concatenate([np.asarray(p['keys'], np.int64).reshape(-1, 3) for p in pl])
        rows, ix = np.unique(rows, return_index=True)
        keys = keys[ix]
        n = len(rows)
        # the stroke's first-touch values on the card: rows not yet held are copied from the atlas now (their dabs have
        # not run: the tracking above gathered them this batch, before any write)
        sb = live.setdefault('sh_bdev', {}).get(oi)
        if sb is None:
            sb = live['sh_bdev'][oi] = {'buf': cp.empty(max(4096, 2 * n) * 729, cp.float16),
                                        'on': cp.zeros(max(4096, 2 * n) * 729, cp.uint8), 'pos': {}, 'n': 0}
        pos = sb['pos']
        if SHEET_FRONT:
            # fs-brush-4 11-13: the row -> buffer slot map as an array (the per-row Python loops cost ~1 ms per 1 000 rows)
            pa = sb.get('posa')
            need_ = int(rows.max()) + 1 if n else 0
            if pa is None or len(pa) < need_:
                pa2 = np.full(max(need_, 2 * (len(pa) if pa is not None else 0), 4096), -1, np.int32)
                if pa is not None:
                    pa2[:len(pa)] = pa
                pa = sb['posa'] = pa2
            new = rows[pa[rows] < 0]
        else:
            new = [int(rw) for rw in rows if int(rw) not in pos]
        if len(new):
            need = (sb['n'] + len(new)) * 729
            if need > sb['buf'].size:
                cap = max(need, int(sb['buf'].size * 1.5))
                b2 = cp.empty(cap, cp.float16); b2[:sb['n'] * 729] = sb['buf'][:sb['n'] * 729]
                o2 = cp.zeros(cap, cp.uint8); o2[:sb['n'] * 729] = sb['on'][:sb['n'] * 729]
                sb['buf'], sb['on'] = b2, o2
            m = len(new)
            th = 256
            dst = sb['buf'][sb['n'] * 729:(sb['n'] + m) * 729]
            self.k_gather(((m * 729 + th - 1) // th,), (th,), (self.view.gpu.sdf, cp.asarray(np.asarray(new, np.int32)),
                                                               np.int32(m), dst))
            if SHEET_FRONT:
                pa[new] = sb['n'] + np.arange(m, dtype=np.int32)
            else:
                for t_, rw in enumerate(new):
                    pos[rw] = sb['n'] + t_
            sb['n'] += m
        bpos = pa[rows].astype(np.int32) if SHEET_FRONT else np.fromiter((pos[int(rw)] for rw in rows), np.int32, n)
        kmin = keys.min(0) - 1
        kd = keys.max(0) - kmin + 2
        kbox = np.full(tuple(int(v) for v in kd), -1, np.int32)
        kbox[tuple((keys - kmin).T)] = np.arange(n, dtype=np.int32)
        # seeds: every dab's press point (and mirror partner) with its facing normal; the seed CELL is the nearest
        # surface along the brush's axis in the first-touch field (host, from live['before'])
        sym = settings.get('symmetry') or [False, False, False]
        seeds, norms, rmax, amax = [], [], 0.0, 0.0
        for p in pl:
            bp = p['bp']
            n0 = np.asarray(bp[4:7], np.float64)
            n0 = n0 / max(float(np.linalg.norm(n0)), 1e-12)
            seeds.append(np.asarray(bp[1:4], np.float64) - org); norms.append(n0)
            if bp[BR.NBP_MIRP + 3] > 0.5:
                nm = n0.copy()
                for ax in range(3):
                    if sym[ax]:
                        nm[ax] = -nm[ax]
                seeds.append(np.asarray(bp[BR.NBP_MIRP:BR.NBP_MIRP + 3], np.float64) - org); norms.append(nm)
            rmax = max(rmax, float(bp[7])); amax = max(amax, abs(float(bp[13])))
        geo, cosc = BR.sheet_reach(settings, rmax, h)
        # fs-core 11-14: the walk reaches the ball's edge with a margin (2 h left nodes on the worked surface near the
        # edge labelled 'another sheet': the dab wrote one neighbour and not the other, a step in the surface)
        reach2 = (rmax + SHEET_REACH_H * h) ** 2
        inner = max(0.5 * rmax, rmax - 4.0 * h)
        d_bpos = cp.asarray(bpos)
        d_bkey = cp.asarray(np.ascontiguousarray(keys.astype(np.int32)).reshape(-1))
        d_kbox = cp.asarray(kbox.reshape(-1)); d_kd = cp.asarray(kd.astype(np.int32)); d_kmin = cp.asarray(kmin.astype(np.int32))
        d_seeds = cp.asarray(np.ascontiguousarray(np.array(seeds, np.float32)).reshape(-1))
        d_norms = cp.asarray(np.ascontiguousarray(np.array(norms, np.float32)).reshape(-1))
        ns = len(seeds)
        th = 256
        cmix = self._scr('sh_cmix', n * 512, np.int8)
        cstep = self._scr('sh_cstep', n * 512, np.int32); cstep.fill(0)
        csid = self._scr('sh_csid', n * 512, np.int32); csid.fill(0)
        nt5 = (n * 512 + th - 1) // th
        self.k_sh_cells((nt5,), (th,), (sb['buf'], d_bpos, np.int32(n), cmix))
        # seeds on the card: the nearest surface facing the brush along each press point's axis (sh_seed)
        nts = 2 * int(rmax / h) + 2
        best = self._scr('sh_best', ns, np.uint32); best.fill(0xFFFFFFFF)
        self.k_sh_seed(((ns * nts + th - 1) // th,), (th,), (sb['buf'], d_bpos, d_kbox, d_kd, d_kmin, d_seeds, d_norms,
                                                            np.int32(ns), np.int32(nts), np.float32(h), best))
        if SHEET_FRONT:
            # fs-brush-4 11-13 (M, speed4/p1: the walk was 12-20 ms a batch at r 60-120, 5-9 ms at r 20-30, as up to
            # 1.6 (r + 2 h) / h launches): the whole walk in one launch (sculpt.cu sh_walk_1b, sh_step's rule step by
            # step); the seeds enter the first of its two frontier lists (counts fc3[0], fc3[1]; fc3[2] = steps taken)
            capf = n * 512
            fl = (self._scr('sh_fa', capf, np.int32), self._scr('sh_fb', capf, np.int32))
            fc3 = self._scr('sh_fcnt3', 3, np.int32); fc3.fill(0)
            self.k_sh_seed_set_f(((ns + 63) // 64,), (64,), (sb['buf'], d_bpos, d_kbox, d_kd, d_kmin, d_seeds, d_norms,
                                                             np.int32(ns), np.float32(h), best, cstep, csid, fl[0], fc3[0:1]))
            self.k_sh_walk_1b((1,), (512,), (sb['buf'], d_bpos, d_bkey, d_kbox, d_kd, d_kmin, cmix, cstep, csid,
                                             np.int32(geo), d_seeds, d_norms, np.int32(ns), np.float32(h),
                                             np.float32(reach2), np.float32(cosc), fl[0], fl[1], fc3, np.int32(capf)))
            k_ = -1                                # the step count stays on the card (fc3[2]): no host sync here
        else:
            self.k_sh_seed_set(((ns + 63) // 64,), (64,), (sb['buf'], d_bpos, d_kbox, d_kd, d_kmin, d_seeds, d_norms,
                                                           np.int32(ns), np.float32(h), best, cstep, csid))
        chg = self._scr('sh_chg', 1, np.int32); chg.fill(0)
        for k_ in (range(1, geo + 1) if not SHEET_FRONT else ()):
            self.k_sh_step((nt5,), (th,), (sb['buf'], d_bpos, d_bkey, d_kbox, d_kd, d_kmin, np.int32(n), cmix, cstep, csid,
                                           np.int32(k_), d_seeds, d_norms, np.int32(ns), np.float32(h),
                                           np.float32(reach2), np.float32(cosc), chg))
            if k_ % 8 == 0:                        # the walk has ended once 8 steps reached no new cell
                if int(chg.get()[0]) == 0:
                    break
                chg.fill(0)
        smask = self._scr('sh_smask', n * 729, np.uint8)
        keep = self._scr('sh_keep', n * 729, np.uint8)
        labo = self._scr('sh_labo', n * 729, np.uint8)
        self.k_sh_label(((n * 729 + th - 1) // th,), (th,), (sb['buf'], d_bpos, d_bkey, d_kbox, d_kd, d_kmin, np.int32(n),
                                                             cmix, cstep, d_seeds, np.int32(ns), np.float32(h),
                                                             np.float32(reach2), np.float32(inner * inner),
                                                             np.float32(BR.SHEET_GMIN), smask, keep, sb['on'],
                                                             np.int32(BR.sheet_side(settings, pl[0]['bp'])),
                                                             np.float32(BR.sheet_band(amax, h, settings)), labo))
        self.k_sh_fill0(((n * 729 + th - 1) // th,), (th,), (d_bkey, d_kbox, d_kd, d_kmin, np.int32(n), d_bpos, labo, smask,
                                                             keep, sb['on']))
        if SHEET_FRONT:
            # rows is sorted (np.unique): a dab's rows -> batch rows by searchsorted
            for p in pl:
                p['smask'] = smask
                p['smidx'] = cp.asarray(np.searchsorted(rows, np.asarray(p['rows'], np.int64)).astype(np.int32))
        else:
            pidx = {int(rw): t for t, rw in enumerate(rows)}
            for p in pl:
                p['smask'] = smask
                p['smidx'] = cp.asarray(np.fromiter((pidx[int(rw)] for rw in p['rows']), np.int32, len(p['rows'])))
        live.setdefault('sh_batch', {})[oi] = (cp.asarray(rows.astype(np.int32)), d_bpos, keep, n, sb['buf'])
        live['sheet_prof'] = {'n': n, 'geo': geo, 'steps': k_, 'host_ms': round((time.perf_counter() - t0) * 1000, 1)}

    def _sheet_restore_card(self, live):
        """fs-brush 11-12: after the live re-distance, the batch's nodes not on the sheet get their first-touch values back"""
        th = 256
        # fs-core 11-14: under the truncated band the protected nodes take the smaller of their stroke-start and current
        # magnitudes (sculpt.cu sh_restore_min); restoring the stroke-start values put back what the band had made true
        k_ = self.k_sh_restore_min if live.get('tsdf') else self.k_sh_restore
        for oi, (d_rows, d_bpos, keep, n, buf) in (live.pop('sh_batch', None) or {}).items():
            k_(((n * 729 + th - 1) // th,), (th,), (self.view.gpu.sdf, d_rows, np.int32(n), buf, d_bpos, keep))

    def _sheet_restore(self, plans, live):
        """fs-brush 11-12: after the live re-distance, the nodes the stroke's reach saw that are not on its sheet get
        their stroke-start values back (they were never written by a dab; only the re-distance could move them)"""
        cp = self.cp
        for oi, kp in (live.get('sheet_keep') or {}).items():
            rws = [rw for rw, m in kp.items() if m.any()]
            if not rws:
                continue
            bef = live['before'][oi]
            n = len(rws)
            vals = np.stack([bef[rw] for rw in rws]).astype(np.float32).reshape(-1)
            keep = np.stack([kp[rw] for rw in rws]).astype(np.uint8).reshape(-1)
            th = 256
            self.k_rrm(((n * 729 + th - 1) // th,), (th,), (self.view.gpu.sdf, cp.asarray(np.asarray(rws, np.int32)),
                                                           np.int32(n), cp.asarray(vals), cp.asarray(keep)))

    def _ft_rows(self, live, oi, rows):
        """fs-brush-3 11-13: the stroke-start (first-touch) values of these card rows -> (n, 729) float32, from the per-row
        dict live['before'] or the first-touch blocks live['blk'] (a stroke shown through the tracer keeps no per-row dict:
        its release's contact clamp read the dict and raised KeyError 0 on her face, the rows were put back and her cheek
        tore). A row in neither was never first-touched: a real fault, raised"""
        bm = (live.get('before') or {}).get(oi) or {}
        rows = [int(r) for r in rows]
        out = np.empty((len(rows), 729), np.float32)
        miss = []
        for i, r in enumerate(rows):
            v = bm.get(r)
            if v is None:
                miss.append(i)
            else:
                out[i] = np.asarray(v, np.float32).reshape(-1)
        if miss:
            bl = (live.get('blk') or {}).get(oi) or []
            ix = live.setdefault('ft_index', {}).get(oi)
            if ix is None or ix[0] != len(bl):
                m = {}
                for bi, (rws, _k, _F) in enumerate(bl):
                    for j, r in enumerate(np.asarray(rws).tolist()):
                        m.setdefault(int(r), (bi, j))
                ix = (len(bl), m)
                live['ft_index'][oi] = ix
            for i in miss:
                bi, j = ix[1][rows[i]]
                out[i] = np.asarray(bl[bi][2][j], np.float32).reshape(-1)
        return out

    def _ft_start_add(self, live, oi, rws, F):
        """fs-tools 11-14 (patch 3): the approved held tools read the field as it was at the press for Follow's anchors
        (toolset/ftools.cuh ft_rule2; sculpt.cu obj_f_start): each row's first-touch values go once into a per-object
        pool on the card (fp16, exactly the gathered values); slot: atlas row -> pool row (-1: untouched, so the atlas row
        itself still holds its press-time values)"""
        cp = self.cp
        nrows = int(self.view.gpu.sdf.size // 729)
        e = live.setdefault('ftb', {}).get(oi)
        if e is None or e['slot'].size < nrows:
            sl = cp.full(max(nrows, 1), -1, cp.int32)
            if e is not None:
                sl[:e['slot'].size] = e['slot']
            e = live['ftb'][oi] = {'slot': sl, 'pool': e['pool'] if e is not None else cp.zeros(0, cp.float16),
                                   'n': e['n'] if e is not None else 0}
        n0, k = e['n'], len(rws)
        if (n0 + k) * 729 > e['pool'].size:
            # amortised growth (capacity doubling): a long moving stroke copies the pool O(n) times in all, not per batch
            buf = cp.empty(max(2 * e['pool'].size // 729, n0 + k, 256) * 729, cp.float16)
            buf[:n0 * 729] = e['pool'][:n0 * 729]
            e['pool'] = buf
        e['pool'][n0 * 729:(n0 + k) * 729] = cp.asarray(np.asarray(F, np.float32).astype(np.float16).reshape(-1))
        e['slot'][cp.asarray(np.asarray(rws, np.int64))] = cp.arange(n0, n0 + k, dtype=cp.int32)
        e['n'] = n0 + k

    def _ft_start_args(self, oi):
        """(slot, nslot, pool, use) for ft_dab: the stroke's press-time pool of object oi, or a dummy (the field as it is)"""
        e = ((getattr(self, '_live', None) or {}).get('ftb') or {}).get(oi)
        if e is None or e['n'] == 0:
            d = getattr(self, '_ftb_dummy', None)
            if d is None:
                d = self._ftb_dummy = (self.cp.full(1, -1, self.cp.int32), self.cp.zeros(729, self.cp.float16))
            return (d[0], np.int32(1), d[1], np.int32(0))
        return (e['slot'], np.int32(e['slot'].size), e['pool'], np.int32(1))

    def _gather(self, rows):
        """the atlas rows' 9^3 values (fp16 on the card) -> (n, 729) float32 on the host.
        fs-brush 11-12 (Flatten 40 on her, probe5c / probe5d: the release ran the card out of memory allocating 426 MB,
        this buffer for all 195 k written rows at once, with 6.5 of 6.9 GB in use): gathered GATHER_ROWS at a time into
        one bounded buffer, each chunk copied into the host array as it comes (the same kernel per row: bit-identical);
        the host no longer holds a full fp16 copy beside the float32 one either"""
        cp = self.cp
        n = len(rows)
        out = np.empty((n, 729), np.float32)
        if n == 0:
            return out
        r32 = np.asarray(rows).astype(np.int32)
        th = 256
        for a in range(0, n, GATHER_ROWS):
            m = min(n, a + GATHER_ROWS) - a
            buf = self._scr('gather', m * 729, np.float16)
            self.k_gather(((m * 729 + th - 1) // th,), (th,),
                          (self.view.gpu.sdf, cp.asarray(r32[a:a + m]), np.int32(m), buf))
            out[a:a + m] = buf.get().reshape(m, 729)
        return out

    def _expand(self, scene, oid, settings, samples):
        """brush.apply_stroke's dab list: (object index, settings with radius_local / strength, c, n, v local)."""
        oi = scene.index_of(oid)
        obj = scene.objects[oi]
        sym = settings.get('symmetry', [False, False, False])
        world = bool(settings.get('sym_world', False))
        for smp in samples:
            c, n, v, s = BR.world_to_local(obj, smp['p'], smp.get('n', (0, 1, 0)), smp.get('v', (0, 0, 0)))
            st = dict(settings)
            st['radius_local'] = float(settings['radius']) / s
            if st['tool'] in BR.FT_TOOLS:             # fs-tools 11-14: held tools: the dab covers dt seconds of the stroke
                st['dt'] = float(smp.get('dt', 0.0))
                st['s_local'] = float(s)
            if st['tool'] in BR.FT_LINE:              # fs-tools patch 4: Pencil: its direction, the window's 'v' (the path's
                st['s_local'] = float(s)              # direction there), else from the tip's previous place (local)
                lv_ = getattr(self, '_live', None)
                pl_ = (lv_.setdefault('pencil_last', {}) if isinstance(lv_, dict) else {})
                if np.linalg.norm(v) > 0.0:
                    st['pencil_dir'] = np.asarray(v, float).tolist()
                elif pl_.get(oi) is not None:
                    st['pencil_dir'] = (np.asarray(c, float) - pl_[oi]).tolist()
                pl_[oi] = np.asarray(c, float)
            line_ = st['tool'] in BR.FT_LINE          # fs-tools patch 4: Pencil's depth is constant (by design): no pen
            if 'k' in smp and not line_:              # pressure, no partial-dab share (a batch's last dab is a full one)
                st['strength'] = float(settings['strength']) * float(smp['k'])
            if 'f' in smp and not line_:              # fs-brush 11-11: a partial dab's path share (brush.frac_of)
                st['frac'] = float(smp['f'])
                st['strength'] = float(st['strength']) * st['frac']
            if not world:
                # fs-brush 11-11 (design C4): midline snap (brush.mirror_dabs); 'mid_snap' off in older logs
                r_l = max(float(st['radius_local']), 2 * float(obj.sculpt.h))
                # fs-tools 11-14: the approved tools reach radius + Smoothing: the midline snap and the feather pair see that
                r_m = r_l + (float(st.get('smoothing', 0.0)) / s if st['tool'] in BR.FT_ALL else 0.0)
                # fs-brush 11-11: the midline snap + the feathered sideways pair (brush.mirror_plan)
                vps_ = BR.snake_vps(obj, smp, c, n, r_l, sym, settings, st['tool'])   # fs-brush 11-12: Rake
                for ci_, (c_, n_, v_, w_, pc_) in enumerate(BR.mirror_plan(c, n, v, r_m, sym, settings, st['tool'])):
                    if w_ >= 1.0:
                        st_w = st
                    else:
                        st_w = dict(st)
                        st_w['strength'] = float(st['strength']) * w_
                    if pc_ is not None:
                        st_w = dict(st_w)
                        st_w['mirror_partner'] = pc_
                    if vps_ is not None:
                        st_w = dict(st_w, snake_vp=vps_[ci_][0], snake_vr=vps_[ci_][1])
                    if st['tool'] in BR.FT_LINE and ci_ > 0 and np.linalg.norm(v_) > 0.0:
                        st_w = dict(st_w, pencil_dir=np.asarray(v_, float).tolist())   # a mirrored tip: its own direction
                    vv = v_ * w_ if st['tool'] in BR.MOVE_TOOLS and w_ < 1.0 else v_   # fs-brush-4 11-13: + the flows
                    yield oi, st_w, c_, n_, vv
                    a_s = float(settings.get('autosmooth', BR.AUTOSMOOTH.get(st['tool'], 0.0)))
                    if st['tool'] in BR.FT_ALL:            # fs-tools 11-14: their Smoothing is the profile's foot, no pass
                        a_s = 0.0
                    if a_s > 0.0:
                        # fs-brush 11-11 (design 4.7): autosmooth: the rebuilt Smooth over the dab's footprint, at the
                        # moved surface (Grab: the anchor's point after the drag); a Grab dab carries the share of a
                        # spacing step it travelled, so the smoothing per length of stroke matches a spaced brush
                        k = 1.0
                        c_s = c_
                        if st['tool'] in BR.MOVE_TOOLS:
                            k = min(float(np.linalg.norm(vv)) / max(float(settings.get('spacing', 0.25)) * r_l, 1e-12), 1.0)
                            c_s = c_ + vv
                        if k > 0.0:
                            st_s = dict(st, tool='smooth', strength=a_s * k * w_, smooth_quad=True, invert=False,
                                        radius_local=float(st['radius_local']) * BR.AUTOSMOOTH_R)
                            yield oi, st_s, c_s, n_, np.zeros(3)
                continue
            for gi, sg in enumerate(BR._symmetry_signs(sym)):
                if gi == 0:
                    yield oi, st, c * sg, n * sg, v * sg
                    continue
                pw = np.asarray(smp['p'], float) * sg
                ti = BR.mirror_target(scene, pw)
                if ti < 0:
                    continue
                o2 = scene.objects[ti]
                c2, n2, v2, s2 = BR.world_to_local(o2, pw, np.asarray(smp.get('n', (0, 1, 0)), float) * sg,
                                                   np.asarray(smp.get('v', (0, 0, 0)), float) * sg)
                st2 = dict(st)
                st2['radius_local'] = float(settings['radius']) / s2
                if 's_local' in st2:
                    st2['s_local'] = float(s2)
                yield ti, st2, c2, n2, v2

    def _plan(self, scene, oi, st, c, n, v, bp_only=False):
        o = scene.objects[oi]
        ov = self.view.mirror.objs.get(o.id)
        if ov is None or ov.vol is None:
            raise RuntimeError('GPU sculpt: object %d has no GPU volume' % o.id)
        vol = ov.vol
        scope_mg = None
        h = float(o.sculpt.h)
        r = max(float(st['radius_local']), 2 * h)
        c = np.asarray(c, float)
        n = np.asarray(n, float)
        n = n / max(np.linalg.norm(n), 1e-12)
        v = np.asarray(v, float)
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(n)) and np.all(np.isfinite(v))):
            return None
        snake_ = st['tool'] == 'grab' and st.get('snake', False) and BR.SNAKE_PATH
        if snake_:                                 # fs-brush 11-12: the dab covers the kernel's travel (brush.snake_dab)
            snake_c0, snake_rt = c, r
            c, r = BR.snake_dab(c, v, r)
        s = float(st['strength'])
        ft_ = st['tool'] in BR.FT_ALL
        if st['tool'] in BR.FT_LINE:
            # fs-tools patch 4: Pencil, the tip where it is (toolset/ftools.cuh ft_pencil): its own size (never the 2 h floor), the
            # ball = the tip + its depth twice (the press surface may lie a depth below the cursor's pick) + k + 2 h
            from ..toolset.ftools import ft_bp as _ft_bp
            ftv = _ft_bp('pencil', dict(st, radius_local=float(st['radius_local'])), h, 0.0,
                         float(st.get('s_local', 1.0)), direction=st.get('pencil_dir'))
            r = ftv[63] + 2.0 * ftv[55] + ftv[52] + 2.0 * h
            A = ftv[55]
        elif ft_:
            # fs-tools 11-14: the approved tools (toolset/ftools.ft_bp): Strength is a rate, the dab's amount = rate x dt;
            # the dab's ball = the profile's radius + Smoothing (the reach, admission and band pass all read bp[7])
            from ..toolset.ftools import ft_bp as _ft_bp
            ftv = _ft_bp(st['tool'][3:], dict(st, radius_local=r), h, float(st.get('dt', 0.0)),
                         float(st.get('s_local', 1.0)), c if st['tool'] == 'ft_flatten' else None)
            r = r + ftv[52]
            if st['tool'] == 'ft_flatten':           # the most a node moves toward the plane in this dab
                A = (1.0 - math.exp(-max(ftv[55], 0.0))) * r
            else:
                A = abs(ftv[55])
        else:
            A = BR.dab_amount(st, r)                 # 11-09 night: the amount per pass (core/brush.py), as the CPU
        if st.get('dab_gain', False) and not ft_:  # r67: the CPU brush's small-dab gain, the same numbers
            key = (st.get('falloff', 'smooth'), str(st.get('curve')))
            if getattr(self, '_glut_key', None) != key:
                self._glut_key, self._glut = key, BR.falloff_lut(st.get('falloff', 'smooth'), st.get('curve'))
            A, s = BR.gained(st, st['tool'], c, n, r, A, s, h, self._glut)
        if st.get('invert', False) and not ft_:     # fs-brush 11-11: Ctrl inverts every brush (brush.invert_dab)
            A, s = BR.invert_dab(st['tool'], A, s)
        band = vol.levels[LEVEL]['band']
        bm = 8 * h
        # r18: every node the CPU brush changes is changed here too, so the two paths agree everywhere (a dab that
        # changed only the band's bricks left steps against the unheld bricks beside them: L 3 -> 166 in the app):
        # all cells of the sphere are held, grown by the displacement reach of the tools that read displaced points
        move = A if st['tool'] not in ('grab', 'stretch', 'grabflow', 'stretchflow') else float(np.linalg.norm(v))
        disp = st['tool'] in ('draw', 'carve', 'grab', 'stretch', 'pinch', 'crease', 'grabflow', 'stretchflow') or ft_
        reach = r + h + (move if disp else 0.0)
        if FLOW_SUPPORT and (st['tool'] in ('grabflow', 'stretchflow') or snake_):
            reach = r + h                        # fs-brush 11-12 speed (a): the ball already bounds the flow's support
        if st['tool'] == 'smooth' and SMOOTH_RING_REACH > 0 and not getattr(ov, 'is_store', False):
            # fs-brush 11-12 (test 1's granular jaw, t1fix/card_jaw.py: from one saved state the card's Smooth took the jaw's
            # roughness 1.70 -> 1.78-1.85 where the CPU's took it to 1.52): the fit samples two rings of the tangent plane
            # (QSM_R1 / QSM_R2 r from each node), so a node near the dab's edge reads up to ~1.6 r from its centre; a read in
            # a brick the card does not hold returns nothing and drops that sample's pair, and which pairs drop changed from
            # node to node: scratches. The card now holds the rings' reach too (cj2: 1.85 -> 1.28, the same fps; test 1
            # t1_fsbrush_17: the chin's pitted patch gone). Not on her store object yet: there Smooth admits band cells
            # only (F5, the card's memory), and a band wide enough for the rings is to be measured first
            reach += SMOOTH_RING_REACH * BR.QSM_R2 * r
        scope_dc = None
        if bp_only:                               # fs-brush 11-13: the move view needs the dab's parameters only
            cells = admit = np.zeros((0, 3), np.int64)
        else:
            i0 = np.floor((c - reach - vol.origin) / bm).astype(np.int64)
            i1 = np.floor((c + reach - vol.origin) / bm).astype(np.int64) + 1
            nbx = np.array(vol.dims) * BPT
            if np.any(i0 < 0) or np.any(i1 > nbx):
                sb_ = None
                if GROW_SURF and st['tool'] in ('grabflow', 'stretchflow') and getattr(self, '_mv_clip', 0) > 0:
                    # fs-brush-3 11-13: an anchored move whose moved surface stays on the grid: the ball's cells past the
                    # grid hold only far values (no surface reaches them), so the plan keeps to the grid
                    mv_ = float(np.linalg.norm(v))
                    sb_ = _swept_surface_box(o, c - reach - mv_, c + reach + mv_, np.asarray(v, float), h)
                    g_lo_, g_hi_ = vol.origin, vol.origin + nbx * bm
                    if sb_ is not None and (np.any(sb_[0] - bm < g_lo_) or np.any(sb_[1] + bm > g_hi_)):
                        sb_ = None
                    elif sb_ is None:
                        sb_ = (c, c)
                if sb_ is None:
                    raise GridOverflow(o.id, 'GPU sculpt: the dab reaches past the volume grid of object %d (rebuild needed)' % o.id,
                                       (c - 2 * reach, c + 2 * reach))
                i0, i1 = np.maximum(i0, 0), np.minimum(i1, nbx)
            cells = np.stack(np.meshgrid(*[np.arange(i0[a], i1[a]) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
            ctr = vol.origin + (cells + 0.5) * bm
            cells = cells[np.linalg.norm(ctr - c, axis=1) <= reach + 0.8661 * bm]
            if st['tool'] == 'snakepath' and len(cells):
                # fs-brush 11-12: the composed Snake's cells: within the kernel radius of any of the path's segments
                cells = cells[BR.snake_path_near(vol.origin + (cells + 0.5) * bm, st['snake_steps_local'], h + 0.8661 * bm)]
            if FLOW_SUPPORT and st['tool'] in ('grabflow', 'stretchflow') and len(cells):
                # fs-brush 11-12 speed (a): only the cells the flow can change or read: within the moving radius of the
                # segment it sweeps (Grab: the kernel's centre path c0 -> c0 + v; Stretch: the axis from the pivot to the
                # fade's end), both centred on c. A node beyond it never moves, and a moved node's pre-image stays inside
                L_ = float(np.linalg.norm(v))
                a_ = v / max(L_, 1e-12)
                if st['tool'] == 'grabflow':
                    hl = 0.5 * L_
                else:
                    hl = 0.5 * r * (float(st.get('flow_D_ratio', 1.0)) + float(st.get('flow_fade_ratio', 1.0)))
                rs = r * float(st.get('flow_ratio', 1.0))
                q_ = vol.origin + (cells + 0.5) * bm - c
                t_ = np.clip(q_ @ a_, -hl, hl)
                cells = cells[np.linalg.norm(q_ - t_[:, None] * a_[None, :], axis=1) <= rs + h + 0.8661 * bm]
            have = ov.rowmap[cells[:, 0], cells[:, 1], cells[:, 2]] >= 0 if len(cells) else np.zeros(0, bool)
            admit = cells[~have]
            if TSDF and self.live_redist and st['tool'] != 'mask':
                # fs-core 11-14 (the truncated band): the held rows within 2 W of the ball take part in the batch's
                # re-distance (seeds for the nodes within W of it; clamped to +-W); none is admitted for it, and the frame's
                # dirty boxes stay the dab's own cells (dcells)
                Wt = TSDF_W_H * h
                rs_ = r + 2.0 * Wt + h
                j0 = np.maximum(np.floor((c - rs_ - vol.origin) / bm).astype(np.int64), 0)
                j1 = np.minimum(np.floor((c + rs_ - vol.origin) / bm).astype(np.int64) + 1, nbx)
                if np.all(j1 > j0):
                    cs_ = np.stack(np.meshgrid(*[np.arange(j0[a], j1[a]) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
                    cs_ = cs_[np.linalg.norm(vol.origin + (cs_ + 0.5) * bm - c, axis=1) <= rs_ + 0.8661 * bm]
                    cs_ = cs_[ov.rowmap[cs_[:, 0], cs_[:, 1], cs_[:, 2]] >= 0]
                    dcells_ = cells
                    cells = _ukeys(np.concatenate([cells, cs_])) if len(cs_) else cells
                    scope_dc = dcells_
                else:
                    scope_dc = cells
            else:
                scope_dc = None
        bp = np.zeros(BR.NBP, np.float32)
        bp[0] = BR.T[st['tool']]
        bp[1:4] = c
        bp[4:7] = n
        bp[7] = r
        bp[8] = s
        bp[9:12] = v
        bp[12] = BR.smooth_step(st, h, r)
        bp[13] = A
        bp[14] = 1.5 * band
        bp[15:20] = BR.target_params(st, h)
        BR.mirror_partner_bp(bp, st)             # fs-brush 11-11: mirror feather
        bp[BR.NBP_SHEET] = 1.0 if st.get('sheet_smooth', False) else 0.0
        bp[BR.NBP_LIM] = 1.0 if st.get('dab_limit', False) else 0.0
        if st['tool'] == 'smooth' and st.get('smooth_quad', False):     # fs-brush 11-11 (design C1)
            bp[BR.NBP_QSM] = 1.0
            bp[BR.NBP_QSM + 1:BR.NBP_QSM + 5] = (BR.QSM_BAND_H, BR.QSM_R1, BR.QSM_R2, BR.QSM_CAP_H)
            BR.smooth_cell_bp(bp, st)            # fs-brush 11-11: the fit's mode, its normal stencil, the cell-scale term
            bp[8] = math.copysign(min(abs(s) * float(st.get('spacing', 0.25)) / 0.25 * BR.QSM_PASS_K, 1.0), s)   # Ctrl: enhance
            if self.live_redist:                 # its own band: nodes beyond it never change (the re-distance band too)
                bp[BR.NBP_SKIP] = BR.QSM_BAND_H * h
        if st.get('strength_v2', False):          # fs-brush 11-11 (decided): strength rises smoothly, never clamps
            fr = BR.frac_of(st)
            sv = float(st['strength']) / fr      # a partial dab: the response of the whole strength, times its share
            if st.get('invert', False) and st['tool'] in ('flatten', 'pinch', 'smooth'):
                sv = -sv
            if st['tool'] in BR.STRENGTH_RESP:
                bp[8] = BR.strength_dab(st['tool'], sv, float(st.get('spacing', 0.25))) * fr
                bp[BR.NBP_STR] = 1.0
            if st['tool'] == 'smooth':
                bp[BR.NBP_QSM + 4] = BR.QSM_CAP_H * max(1.0, abs(sv))
            if st['tool'] == 'crease':
                bp[BR.NBP_STR] = 1.0
        if st['tool'] in ('clay', 'flatten') and st.get('area_plane', False):   # fs-brush 11-11: area plane on the card
            bp[BR.NBP_PLANE + 6] = 1.0
        if st['tool'] == 'pinch' and st.get('pinch_bound', False):
            bp[BR.NBP_PINCH] = BR.PINCH_K
        if st['tool'] == 'stretchflow':            # fs-brush 11-12: Stretch as an anchored elongation (sculpt.cu typ 15)
            BR.stretch_bp(bp, c, v, r, st)
        if snake_:
            BR.snake_bp(bp, st, snake_c0, snake_rt, v)   # fs-brush 11-12: Snake Hook's step as a flow (sculpt.cu typ 16)
        if st['tool'] == 'grabflow':               # fs-brush 11-12: the anchored flow Grab (sculpt.cu typ 14; brush.py)
            rt = r * float(st.get('flow_ratio', 1.0))
            bp[BR.NBP_PLANE:BR.NBP_PLANE + 3] = c - 0.5 * v
            bp[BR.NBP_PLANE + 3] = rt
            bp[BR.NBP_PLANE + 4] = min(64, max(4, int(math.ceil(float(np.linalg.norm(v)) / (0.15 * rt)))))
            bp[BR.NBP_PLANE + 6] = 0.0
            scope_mg = BR.scope_bp(bp, st)           # fs-brush 11-12: Grab's scope (sculpt.cu scope_at)
        if self.live_redist and st['tool'] in BR.SKIP_TOOLS and not os.environ.get('FSB_NOSKIP'):   # fs-brush 11-11: nodes beyond the move + band keep values
            bp[BR.NBP_SKIP] = abs(move) * 1.6 + BR.SKIP_BAND_H * h
            if SKIP_SIDE:
                # fs-brush 11-12: the card's node skip one-sided, as the CPU's band (c8040703): a dab that grows the form
                # (Draw, Inflate, Clay, Swell; Carve inverted) skips deep inside only, one that cuts in deep outside only
                grow = (A < 0) if st['tool'] == 'carve' else (A > 0)
                bp[BR.NBP_SIDE] = 1.0 if grow else -1.0
        # fs-brush 11-11: the keep-apart guard refuses the sign flips of a large drag it cannot order, and the refused
        # nodes keep their old values: air pockets and specks in a pulled neck (M, test 1 stage 2: guard on 6 inside parts
        # and 31 enclosed pockets; off: 1 and 0). Grab and Stretch run without it unless 'keep_apart_drag' is set.
        if (st['tool'] == 'swell' and st.get('swell_toward', False) and st.get('swell_mode', 'ease') == 'ease'
                and not st.get('invert', False)):
            # sculpt-contact 11-12: Swell to contact (core/collide.swell_k; the contact rows: gpu/collide_card.contact_rows)
            from ..core import collide as _CLs
            from ..tools.protected_merge import _rs as _rs_
            g_, E_, R_, gap_ = _CLs.swell_params(st, r, h, _rs_(o)[2])
            bp[BR.NBP_SWC:BR.NBP_SWC + 5] = (1.0, g_, E_, R_, gap_)
            if bp[BR.NBP_SKIP] > 0:                # its growth is up to gain x the amount
                bp[BR.NBP_SKIP] = abs(move) * g_ * 1.6 + BR.SKIP_BAND_H * h
        if ft_:
            bp[15:20] = 0.0                          # fs-tools 11-14: Reach is the sheet under the brush (bp gates), not targeting
            for k_, v_ in ftv.items():
                bp[k_] = v_
            if self.live_redist and not os.environ.get('FSB_NOSKIP'):
                # nodes farther from the surface than the dab's move + the band keep their values (the band pass rebuilds
                # them); Swell grows (skip deep inside), Settle cuts (deep outside), Flatten both ways
                bp[BR.NBP_SKIP] = abs(A) * 1.6 + BR.SKIP_BAND_H * h
                bp[BR.NBP_SIDE] = {'ft_swell': 1.0, 'ft_settle': -1.0}.get(st['tool'], 0.0)
            if st.get('stop_contact', False) and st['tool'] in ('ft_swell', 'ft_settle'):
                # Stop at contact: the contacts' distance at the plan's rows (gpu/collide_card.contact_rows reads the reach
                # bp[49] + the gap bp[50]); the rule clamps max(f, gap - tc) (ftools.cuh)
                bp[BR.NBP_SWC:BR.NBP_SWC + 5] = (1.0, 0.0, 0.0, r + abs(A), ftv[57])
        drag_ok = st['tool'] not in ('grab', 'stretch', 'grabflow', 'stretchflow') or st.get('keep_apart_drag', False) or bool(os.environ.get('FSB_GRAB_GUARD'))
        bp[BR.NBP_TOPO] = 1.0 if (st.get('keep_apart', False) and drag_ok and st['tool'] != 'mask' and not os.environ.get('FS_DIAG_NO_TOPO')) else 0.0
        return {'oi': oi, 'cells': cells, 'admit': admit, 'bp': bp, 'vol': vol, 'bm': bm, 'dcells': scope_dc,
                'steps': st.get('snake_steps_local') if st['tool'] == 'snakepath' else None,
                'mg': scope_mg if st['tool'] == 'grabflow' else None}

    def _guard_batch(self, ru, ku, d_ru, S16, mv, vol, bm):
        """fs-brush 11-11: the keep-apart guard on the batch's net change: the rows' current values become the targets, the
        batch-start values are restored, then dab_commit_split commits the nodes keeping their sign and the colour
        sweeps admit the flips that keep the inside and the outside as they were (as _run did per dab)"""
        cp = self.cp
        G = self.view.gpu
        n = len(ru)
        th = 256
        nt = (n * 729 + th - 1) // th
        cur = self._scr('g_cur', n * 729, np.float16)
        self.k_gather((nt,), (th,), (G.sdf, d_ru, np.int32(n), cur))
        out = self._scr('g_out', n * 729, np.float32)
        out[...] = cur
        self.k_restore((nt,), (th,), (G.sdf, d_ru, np.int32(n), S16))
        idx = self._scr('g_idx', n * 729, np.int64)
        cnt = cp.zeros(1, cp.int32)
        self.k_split((nt,), (th,), (G.sdf, d_ru, np.int32(n), out, idx, cnt))
        nflip = int(cnt.get()[0])
        pf = self.prof if PROF else None
        if pf is not None:
            pf['flips'] = pf.get('flips', 0) + nflip
        if not nflip:
            return
        flip = idx[:nflip]
        vidx = G.vols.index(vol)
        d_org = cp.asarray((vol.origin + ku * bm).astype(np.float32).reshape(-1))
        done = cp.zeros(nflip, cp.int32)
        acc = cp.zeros(1, cp.int32)
        tb = 128
        n_sw = int(min(64, max(TOPO_SWEEPS, math.ceil(mv / max(float(vol.levels[LEVEL]['h']), 1e-9)) + 4)))
        for sweep in range(n_sw):
            acc[...] = 0
            for col in range(8):
                self.k_topo_list(((nflip + tb - 1) // tb,), (tb,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf,
                                 np.int32(vidx), np.int32(LEVEL), d_ru, d_org, flip, np.int32(nflip), out,
                                 np.int32(col), done, acc))
            if pf is not None:
                pf['sweeps'] = pf.get('sweeps', 0) + 1
            if int(acc.get()[0]) == 0:
                break

    def _mdummy(self):
        if getattr(self, '_md', None) is None:
            self._md = self.cp.zeros(729, self.cp.float16)
            self._mds = self.cp.full(1, -1, self.cp.int32)
        return self._md

    def _sargs(self, p):
        """fs-brush 11-12: the composed Snake's steps for dab_compute (typ 17); a dummy for every other dab"""
        cp = self.cp
        st_ = p.get('steps')
        if st_ is None or not len(st_):
            if getattr(self, '_stdummy', None) is None:
                self._stdummy = cp.zeros(12, cp.float32)
            sa = (self._stdummy, np.int32(0))
        else:
            sa = (cp.asarray(np.ascontiguousarray(st_, np.float32).reshape(-1)), np.int32(len(st_)))
        mg = p.get('mg')                          # fs-brush 11-12: Grab's scope grid (uint8, x-major)
        if mg is None:
            if getattr(self, '_mgdummy', None) is None:
                self._mgdummy = cp.full(8, 255, cp.uint8)
            return sa + (self._mgdummy, np.int32(2), np.int32(2), np.int32(2))
        key = id(mg)
        if getattr(self, '_mgkey', None) != key:
            self._mgkey, self._mgdev = key, cp.asarray(np.ascontiguousarray(mg, np.uint8).reshape(-1))
        return sa + (self._mgdev, np.int32(mg.shape[0]), np.int32(mg.shape[1]), np.int32(mg.shape[2]))

    def _targs(self, p, n):
        """sculpt-contact: (contact rows, row -> slot) for dab_compute; none (every slot -1) outside Swell to contact.
        fs-brush 11-12: the 'none' buffers are kept and grown, not allocated every dab"""
        t = p.get('T')
        if t is not None:
            return t
        n = max(int(n), 1)
        tz = getattr(self, '_tzero', None)
        if tz is None:
            tz = self._tzero = self.cp.zeros(729, self.cp.float32)
        ts = getattr(self, '_tnone', None)
        if ts is None or ts.size < n:
            ts = self._tnone = self.cp.full(max(n, 2 * (0 if ts is None else ts.size), 4096), -1, self.cp.int32)
        return tz, ts[:n]

    def _margs(self, um):
        """(mask buffer, row -> slot map, flag) for the kernels"""
        if um and getattr(self, 'mpool', None) is not None and getattr(self, 'mslot', None) is not None:
            return self.mpool, self.mslot, np.int32(1)
        self._mdummy()
        return self._md, self._mds, np.int32(0)

    def _scr(self, name, n, dtype):
        """fs-brush 11-11 (fs-trust R9d: ~1 GB of transient card buffers per stroke, freed at its end and taken again by
        the next; the batch-sized buffers came in a new size every batch, so the pool kept a block of each): one
        persistent buffer per role, grown by 1.5x, sliced to n. A role's buffer is used up within one batch (one stream,
        so the next batch's kernels run after this one's)."""
        dt = np.dtype(dtype)
        if not SCRATCH:                          # FSB_SCRATCH=0: a new buffer each time (as before the arena)
            return self.cp.empty(n, dt)
        d = self.__dict__.setdefault('_scratch', {})
        b = d.get(name)
        if b is None or b.size < n or b.dtype != dt:
            d[name] = None
            b = d[name] = self.cp.empty(max(int(n * 1.5), 4096), dt)
        return b[:n]

    def _run(self, p, d_lut, nlut):
        cp = self.cp
        rows, keys = p['rows'], p['keys']
        if len(rows) == 0:
            return 0
        G = self.view.gpu
        vidx = G.vols.index(p['vol'])
        d_rows = cp.asarray(rows.astype(np.int32))
        d_org = cp.asarray((p['vol'].origin + keys * p['bm']).astype(np.float32).reshape(-1))
        out = self._scr('dab_out', len(rows) * 729, np.float32)
        th = 256
        nt = (len(rows) * 729 + th - 1) // th
        d_bp = cp.asarray(p['bp'])
        if p['bp'][BR.NBP_PLANE + 6] > 0.5:
            self.k_area((1,), (256,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL), d_bp,
                                       d_lut, np.int32(nlut)))
        sm_ = p.get('smask')
        if sm_ is not None:
            d_sm, u_sm = (sm_ if isinstance(sm_, cp.ndarray) else cp.asarray(sm_)), np.int32(1)
            d_si = p.get('smidx')
            if d_si is None:
                d_si = cp.arange(len(rows), dtype=cp.int32)
        else:
            d_sm = getattr(self, '_smdummy', None)
            if d_sm is None:
                d_sm = self._smdummy = cp.zeros(1, cp.uint8)
                self._sidummy = cp.zeros(1, cp.int32)
            d_si = self._sidummy
            u_sm = np.int32(0)
        ft_ = int(p['bp'][0]) in self.ft_typs
        k_ = self.k_ft if ft_ else self.k_dab     # fs-tools 11-14: dab_compute's arguments (+ the press-time pool, patch 3)
        k_((nt,), (th,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf, np.int32(vidx), np.int32(LEVEL), d_rows,
                                  d_org, np.int32(len(rows)), d_bp, d_lut, np.int32(nlut), out) + self._margs(bool(p.get('mask')))
                   + (d_sm, d_si, u_sm)
                   + self._targs(p, len(rows))
                   + self._sargs(p)
                   + (self._ft_start_args(p['oi']) if ft_ else ()))
        pf = self.prof if PROF else None
        if pf is not None:
            cp.cuda.Stream.null.synchronize(); _tp = time.perf_counter()
            pf['dab_ms'] = pf.get('dab_ms', 0.0) + (_tp - pf['_t']) * 1000; pf['_t'] = _tp
        flip = None
        if p['bp'][BR.NBP_TOPO] > 0.5:
            # 11-10 (acceptance S7: Grab 21.9 -> 7-8 fps with sweeps over every node): the nodes keeping their sign are
            # committed at once; the colour sweeps run over the flipping ones only
            idx = self._scr('dab_idx', len(rows) * 729, np.int64)
            cnt = cp.zeros(1, cp.int32)
            self.k_split((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), out, idx, cnt))
            nflip = int(cnt.get()[0])
            flip = idx[:nflip] if nflip else None
            if flip is None:
                return len(rows)
        if flip is not None:
            n = int(flip.size)
            done = cp.zeros(n, cp.int32)
            acc = cp.zeros(1, cp.int32)
            tb = 128
            # fs-brush 11-11: enough sweeps for the dab's own move (each sweep advances a front by at least one node;
            # 16 sweeps refused the far nodes of a 20-cell Grab move, leaving specks)
            mv = float(np.linalg.norm(p['bp'][9:12])) if int(p['bp'][0]) in (BR.T['grab'], BR.T['stretch'], BR.T['grabflow'], BR.T['stretchflow'], BR.T['snakeflow']) else abs(float(p['bp'][13]))
            if int(p['bp'][0]) == BR.T['snakepath'] and p.get('steps') is not None and len(p['steps']):
                # fs-brush-4 11-13: the composed Snake moves a node up to the whole path's length (core/brush.py keep_apart_sweeps)
                mv = float(np.linalg.norm(np.asarray(p['steps'], float)[:, 3:6], axis=1).sum())
            n_sw = int(min(64, max(TOPO_SWEEPS, math.ceil(mv / max(float(p['vol'].levels[LEVEL]['h']), 1e-9)) + 4)))
            if pf is not None:
                _tp = time.perf_counter(); pf['split_ms'] = pf.get('split_ms', 0.0) + (_tp - pf['_t']) * 1000; pf['_t'] = _tp
                pf['flips'] = pf.get('flips', 0) + n
            for sweep in range(n_sw):
                acc[...] = 0
                for col in range(8):
                    self.k_topo_list(((n + tb - 1) // tb,), (tb,), (G.d_vft, G.d_vit, G.d_grid, G.d_ttab, G.sdf,
                                     np.int32(vidx), np.int32(LEVEL), d_rows, d_org, flip, np.int32(n), out,
                                     np.int32(col), done, acc))
                if pf is not None:
                    pf['sweeps'] = pf.get('sweeps', 0) + 1
                if int(acc.get()[0]) == 0:
                    break
            if pf is not None:
                _tp = time.perf_counter(); pf['sweep_ms'] = pf.get('sweep_ms', 0.0) + (_tp - pf['_t']) * 1000; pf['_t'] = _tp
        else:
            self.k_commit((nt,), (th,), (G.sdf, d_rows, np.int32(len(rows)), out))
            if pf is not None:
                cp.cuda.Stream.null.synchronize(); _tp = time.perf_counter()
                pf['commit_ms'] = pf.get('commit_ms', 0.0) + (_tp - pf['_t']) * 1000; pf['_t'] = _tp
        return len(rows)

    def _admit(self, scene, oi, ov, cells):
        """New GPU bricks for the whole batch, before any of its dabs: values from the CPU field (current at the batch
        start) -> rows allocated in the atlas."""
        mirror = self.view.mirror
        o = scene.objects[oi]
        h = float(o.sculpt.h)
        bm = 8 * h
        vol = ov.vol
        rows = mirror._alloc_bricks(ov, cells)
        self.view.gpu._reserve(0)
        mirror.geval.eval(scene, o, vol.origin + cells * bm, cells - ov.boff, rows=rows, sdf=self.view.gpu.sdf)   # r22
        return
        band = vol.levels[LEVEL]['band']
        # r14: true values, no clamp (see mirror._admit_cells)
        rows = mirror._alloc_bricks(ov, cells)
        self.view.gpu.write_bricks(rows, vals.astype(np.float16))
        self.view.gpu._tables()

    def _write_back(self, scene, oi, rows, keys_in, Fb, Fa, rec):
        """delta += (F_after - F_before) at the bricks' 8^3 sculpt nodes, into the CPU sculpt layer (snapshots / new
        bricks into rec, as brush.apply_sample does); mark them current in the mirror."""
        o = scene.objects[oi]
        g = o.sculpt
        stt = scene.store
        h = float(g.h)
        ov = self.view.mirror.objs[o.id]
        _tw = time.perf_counter(); wt = self.stats.setdefault('wb', {})
        if isinstance(keys_in, dict):
            keys = np.array([keys_in[r] for r in rows.tolist()], np.int64).reshape(-1, 3)
        else:
            keys = np.asarray(keys_in, np.int64).reshape(-1, 3)
        Fa_all = Fa
        dF = (Fa - Fb).reshape(len(rows), 9, 9, 9)[:, :8, :8, :8]
        bad = ~np.isfinite(dF)
        if bad.any():                                    # 11-09 guard: a non-finite card value never reaches the CPU
            self.stats['wb_nonfinite'] = self.stats.get('wb_nonfinite', 0) + int(bad.sum())
            dF = np.where(bad, 0.0, dF)
        ch = np.any(dF != 0, axis=(1, 2, 3))
        if WB_BAND_H > 0 and ch.any():
            # fs-brush 11-11 (band-only allocation, step 1; fs-trust measured every stroke allocating its whole ball of
            # sculpt bricks, the interior included, never freed: 5,640 B a brick): a brick written back only where the
            # surface is near before or after the stroke (within WB_BAND_H h, beyond the mirror's admission reach) or a
            # node changed sign. A deep brick that changed no sign keeps its old CPU values: the same sign, far from
            # any surface (the card keeps the stroke's values there until it resamples the brick).
            fb8 = Fb.reshape(len(rows), 9, 9, 9)[:, :8, :8, :8]
            fa8 = Fa.reshape(len(rows), 9, 9, 9)[:, :8, :8, :8]
            band = WB_BAND_H * h
            near = (np.abs(fb8).min(axis=(1, 2, 3)) <= band) | (np.abs(fa8).min(axis=(1, 2, 3)) <= band)
            flip = np.any((fb8 < 0) != (fa8 < 0), axis=(1, 2, 3))
            keep_ = near | flip
            self.stats.setdefault('wb', {})['deep_skipped'] = int((ch & ~keep_).sum())
            ch = ch & keep_
        if not ch.any():
            return None
        keys, dF = keys[ch], dF[ch]
        sk = keys - ov.boff                              # sculpt brick keys (GPU brick g = sculpt brick g - boff)
        if g.ensure_index(sk.min(0), sk.max(0)):
            scene.touch(index=True)
        # r24: slots looked up through the sculpt index at once; only missing bricks and first snapshots loop
        slots = np.full(len(sk), -1, np.int64)
        if g.index.size:
            rel = sk - np.asarray(g.origin, np.int64)
            inb = np.all((rel >= 0) & (rel < np.array(g.index.shape)), 1)
            slots[inb] = g.index[rel[inb, 0], rel[inb, 1], rel[inb, 2]]
        sk_t = list(map(tuple, sk.tolist()))
        miss = np.nonzero(slots < 0)[0]
        if len(miss):                                   # fs-brush 11-11: allocated at once (store.alloc_many, set_slots)
            ns = stt.alloc_many(len(miss))
            g.set_slots(sk[miss], ns)
            slots[miss] = ns
            nk_ = [(o.id, sk_t[i]) for i in miss.tolist()]
            rec.new.extend(nk_)
            rec.new_set.update(nk_)
        oid_ = o.id
        snap_needed = [i for i, b in enumerate(sk_t) if (oid_, b) not in rec.snap and (oid_, b) not in rec.new_set]
        if snap_needed:
            sn = slots[snap_needed]
            dsn = stt.delta[sn].copy()
            msn = stt.mask[sn].copy()
            lsn = stt.Lb[sn].copy()
            for j, i in enumerate(snap_needed):
                rec.snap[(o.id, sk_t[i])] = (dsn[j], msn[j], float(lsn[j]))
        wt['alloc_snap_ms'] = (time.perf_counter() - _tw) * 1000; _tw = time.perf_counter()
        stt.delta[slots] = (stt.delta[slots] + dF).astype(np.float32)
        # r42: what replay will apply, merged by key over the stroke's write-backs (a grid-overflow retry writes back
        # mid-stroke), the later value winning
        nk, na = sk.copy(), stt.delta[slots].copy()
        prev = self._result.get(o.id)
        if prev is not None and len(prev[0]):
            def _code(k):                                # fs-brush 11-11: keys compared as one int64 each
                k = np.asarray(k, np.int64) + (1 << 20)
                return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
            keep = ~np.isin(_code(prev[0]), _code(nk))
            nk, na = np.concatenate([prev[0][keep], nk]), np.concatenate([prev[1][keep], na])
        self._result[o.id] = (nk, na)
        wt['result_ms'] = (time.perf_counter() - _tw) * 1000; _tw = time.perf_counter()
        scene.touch(index=True)
        pk = scene.pack()
        wt['pack_ms'] = (time.perf_counter() - _tw) * 1000; _tw = time.perf_counter()
        if os.environ.get('FS_GPU_SCULPT_CHECK'):        # debug: the CPU field at the written nodes vs the card
            chk = np.empty((len(keys), 729))
            _eval_nodes9(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, oi,
                         np.ascontiguousarray(ov.vol.origin + keys * 8 * h, np.float64), h, chk)
            fa = Fa_all[ch].reshape(-1, 9, 9, 9)[:, :8, :8, :8].reshape(len(keys), -1)
            cc = chk.reshape(-1, 9, 9, 9)[:, :8, :8, :8].reshape(len(keys), -1)
            band = ov.vol.levels[LEVEL]['band']
            m = np.abs(fa) < band
            self._wbc = float(np.abs(fa - cc)[m].max() * 1e6) if m.any() else 0.0
        if TSDF and self.live_redist:
            # fs-core 11-14: L over the band's cells only (core/brush.py brick_lipschitz_band), from the card's own values
            BR.measure_bricks_band(scene, oi, sk_t, np.asarray(Fa_all, np.float32).reshape(-1, 9, 9, 9)[ch],
                                   TSDF_W_H * h)
        else:
            BR.measure_bricks(scene, oi, set(sk_t))
        wt['measure_ms'] = (time.perf_counter() - _tw) * 1000; _tw = time.perf_counter()
        self.view.mirror.mark_current(scene, o, sk_t)   # current on the card already
        wt['mark_ms'] = (time.perf_counter() - _tw) * 1000
        bm = 8 * h
        lo = ov.vol.origin + keys.min(0) * bm
        hi = ov.vol.origin + (keys.max(0) + 1) * bm
        return self.view.mirror._world_box(pk.of_tab[oi], (lo, hi))
