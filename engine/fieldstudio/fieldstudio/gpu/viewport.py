"""The app's GPU frame source (phase 0, 2026-11-08): one picture for editing and final.

GpuView.frame() runs on the engine thread with the engine lock held, in place of render.raymarch.render(). It syncs
the scene mirror (only what changed is sent), then:
  - camera, size or a whole object changed: trace the whole frame (pass 0) and light it;
  - only sculpt bricks changed: re-trace the dirty rectangles (the projected changed boxes, grown by the sky reach
    and swept along the sun direction to the ground, so every pixel whose light the edit can change is redone) and
    restart their light; the rest of the frame keeps its converged light;
  - nothing changed: add light passes to the unconverged regions (progressive) until TARGET_PASSES.
It returns the arrays ui/viewport.py already draws (rgb, obj ids, depth) and stats.
"""
import os
import time

import numpy as np

from .launch_guard import GuardStop

TARGET_PASSES = 16
SKY_REACH = 0.15
PENUMBRA = 0.02      # a changed box is grown by this for the soft-shadow edge (checked: bench B5 rect vs fresh)


class Frame:
    __slots__ = ('rgb', 'obj', 'depth', 'stats', 'sel_alpha', 'sel_ids', 'sel_hidden', 'hover_mask')


def _cam_key(cam, w, h):
    # fs-shell 11-11: the orthographic flag is part of the view (toggling it with the camera still re-renders)
    # fs-shell 11-11 (design 4.4): the section plane too (moving or flipping it re-renders)
    sec = tuple(np.round(cam.section_vec(), 9)) if getattr(cam, 'section', None) is not None else None
    # fs-shell 11-11 (design 4.8): the hidden boxes too (box hide / Alt+H re-render)
    cl = getattr(cam, 'clip_planes', None)
    clk = tuple(np.round(np.asarray(cl, float).ravel(), 9)) if cl is not None and len(cl) else None
    return (tuple(np.round(cam.pos, 9)), tuple(np.round(cam.target, 9)), tuple(np.round(cam.up, 9)), round(cam.fov, 6),
            w, h, bool(getattr(cam, 'ortho', False)), sec) + ((clk,) if clk is not None else ())


def _with_clips(cam, scene):
    """fs-shell 11-11: the frame's camera carries the scene's hidden boxes (render/raymarch.view_clips' planes)"""
    vb = getattr(scene, 'view_boxes', None)
    if not vb:
        return cam
    from ..render.raymarch import clip_planes
    c = cam.copy()
    c.clip_planes = [clip_planes(v) for v in vb]
    return c


class GpuView:
    def __init__(self, light=None):
        from .scene import GpuScene
        from .mirror import SceneMirror
        self.gpu = GpuScene(cap_bricks=1 << 17)
        self.mirror = SceneMirror(self.gpu)
        self.key = None
        self.pending = None            # the bounding rect of pixels still converging (None: converged)
        self.changed = True
        self.light = light or {}
        # r26: 'key' = a sculpting key light that follows the view (40 deg above, from the upper left, the HDRI's sun
        # irradiance; the HDRI sky stays the fill); 'hdri' = the environment's own sun (85 deg up on noon_grass,
        # which leaves the front of a standing figure in its own shadow)
        self.light_mode = os.environ.get('FIELDSTUDIO_LIGHT', 'key')
        # sculpting visibility aid (11-08 night: deep recesses were near-black): a camera headlight, the sky
        # occlusion's strength, exposure in stops. Defaults = the review light; the app turns the headlight on.
        self.aid = {'headlight': 0.0, 'cavity': 1.0, 'ev': 0.0}
        self.last = {}
        self.ground_y = None
        self.light_rt = None           # fs-light 11-11: the look-dev light runtime (engine/light/light_runtime.py), set by
                                       # the Light panel when a look-dev rig is picked; None = this view as before
        try:                           # fs-materials 11-14: library materials' look (materials/look.py: the per-volume table)
            from ..materials.look import Materials
            self.materials = Materials()
        except Exception:
            self.materials = None

    def note_change(self):
        self.changed = True

    def frame(self, scene, cam, w, h, budget_ms=40.0, interactive=False, selected=None):
        """r9: per-pixel progressive state on the card (each pixel's own pass count). A change restarts only the pixels
        it can change; refine passes skip converged pixels; `pending` is the bounding rect still converging."""
        t0 = time.perf_counter()
        G = self.gpu
        sync = None
        tm = {}
        cam = _with_clips(cam, scene)            # fs-shell 11-11: box hide (design 4.8)
        if self.changed:
            # r18: a GPU stroke changes nothing on the CPU until its release; skip the sync when the scene, its store
            # and its objects are as at the last sync (every CPU edit path bumps scene.version through touch /
            # refresh_object)
            key = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
            if getattr(self, 'hold_sync', False):
                # fs-brush-2 11-13: a stroke's write is behind its release (gpu/sculpt.py PW): the card is ahead of the
                # CPU until its last slice, which syncs what it wrote; a sync from the CPU now could hand back old values
                pass
            elif key != getattr(self, '_sync_key', None):
                self._sync(scene)
                self._sync_key = key
            else:
                self.changed = False
        if not interactive and self.mirror.coarse_pending:
            # r37: queued coarse-level refreshes (after GPU strokes) on an idle frame; their boxes re-light
            self.note_dirty(self.mirror.coarse_flush(scene))
        sync, self._held = self._held, None      # what syncs since the last frame changed (also picks' syncs)
        tm['sync'] = (time.perf_counter() - t0) * 1000
        if self.light_mode == 'key':
            f_, r_, u_ = cam.basis()
            d_ = -0.62 * f_ + 0.64 * u_ - 0.45 * r_
            self.light = dict(self.light, sun_dir=d_ / np.linalg.norm(d_))
        else:
            self.light = {k: v for k, v in self.light.items() if k != 'sun_dir'}
        self.light = dict(self.light, **self.aid)
        if self.light_rt is not None and self.light_rt.before_frame(G, cam, self.light, interactive,
                                                                    idle=self.pending is None):
            self.key = None              # fs-light: the look-dev caches swapped in (or out): one full restart
        if self.materials is not None and self.materials.before_frame(G, self.mirror, scene, cam, w, h):
            self.key = None              # fs-materials: a material changed: one full restart
        key = _cam_key(cam, w, h) + (self.light_mode, tuple(sorted(self.aid.items())))
        full = key != self.key or (sync is not None and (sync['rebuilt'] or self._rows_changed(sync)))
        mode = 'refine'
        hot = None                       # the touched patch: during a live stroke the frame's pass goes there first
        if full:
            mode = 'full'
            G.restart(cam, w, h, light=self.light)
            self.pending = (0, 0, w, h)
            self.key = key
        elif sync is not None and sync['dirty_world']:
            mode = 'rects'
            _pf = bool(os.environ.get('FSB_PROF'))
            # fs-store-5 11-13 (lane copy, FS_VP_RECTFAST, 0 = as before): a stroke frame spent 5-12 ms of CPU in _rect
            # (fs-brush-3's pf_tk2 / pf_tk4: rect_ms), which recomputed the light parameters and the scene's bounds (a full
            # pack when the index changed) on every call. Once per frame here: the light direction, and the floor from the
            # pack's bounds when the pack is current (else the last frame's floor); the projection vectorised.
            fast = os.environ.get('FS_VP_RECTFAST', '1') != '0'
            pre = None
            if fast:
                _tl = time.perf_counter()
                L0 = self.gpu.light_params(cam, **self.light)[0:3].astype(float)
                _tb = time.perf_counter()
                pk_ok = (scene._pack is not None and scene._store_gen == scene.store.generation
                         and not scene._index_dirty)
                if pk_ok or getattr(self, '_floor_y', None) is None:
                    self._floor_y = float(scene.bounds()[0][1])
                pre = (L0, self._floor_y)
                if _pf:
                    tm['rect_light_ms'] = (_tb - _tl) * 1000
                    tm['rect_bounds_ms'] = (time.perf_counter() - _tb) * 1000
            for lo, hi in sync['dirty_world']:
                _ta = time.perf_counter()
                own = self._rect(cam, w, h, lo, hi, scene, sweep=False, pre=pre)
                sw = self._rect(cam, w, h, lo, hi, scene, sweep=True, pre=pre)
                if _pf:
                    tm['rect_ms'] = tm.get('rect_ms', 0.0) + (time.perf_counter() - _ta) * 1000
                    tm['n_boxes'] = tm.get('n_boxes', 0) + 1
                if own is None and sw is None:
                    continue
                bx = (np.asarray(lo, float) - PENUMBRA, np.asarray(hi, float) + PENUMBRA)
                if _pf:
                    _ta = time.perf_counter(); G._tables(); G.cp.cuda.Stream.null.synchronize()
                    tm['tables_ms'] = tm.get('tables_ms', 0.0) + (time.perf_counter() - _ta) * 1000
                    _ta = time.perf_counter(); G.frame_buffers(w, h); Lp_ = G.light_params(cam, **(self.light or {}))
                    G.cp.cuda.Stream.null.synchronize()
                    tm['fb_light_ms'] = tm.get('fb_light_ms', 0.0) + (time.perf_counter() - _ta) * 1000
                _ta = time.perf_counter()
                G.restart(cam, w, h, rect=sw or own, boxes=[bx], trace_rect=own or (0, 0, 0, 0), light=self.light)
                if _pf:
                    G.cp.cuda.Stream.null.synchronize()
                    tm['restart_call_ms'] = tm.get('restart_call_ms', 0.0) + (time.perf_counter() - _ta) * 1000
                self.pending = self._union(self.pending, sw or own)
                if own:
                    hot = self._union(hot, own)
        G.cp.cuda.Stream.null.synchronize()
        tm['restart'] = (time.perf_counter() - t0) * 1000 - tm['sync']
        t1 = time.perf_counter()
        left = G.TARGET
        guard_stop = None
        if self.pending:
            # one pass at once (the changed pixels' first light); more while the budget lasts, unless a stroke is live
            # r35: an interactive frame's first pass casts one sky ray (the per-pixel ray count is kept, so the
            # converged image averages every ray it got); refine passes cast two
            lit1 = dict(self.light, sky_n=1) if interactive else self.light
            try:
                if interactive and hot is not None:
                    G.progress(cam, w, h, rect=hot, passes=1, light=lit1)
                    left = 0                 # the sweep (shadows below the edit) settles once the stroke pauses
                else:
                    left = G.progress(cam, w, h, rect=self.pending, passes=1, light=lit1)
                while left < G.TARGET and not interactive and (time.perf_counter() - t0) * 1000 < budget_ms:
                    # r31: as many passes as the rest of the budget allows between progress checks (each check syncs)
                    ts = time.perf_counter()
                    k = max(1, min(G.TARGET - left, int((budget_ms - (ts - t0) * 1000) / max(self._pass_ms, 0.5))))
                    left = G.progress(cam, w, h, rect=self.pending, passes=k, light=self.light)
                    self._pass_ms = 0.7 * self._pass_ms + 0.3 * ((time.perf_counter() - ts) * 1000 / k)
            except GuardStop as e_:
                # fs-shell 11-13: the launch guard stopped this frame (a launch past 1 s, or the app is closing): nothing
                # more is launched now; the pending pixels stay pending and the next frame is planned from what this
                # one measured (never relaunched larger, never a whole-frame fallback)
                guard_stop = str(e_)
                left = 0
            if left >= G.TARGET:
                self.pending = None
        tm['render'] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        # r11: RGB and object ids made on the card; only what the viewport draws crosses the bus
        cp = G.cp
        f = Frame()
        _pf = bool(os.environ.get('FSB_PROF'))       # fs-brush 11-11: the host side split (synchronising; benches only)
        if _pf:
            cp.cuda.Stream.null.synchronize(); _t = time.perf_counter(); tm['gpu_wait'] = (_t - t2) * 1000
        out = G.d_out
        xr = float(getattr(self, 'xray', 0.0) or 0.0)
        if (xr > 0.0 or getattr(self, 'xray_any', False)) and guard_stop is None:
            # fs-shell 11-13 (Tier 0 part 4): the X-ray composite over this frame (gpu/viewmodes.py xray_comp)
            _tx = time.perf_counter()
            try:
                vm = getattr(self, 'viewmodes', None)
                if vm is None or vm.G is not G:
                    from .viewmodes import ViewModes
                    vm = self.viewmodes = ViewModes(G)
                out = vm.xray(G, cam, w, h, min(max(xr, 0.0), 0.95))
            except GuardStop as e_:
                guard_stop = str(e_)
            tm['xray'] = (time.perf_counter() - _tx) * 1000
        f.rgb = cp.ascontiguousarray(out.view(cp.uint8).reshape(h, w, 4)[:, :, 2::-1]).get()
        if _pf:
            _t2 = time.perf_counter(); tm['rgb'] = (_t2 - _t) * 1000; _t = _t2
        ids = cp.asarray(np.array([getattr(v, 'oid', -1) for v in G.vols] + [-1], np.int16))
        f.obj = ids[G.d_obj].get()                   # r35b: mapped on the card, int16 across the bus
        if _pf:
            _t2 = time.perf_counter(); tm['obj'] = (_t2 - _t) * 1000; _t = _t2
        f.depth = G.d_depth.get()                    # (converting depth on the host cost more than the copy saved)
        if _pf:
            _t2 = time.perf_counter(); tm['depth'] = (_t2 - _t) * 1000; _t = _t2
        f.sel_alpha, f.sel_ids = None, None
        if selected:
            # r39: the selection outline's alpha on the card (ui/viewport blends it after the reference planes)
            if not hasattr(self, '_k_sel'):
                self._k_sel = G.mod.get_function('sel_alpha')
            obj16 = ids[G.d_obj]
            a = cp.empty((h, w), cp.uint8)
            sid = cp.asarray(np.array(list(selected), np.int16))
            self._k_sel(((w + 15) // 16, (h + 7) // 8, 1), (16, 8, 1), (obj16, np.int32(w), np.int32(h), sid,
                                                                       np.int32(len(selected)), a))
            f.sel_alpha, f.sel_ids = a.get(), list(selected)
            if _pf:
                tm['sel'] = (time.perf_counter() - _t) * 1000
        # fs-shell-2 11-14: select inside part 3 (the selected objects' outline where something covers them) and part 4
        # (the Scene list's hovered object, outline_hover set by the window): the ID image of those volumes on their
        # own, gpu/viewmodes.vol_mask (the patent guards are written there). Kept while camera, scene and ids hold.
        f.sel_hidden, f.hover_mask = None, None
        try:
            sel_v = tuple(i for i, v in enumerate(G.vols) if selected and getattr(v, 'oid', -1) in set(selected))[:4]
            hov = getattr(self, 'outline_hover', None)
            hov_v = tuple(i for i, v in enumerate(G.vols) if hov is not None and getattr(v, 'oid', -1) == hov)[:1]
            if sel_v or hov_v:
                mkey = (_cam_key(cam, w, h), getattr(self, '_sync_key', None), sel_v, hov_v, len(G.vols))
                memo = getattr(self, '_volmask_memo', None)
                if memo is not None and memo[0] == mkey:
                    f.sel_hidden, f.hover_mask = memo[1], memo[2]
                else:
                    from .viewmodes import ViewModes, vol_mask
                    vm = getattr(self, 'viewmodes', None)
                    if vm is None or vm.G is not G:
                        vm = self.viewmodes = ViewModes(G)
                    _tvm = time.perf_counter()
                    res = []
                    for vs in (sel_v, hov_v):
                        acc = None
                        for vi in vs:
                            m_ = vol_mask(vm, G, cam, w, h, vi)
                            acc = m_ if acc is None else (acc | m_)
                        res.append(acc.get() if acc is not None else None)
                    f.sel_hidden, f.hover_mask = res
                    self._volmask_memo = (mkey, res[0], res[1])
                    tm['volmask'] = (time.perf_counter() - _tvm) * 1000
        except GuardStop as e_:
            guard_stop = str(e_)
        tm['host'] = (time.perf_counter() - t2) * 1000
        ms = (time.perf_counter() - t0) * 1000
        f.stats = {'ms': ms, 'mode': mode, 'pending': self.pending, 'passes_left_min': int(left), 'guard_stop': guard_stop,
                   'guard': dict(G.guard.stats) if hasattr(G, 'guard') else None,
                   'converged': self.converged(), 'times_ms': {k: round(v, 1) for k, v in tm.items()},
                   'sync': ({k: v for k, v in sync.items() if k != 'dirty_world'} if sync else None),
                   'internal_res': (w, h), 'unresolved': 0, 'steps': 0, 'gpu_mb': round(G.gpu_bytes() / 2 ** 20, 1),
                   'bricks': int(G.used)}
        self.last = f.stats
        return f

    def converged(self):
        # fs-light: not while the look-dev light runtime has slices left (the engine keeps drawing idle frames)
        return self.pending is None and not (self.light_rt is not None and self.light_rt.wants_frames)

    _held = None
    _pass_ms = 5.0

    def _sync(self, scene, known_current=False):
        """Mirror the scene now; what it changed is held for the next frame (a pick may sync first).
        known_current: a card stroke's release (gpu/sculpt.py), mirror.sync's light path"""
        s = self.mirror.sync(scene, known_current=known_current)
        self.changed = False
        self._sync_key = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        if self._held is None:
            self._held = s
        else:
            h = self._held
            h['rebuilt'] = h['rebuilt'] + s['rebuilt']
            h['bricks'] += s['bricks']
            h['dirty_world'] = h['dirty_world'] + s['dirty_world']
            h['ms'] += s['ms']
        return s

    def set_reference_store(self, path, on=True, offset=(0.6, 0.0, 0.0), copies=1):
        """r33: a published FV8 store drawn beside the scene, view only (not in the scene: never saved, hashed or
        picked). FV8 local is z up with the feet at z = -1.424 (her L3 bricks, read 11-08); placed feet on y = 0.
        copies > 1 places more copies 0.9 m apart (the phase 0 benchmark's 2x load, in the app)."""
        refs = getattr(self, '_refs', [])
        for v in refs:
            v.visible = bool(on)
        if on and not refs and path:
            R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
            for c in range(int(copies)):
                v = self.gpu.add_fv8_store(path, name='reference%d' % c)
                v.set_transform(R=R, t=np.array(offset, float) + np.array([0.9 * c, 1.424, 0.0]))
                v.oid = -1
                refs.append(v)
            self._refs = refs
        self.gpu._dirty_tables = True
        self.gpu.refresh_transforms()
        self.key = None
        return len(refs)

    def note_dirty(self, boxes):
        """GPU strokes changed these world boxes on the card directly (the mirror was told; no re-sampling)."""
        if not boxes:
            return
        if self.light_rt is not None:
            self.light_rt.note_dirty(boxes)      # fs-light: appended only; its rebuild waits for an idle frame
        if self._held is None:
            self._held = {'rebuilt': [], 'bricks': 0, 'dirty_world': [], 'ms': 0.0}
        self._held['dirty_world'] = self._held['dirty_world'] + list(boxes)

    def scrub(self, scene, budget_slots=None):
        """fs-brush 11-12: one idle slice of the mirror's check of the sculpt store (mirror.scrub_step), only while the
        scene is as last synced; a slot found changed without a touch forces a full sync now. -> the step's dict or None"""
        key = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        if key != getattr(self, '_sync_key', None):
            return None
        r = self.mirror.scrub_step(scene) if budget_slots is None else self.mirror.scrub_step(scene, budget_slots)
        if r and r.get('differ'):
            self._sync(scene)
            self.changed = True
        return r

    def sync_if_changed(self, scene):
        """fs-brush 11-11: the mirror sync only when the CPU scene changed (the key frame() checks). During a card stroke
        the engine notes a change every frame while nothing on the CPU changes until the release; picks and dab batches
        ran the full sync (a checksum of every sculpt brick, a dict over every slot) each time (M: head scene 5-10 fps
        whatever the brush size)."""
        if not self.changed or getattr(self, 'hold_sync', False):     # fs-brush-2 11-13: see frame()
            return
        key = (scene.version, scene.store.used, scene.store.generation, len(scene.objects), id(scene))
        if key != getattr(self, '_sync_key', None):
            self._sync(scene)

    def pick_many(self, scene, views, exclude=()):
        """r30: [(cam, x, y, w, h)] -> [(hit, point, normal, object id, t)] in one launch."""
        self.sync_if_changed(scene)
        rays = [cam.ray(x, y, w, h) for cam, x, y, w, h in views]
        org = np.array([r[0] for r in rays])
        dirs = np.array([r[1] for r in rays])
        t, v, n = self.gpu.pick(org, dirs, exclude)
        out = []
        for i in range(len(rays)):
            if t[i] < 0:
                out.append((False, None, None, -1, float('inf')))
            else:
                out.append((True, org[i] + dirs[i] * float(t[i]), n[i].astype(float),
                            getattr(self.gpu.vols[int(v[i])], 'oid', -1), float(t[i])))
        return out

    def pick(self, scene, cam, x, y, w, h, exclude=()):
        """r12: a pick on the card's field (the one the viewport draws): -> (hit, point, normal, object id, t)."""
        if (getattr(scene, 'view_boxes', None) or getattr(scene, 'view_regions_hidden', None)) and not exclude:
            # fs-shell 11-11 (design 4.8): with hidden boxes or regions the pick sees what the view shows (the CPU
            # pick's view-clip rule; one ray)
            from ..render.raymarch import pick as _cpu_pick
            r = _cpu_pick(scene, cam, x, y, w, h)
            return bool(r.hit), r.point, r.normal, r.object_id, r.t
        self.sync_if_changed(scene)
        org, d = cam.ray(x, y, w, h)
        t_add, t_far = 0.0, float('inf')
        if getattr(cam, 'section', None) is not None:
            # fs-shell 11-11 (design 4.4): a pick sees what the sectioned view shows (render/raymarch.pick's rule)
            sv = cam.section_vec()
            sn, sd = sv[:3], float(sv[3])
            s0, dn = float(sn @ org) - sd, float(sn @ d)
            if s0 > 0.0:
                if dn >= 0.0:
                    return False, None, None, -1, float('inf')
                tp = -s0 / dn
                q = org + d * tp
                from ..core.program import STACK, eval_scene, eval_scene_f
                pk = scene.pack()
                vs, ps = np.empty(STACK), np.empty(4 * STACK)
                _fq, _dq, oq = eval_scene(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta,
                                          q[0], q[1], q[2], vs, ps)
                fq = eval_scene_f(pk.code, pk.prm, pk.oi_tab, pk.of_tab, pk.index_flat, pk.delta, q[0], q[1], q[2],
                                  vs, ps, 1e9)                  # the exact reader (the one inside rule)
                if fq < 0.0 and oq >= 0:
                    return True, q, np.array(sn, float), pk.ids[int(oq)], float(tp)
                org, t_add = q, tp
            elif dn > 0.0:
                t_far = -s0 / dn
        t, v, n = self.gpu.pick(org[None], d[None], exclude)
        if t[0] < 0 or float(t[0]) + t_add > t_far:
            return False, None, None, -1, float('inf')
        oid = getattr(self.gpu.vols[int(v[0])], 'oid', -1)
        return True, org + d * float(t[0]), n[0].astype(float), oid, float(t[0]) + t_add

    @staticmethod
    def _union(a, b):
        if a is None:
            return b
        x0, y0 = min(a[0], b[0]), min(a[1], b[1])
        x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
        return (x0, y0, x1 - x0, y1 - y0)

    def _rows_changed(self, sync):
        # transforms, visibility and colour are refreshed every sync; compare them with what the frame used
        # r36: transforms, visibility and colour only (ftab[0:25], [43:47]); the tight box (ftab[37:43]) changes when
        # bricks are admitted and does not change the picture (it restarted the whole frame after every stroke)
        sig = tuple(tuple(np.round(np.r_[v.ftab()[0:25], v.ftab()[43:47]], 7)) for v in self.gpu.vols)
        ch = sig != getattr(self, '_rowsig', None)
        self._rowsig = sig
        return ch

    @staticmethod
    def _merge(pending, new, w, h):
        """Keep few regions: a new rect overlapping a pending one merges into their union (restarting it)."""
        out = list(pending)
        for r, n in new:
            x0, y0, x1, y1 = r[0], r[1], r[0] + r[2], r[1] + r[3]
            keep = []
            for (q, m) in out:
                if q[0] < x1 and x0 < q[0] + q[2] and q[1] < y1 and y0 < q[1] + q[3]:
                    x0, y0 = min(x0, q[0]), min(y0, q[1])
                    x1, y1 = max(x1, q[0] + q[2]), max(y1, q[1] + q[3])
                else:
                    keep.append((q, m))
            out = keep + [((x0, y0, x1 - x0, y1 - y0), n)]
        return out

    def _rect(self, cam, w, h, lo, hi, scene, sweep=True, pre=None):
        g = SKY_REACH + PENUMBRA if sweep else 0.0
        lo = np.asarray(lo, float) - g
        hi = np.asarray(hi, float) + g
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts = [c]
        L = pre[0] if pre is not None else self.gpu.light_params(cam, **self.light)[0:3].astype(float)
        if sweep and L[1] > 1e-3:                               # the sun-shadow sweep, down to the scene's floor
            fy = pre[1] if pre is not None else float(scene.bounds()[0][1])
            floor = min(fy, float(c[:, 1].min())) - 0.01
            tt = np.minimum(10.0, (c[:, 1] - floor) / L[1])
            pts.append(c - tt[:, None] * L[None])
        pts = np.concatenate(pts)
        f, r, u = cam.basis()
        z = (pts - cam.pos) @ f
        if np.all(z <= 1e-4):
            return None
        if np.any(z <= 1e-4):
            return (0, 0, w, h)
        if pre is not None and not getattr(cam, 'ortho', False):
            # fs-store-5 (FS_VP_RECTFAST): camera.project's own formulas, all points at once (one basis)
            v = pts - cam.pos
            th = np.tan(np.radians(cam.fov) * 0.5); a = w / float(h)
            xy = np.stack([((v @ r) / (z * th * a) + 1.0) * 0.5 * w - 0.5, (1.0 - (v @ u) / (z * th)) * 0.5 * h - 0.5], 1)
        else:
            xy = np.array([cam.project(p, w, h)[:2] for p in pts])
        x0, y0 = np.floor(xy.min(0)).astype(int) - 2
        x1, y1 = np.ceil(xy.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
