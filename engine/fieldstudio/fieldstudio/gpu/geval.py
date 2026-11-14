"""r22: evaluate an object's true field at GPU brick nodes on the card (gpu/program.cu): base program in double precision
+ the sculpt layer's node values. Used for volume builds and brick admissions (the CPU evaluation was the cost of
merge-all (17 s) and of a stroke entering fresh ground (dab planning p90 spikes)).
"""
import os
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 11-09 night (two launches lost the card on the first strokes; phase0/card_fault_repro.py). Two faults in how
# program.cu's FD global (the address of her float32 store buffer) was written, both measured:
#  1. it was copied from np.array([ptr]).ctypes.data: that array is freed before the copy reads it, so FD took whatever
#     the host heap held by then (0x1 seen) whenever another thread allocated in between -> cudaErrorIllegalAddress in
#     eval_nodes8. Ablation with everything else equal: the dangling copy 2/2 fail, the same copy from a held array
#     3/3 pass, the fill below 8/8 pass.
#  2. CuPy loads ONE module for the same source and options, so every GpuEval (the viewport's in gpu/mirror.py, the
#     release re-distance's in gpu/redist.py) shares the one FD; each instance remembered on its own what it had
#     written, skipped the write, and could launch with FD on the OTHER instance's buffer.
# What FD holds is recorded once, by the global's own device address, and FD is written and the kernel launched under
# one lock (the open's history replay evaluates on its own thread).
_FD_NOW = {}
_FD_LOCK = threading.RLock()
# 11-09 night: her store's float32 samples on the card once per process, shared by every GpuEval (the viewport's and
# the release re-distance's each held its own 1.1 GB copy; phase0/pool_growth_probe.py: +1.16 GB at the first stroke)
_DEV_DATA = {}
OFFS8 = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], np.int64)   # order = (i>>3)*4+(j>>3)*2+(k>>3)


class GpuEval:
    def __init__(self, cp):
        self.cp = cp
        src = open(os.path.join(HERE, 'program.cu'), encoding='utf-8').read()
        # r29: single precision by default (this card runs fp64 at a small fraction of fp32; ~0.1 um at body scale);
        # FIELDSTUDIO_GEVAL_DOUBLE=1 keeps double. No fast math: the CPU interpreter's operations, in order.
        self.real = 'double' if os.environ.get('FIELDSTUDIO_GEVAL_DOUBLE') == '1' else 'float'
        self.mod = cp.RawModule(code=src, options=('-std=c++14', '-DREAL=%s' % self.real))
        self.k = self.mod.get_function('eval_bricks')
        # milestone (r-dbl): programs holding her store run in double: her card volume's nodes are her store's nodes,
        # many on brick faces, where float32 header values flipped the brick choice (a missing brick: the card read
        # -6 mm where the CPU read 0.09 mm); in double the card takes the CPU's own decisions
        self.mod_d = self.mod if self.real == 'double' else cp.RawModule(code=src, options=('-std=c++14', '-DREAL=double'))
        self.k_d = self.mod_d.get_function('eval_bricks')
        self.k8 = self.mod.get_function('eval_nodes8')
        self.k8_d = self.mod_d.get_function('eval_nodes8')
        self._prog = {}
        self._dev_data = _DEV_DATA   # r-ptr: her store's data on the card, once per store (and per process, 11-09)
        self._fd = {}            # 11-09: object id -> the float32 store buffer its program reads (program.cu FD)

    def forget(self, oid):
        """fs-mouth 11-12: an object gone from the scene lets go of its device programs and store buffer (undoing the
        mouth set had kept them on the card: fs-bench-full mem_card1, 11.09 -> 11.05 GB)"""
        for k in (oid, (oid, 'f')):
            self._prog.pop(k, None)
        self._fd.pop(oid, None)

    def _program(self, o, fp32=False):
        comp = o.compiled()
        key = id(comp)
        pid = (o.id, 'f') if fp32 else o.id
        hit = self._prog.get(pid)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2], hit[3]
        code, prm = comp[0], comp[1]
        d_code = self.cp.asarray(np.ascontiguousarray(code, np.int32).reshape(-1))
        rows = [int(r) for r in np.nonzero(np.asarray(code)[:, 0] == 53)[0]]     # OP_FV8 (her store)
        dt = np.float64 if (self.real == 'double' or (rows and not fp32)) else np.float32
        if rows:
            # r-ptr: the CPU program reads her store's shared array through its address; the card's program gets the
            # data appended to its own copy, slot 15 = its offset from the op's parameters (program.cu fv8_d)
            from ..core import fv8base
            hp = np.array(prm, dt)
            # 11-09: the store's samples stay float32 on the card, in their own buffer (program.cu FD); the parameter
            # slot 15 holds the store's offset in that buffer (r: 1.1 GB float32 went to the card as 2.2 GB float64, and
            # once more inside the concatenated parameters: 4.4 GB)
            bufs, n = [], 0
            for r in rows:
                oo = int(np.asarray(code)[r, 1])
                addr = int(prm[oo + 46])
                dd = self._dev_data.get(addr)
                if dd is None:
                    dat = next(d for d in fv8base._DATA.values() if d.ctypes.data == addr)
                    dd = self._dev_data[addr] = self.cp.asarray(np.ascontiguousarray(dat, np.float32))
                bufs.append(dd)
                hp[oo + 15] = n
                n += dd.size
            fd = bufs[0] if len(bufs) == 1 else self.cp.concatenate(bufs)
            d_prm = self.cp.asarray(hp)
            self._fd[o.id] = fd
        else:
            d_prm = self.cp.asarray(np.ascontiguousarray(prm, dt))
        self._prog[pid] = (key, d_code, d_prm, int(code.shape[0]), bool(rows))
        return d_code, d_prm, int(code.shape[0])

    def _set_fd(self, mod, fd):
        """program.cu's FD <- the device address of fd (the float32 store buffer); the caller holds _FD_LOCK through
        its launch"""
        g = mod.get_global('FD')
        if _FD_NOW.get(g.ptr) == fd.data.ptr:
            return
        # written by a fill (the value travels as the kernel's argument), as gpu/scene.py sets its globals: no host
        # buffer whose lifetime the write depends on
        self.cp.ndarray((1,), self.cp.uint64, g)[...] = fd.data.ptr
        _FD_NOW[g.ptr] = fd.data.ptr

    def eval(self, scene, o, origins, sculpt_keys0, rows=None, sdf=None, stride=1, fp32=False, n8=False):
        """origins (n, 3) local GPU brick origins; sculpt_keys0 (n, 3): the sculpt brick holding each brick's node 0.
        rows given: written into sdf rows as well. -> cupy (n, 729) float32."""
        with _FD_LOCK:
            return self._eval(scene, o, origins, sculpt_keys0, rows, sdf, stride, fp32, n8)

    def _eval(self, scene, o, origins, sculpt_keys0, rows, sdf, stride, fp32, n8):
        cp = self.cp
        n = len(origins)
        out = cp.empty((max(n, 1), 512 if n8 else 729), cp.float32)   # n8: each brick's own 8^3 nodes (stride 1)
        if n == 0:
            return out[:0]
        d_code, d_prm, ncode = self._program(o, fp32)
        pid = (o.id, 'f') if fp32 else o.id
        store = self._prog[pid][4]
        if fp32 and self.real != 'double':
            # 11-09 (the release re-distance on the card): a store program in single precision too, on request. Since
            # r69 her field is continuous across brick faces, so the brick choice a float32 position flips there no
            # longer changes the value (gpu/redist.py measures it against the CPU)
            kern = self.k
            if store:
                self._set_fd(self.mod, self._fd[o.id])
        else:
            kern = self.k_d if store else self.k
            if store:
                self._set_fd(self.mod_d, self._fd[o.id])
        g = o.sculpt
        st = scene.store
        m = int(stride) + 1                     # r34: sculpt bricks per axis under one GPU brick of node spacing stride
        offs = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing='ij'), -1).reshape(-1, 3)
        sk = (np.asarray(sculpt_keys0, np.int64)[:, None, :] + offs[None]).reshape(-1, 3)
        slots = np.full(len(sk), -1, np.int64)
        if g.index.size:
            rel = sk - np.asarray(g.origin, np.int64)
            shp = np.array(g.index.shape)
            ok = np.all((rel >= 0) & (rel < shp), 1)
            slots[ok] = g.index[rel[ok, 0], rel[ok, 1], rel[ok, 2]]
        used = np.unique(slots[slots >= 0])
        remap = np.full(int(slots.max()) + 2 if len(slots) else 1, -1, np.int64)
        remap[used] = np.arange(len(used))
        dsl = np.where(slots >= 0, remap[np.maximum(slots, 0)], -1).astype(np.int32)
        dblk = cp.asarray(np.ascontiguousarray(st.delta[used], np.float32).reshape(-1)) if len(used) else cp.zeros(1, cp.float32)
        th = 128
        if n8:
            k8 = (self.k8_d if kern is self.k_d else self.k8)
            k8(((n * 512 + th - 1) // th,), (th,),
               (d_code, d_prm, np.int32(0), np.int32(ncode), cp.asarray(np.ascontiguousarray(origins, np.float64).reshape(-1)),
                np.float64(g.h), cp.asarray(dsl), dblk, np.int32(n), out))
            return out[:n]
        kern(((n * 729 + th - 1) // th,), (th,),
               (d_code, d_prm, np.int32(0), np.int32(ncode), cp.asarray(np.ascontiguousarray(origins, np.float64).reshape(-1)),
                np.float64(g.h), cp.asarray(dsl), dblk,
                cp.asarray(np.asarray(rows, np.int32)) if rows is not None else np.intp(0), np.int32(n),
                sdf if sdf is not None else np.intp(0), out, np.int32(stride)))
        return out[:n]
