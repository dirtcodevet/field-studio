"""fs-hair (11-12; was fs-hair/fibre_points/fsh_points.py): the hair's light at strand points for the Field Studio window, by the SAME calls as run_hair's fibre layer
(engine/hair/fibre_layer.cuh fl_trace_round, the per-hit block with FSHADOW), so the window and the review renderer give one
answer (design: append to FLm.source(npt) in our own module,
fibre_layer untouched).

Per point (strand i, point k): vis (rgb) the key through the hair (fl_Trc over a cone of the key's size) and her body
(fl_body_vis); vsk (rgb) the sky (one cone of 0.35 rad along N, or the four cosine-weighted cones when skyrays=1) and her
body's share (fl_body_sky); bk (rgb) the backscatter Bk(sigma, view perpendicular) x dense, dense = 1 - mean(tdn), tdn the
light through the hair below (toward -N) x vsk. N: the root normal blended to her body's gradient over the first 1-4 cm of a
long strand (nloc). The window shades: the key's lobes x vis, the sky term x vsk, plus bk x (sun vis 0.5 + sky vsk pi/2) / pi.

Cost (hair-shadow, measured on n2c 11-11): ~0.25-0.3 us per cone; points start inside the mass (the high end). Launched in
chunks of points, each timed so a launch stays near BAND_S (the display watchdog resets the card past ~2 s).

Caution (hair-shadow): the wide sky cone gathers only the cells on its centre line; for her skin it made the scalp's shadow 7 %
too dark and hard. Inside the mass it is consistent with run_hair, not measured right."""
import os
import threading
import time

import numpy as np

BAND_S = 0.35
_SRC_LOCK = threading.Lock()
_ORIG_SOURCE = None

POINTS_CU = r"""
// fs-hair (11-12): the per-hit light block of fl_trace_round at given strand points (ids = i x FNPT + k); out m x 9
extern "C" __global__ void fsh_points(int m, const int* pid, float* out, int i0) {
    int t = i0 + blockIdx.x * blockDim.x + threadIdx.x; if (t >= m) return;
    int id = pid[t]; int i = id / FNPT, k = id - i * FNPT;
    float* o = out + (size_t)t * 9;
    const float* p = FP + (size_t)id * 3; float3 Ph = make_float3(p[0], p[1], p[2]);
    if (!(Ph.x == Ph.x) || i >= FD.n) { for (int c = 0; c < 6; ++c) o[c] = 1.f; o[6] = 0.f; o[7] = 0.f; o[8] = 0.f; return; }
    int kk = min(k, FSPN - 1); float uu = k > kk ? 1.f : 0.f;
    float rr = frad(i, kk, uu); float s = FLEN[i] * ((float)kk + uu) / (float)FSPN;
    float3 N = make_float3(FNRM[3*i], FNRM[3*i+1], FNRM[3*i+2]);
    if (FBODY && FD.pad3 > 0.5f) { float wn = fminf(fmaxf((s - 0.01f) / 0.03f, 0.f), 1.f); wn = wn * wn * (3.f - 2.f * wn);
        if (wn > 0.f) N = hnorm(N * (1.f - wn) + fl_body_grad(Ph) * wn); }
    float3 one = make_float3(1.f, 1.f, 1.f), vis = one, vsk = one, tdn = one;
    if (FBODY) { vis = one * fl_body_vis(Ph, HSUN, FD.wl > 0.125f ? 1.f / FD.wl : 8.f, 0.4f); vsk = one * fl_body_sky(Ph); }
    if (vis.x > 1e-3f) vis = vis * fl_Trc(Ph + HSUN * (2.5f * rr), HSUN, fmaxf(0.00465f, FD.wl) + 2.f * rr / 0.01f, 0.25f);
    if (FD.pad0 == 0) vsk = vsk * fl_Trc(Ph + N * (2.5f * rr), N, 0.35f, 0.25f);
    else {
        float3 a1 = hnorm(hcross(N, fabsf(N.z) < 0.9f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f))), a2 = hcross(N, a1);
        float3 acc = fl_Trc(Ph + N * (2.5f * rr), N, 0.35f, 0.25f, 1) * 0.25f;
        const float cz[3] = {1.f, -0.5f, -0.5f}, sz[3] = {0.f, 0.8660254f, -0.8660254f};
        for (int q3 = 0; q3 < 3; ++q3) {
            float3 dq = hnorm(N * 0.5f + (a1 * cz[q3] + a2 * sz[q3]) * 0.8660254f);
            acc = acc + fl_Trc(Ph + dq * (2.5f * rr), dq, 0.35f, 0.18f, 1) * 0.25f;
        }
        vsk = vsk * acc;
    }
    tdn = fl_Trc(Ph - N * (2.5f * rr), -N, 0.35f, 0.05f) * vsk;
    float dense = 1.f - (tdn.x + tdn.y + tdn.z) / 3.f;
    float3 bk = make_float3(0.f, 0.f, 0.f);
    if (dense > 0.f) bk = fl_backscatter(fl_sigma(i, kk, uu), 0.f) * (dense * FD.amb);
    o[0] = vis.x; o[1] = vis.y; o[2] = vis.z; o[3] = vsk.x; o[4] = vsk.y; o[5] = vsk.z; o[6] = bk.x; o[7] = bk.y; o[8] = bk.z;
}
"""


def layer(cp, hs, body=None, key_r=0.0, skyrays=0, nloc=1):
    """a FibreLayer over hs whose module also carries fsh_points (FLm.source patched for the construction only)"""
    import engine.hair.fibre_layer as FLm
    global _ORIG_SOURCE
    # FibreLayer builds its module from FLm.source(npt): ours appended under a lock, from the TRUE original captured
    # once, restored after (two shade jobs patching at once appended it twice: 'fsh_points already defined', herL4 11-12)
    with _SRC_LOCK:
        if _ORIG_SOURCE is None:
            _ORIG_SOURCE = getattr(FLm.source, "_fsh_orig", FLm.source)
        orig = _ORIG_SOURCE

        def patched(npt=24):
            return orig(npt) + POINTS_CU
        patched._fsh_orig = orig
        FLm.source = patched
        try:
            FL = FLm.FibreLayer(cp, hs, ht=(hs.meta.get("ht") if hasattr(hs, "meta") else None), body=body, key_r=key_r,
                                nloc=nloc, skyrays=skyrays)
        finally:
            FLm.source = orig
    FL.k_points = FL.mod.get_function("fsh_points")
    return FL


def sample_ks(npt, K=3):
    """the points lit exactly: every K-th along each strand and the tip (the drawer interpolates between)"""
    return np.unique(np.r_[np.arange(0, npt, K), npt - 1]).astype(np.int64)


def sample_points(FL, sun_dir, ks, info=None):
    """-> (n, len(ks), 9) float32 on the host: vis 3, vsk 3, bk 3 at the points ks of every strand. The card's output is
    only these samples (her 138k strands x 22 x 9 floats ~110 MB, freed with the layer)"""
    cp = FL.cp
    FL.light(sun_dir, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    n, npt = FL.hs.n, FL.hs.a["P"].shape[1]
    ids = (np.arange(n, dtype=np.int64)[:, None] * npt + ks[None, :]).reshape(-1).astype(np.int32)
    m = len(ids)
    d_ids = cp.empty(m, cp.int32)
    d_ids.set(ids)
    out = cp.zeros((max(m, 1), 9), cp.float32)
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    # fs-shell 11-14 (Tier 0 part 1): the first chunk 4,096 points (was 65,536: a first meeting with dense hair could run
    # 16x the measured size in one launch), growth <= 1.5x, and a launch past 1 s stops the call (nothing relaunched)
    i0, chunk, worst, launches = 0, 4096, 0.0, 0
    t = time.perf_counter()
    while i0 < m:
        c = min(chunk, m - i0)
        e0.record()
        FL.k_points(((c + 127) // 128,), (128,), (np.int32(i0 + c), d_ids, out, np.int32(i0)))
        e1.record(); e1.synchronize()
        dt = cp.cuda.get_elapsed_time(e0, e1) * 1e-3
        worst = max(worst, dt); launches += 1
        if dt > float(os.environ.get('FL_ABORT_MS', '1000')) / 1000.0:
            cp.cuda.Device().synchronize()
            raise RuntimeError('hair light: a launch took %.0f ms (%d points): stopped, nothing more launched' % (dt * 1e3, c))
        i0 += c
        chunk = int(max(1024, min(chunk * 1.5, chunk * BAND_S / max(dt, 1e-4))))
    O = out[:m].get().reshape(n, len(ks), 9)
    del out, d_ids
    if info is not None:
        info.update(points=m, launches=launches, worst_launch_s=round(worst, 3), kernel_s=round(time.perf_counter() - t, 2))
    return O


def window_light(g, key_dir, key_r=0.0, K=3, info=None):
    """run_hair's light for groom g's current hair at every K-th point (and the tip), for the drawer's light buffer:
    -> (uint16 (n, ns, 9) fixed point / 65535: the key's transmittance rgb, the sky's rgb, the backscatter weight rgb; K).
    key_dir: toward the key, in the object's frame. On the card; her body from the groom's lattice when it has one. The
    fibre layer (her strands, their span grid) lives on the card only while this runs (its size in info['layer_mb'])"""
    import sys
    import cupy as cp
    from .generate import ROOT
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    t = time.perf_counter()
    hs = g.to_hairset()
    npt = hs.a["P"].shape[1]
    ks = sample_ks(npt, K)
    body = None
    gr = getattr(g, 'grid', None)
    if gr is not None and getattr(gr, 'A', None) is not None:
        body = (np.ascontiguousarray(gr.A, np.float32), np.asarray(gr.lo, np.float64), float(gr.h))
    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    FL = layer(cp, hs, body=body, key_r=float(key_r))
    del hs
    t1 = time.perf_counter()
    layer_mb = round((pool.used_bytes() - used0) / 1e6, 1)
    try:
        O = sample_points(FL, np.asarray(key_dir, np.float64), ks, info=info)
    finally:
        FL.keep = []
        del FL
        pool.free_all_blocks()
    FLB = np.clip(np.rint(np.clip(O, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)
    if info is not None:
        info.update(build_s=round(t1 - t, 2), total_s=round(time.perf_counter() - t, 2), body=body is not None,
                    layer_mb=layer_mb, light_mb=round(FLB.nbytes / 1e6, 1), K=int(K), ns=int(len(ks)))
    return FLB, int(K)


if __name__ == "__main__":
    # compile only (outside the lock, as hair-shadow's hs_precompile): the module for a point count, into CuPy's cache
    import os, sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
    sys.path.insert(0, ROOT)
    import cupy as cp
    import engine.hair.fibre_layer as FLm
    inc = FLm.cuda_include()
    opts = ("--use_fast_math", "-std=c++17") + (("-I" + inc,) if inc else ())
    for npt in [int(a) for a in sys.argv[1:]] or [64]:
        t = time.time()
        mod = cp.RawModule(code=FLm.source(npt) + POINTS_CU, options=opts)
        mod.compile()
        mod.get_function("fsh_points")
        print("npt %d: compiled + loaded in %.1f s" % (npt, time.time() - t), flush=True)
