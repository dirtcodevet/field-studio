"""Hair Studio flow (numpy + scipy, Apache-2.0): the comb - strands walked from their roots over the scalp along a
direction field and hung under gravity, kept out of the body - and the gathering after a tie (ponytail, bun).

The walk (one batch of strands, `walk`): at each step (step_mm) a strand's direction becomes
    d <- normalise(stiffness d + (1 - stiffness) normalise(comb_strength on(P) tangent(comb(P)) + gravity (0, 0, -1)))
where tangent() removes the comb's normal part at the nearest skin point and on(P) fades the comb out as the strand
leaves the scalp (it is a brush: it acts along the scalp). Gravity is free, so on the top of the head it presses the
strand onto the scalp (the push-out keeps it there and it slides along) and past the head's widest point it takes the
strand off it: the hair hangs from where the head turns under, as real hair does. After each step the point is pushed
out to h(s) above the skin (the layering: s tan(lift) up to a plateau, the upstream hair lies on top) along the nearest
vertex's normal, any constraint (the ear tuck) is applied, and the step is renormalised. Gathered strands walk toward
the gather point and stop `converge_mm` from it.

The comb for loose styles (`comb_field`): away from the part (on the top of the head), back, to one side, up; an extra
back-sweep over the front and temples above the ears; a tuck that aims the hair in front of an ear over its top and
behind it; a push out of the face corridor.

Gathering: `tie_frame` (the tie's centre, axis and frame on the scalp), `bridge` (each strand from where it left the
scalp to its place in the tie's cross-section, a cubic Hermite), `spine_walk` (the tail's centre line: from the tie
along its axis, then hanging, kept its radius off the body), `coil_spine` (a bun: the rope coiled about the tie axis,
each turn further out, the end tucked under), `rmf` (rotation-minimising frames) and `place` (each strand at its
offset in the tail's cross-section, which fans out, tapers and twists).
"""
import numpy as np

from .scalp import smoothstep

Z = np.array([0.0, 0.0, 1.0])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def comb_field(scalp, cfg, ears_cfg, corr_cfg, side_sign, P, s, ids, Pv, sd):
    """loose styles: the comb vectors at P (m, 3) of strands ids on part side side_sign[ids]; Pv / sd = each point's
    nearest skin vertex (position) and height above it. Returns (the comb: tangential, fades off the scalp, a free
    force: the face corridor)"""
    phi, th, _ = scalp.sph(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    sg = side_sign[ids]
    F = np.zeros_like(P)
    # the part zone (the front `depth` of the top): lateral, away from the part line
    depth = cfg["_part_depth"]
    w_pz = (1 - smoothstep(55.0, 88.0, th)) * (1 - smoothstep(depth - 0.02, depth + 0.01, y - cfg["_y_front"]))
    F[:, 0] += cfg["away_from_part"] * sg * w_pz
    # behind it, the crown: away from the whorl (the part's end), down the back
    if cfg["_whorl"] is not None:
        w_cr = (1 - w_pz) * (1 - smoothstep(60.0, 100.0, th))
        F += cfg["away_from_part"] * w_cr[:, None] * _unit(P - cfg["_whorl"])
    F[:, 1] += cfg["back"]
    F[:, 0] += cfg["side"]
    F[:, 2] += cfg["up"]
    ears = scalp.ears
    z_eb = min(e["z_bot"] for e in ears.values())
    z_et = max(e["z_top"] for e in ears.values())
    # the back-sweep over the front and the temples, above the ears' lower end
    w_fb = smoothstep(18.0, 50.0, np.abs(phi)) * (1 - smoothstep(105.0, 135.0, np.abs(phi))) * \
        smoothstep(z_eb, z_eb + 0.02, z) * (1 - smoothstep(z_et + 0.05, z_et + 0.08, z))
    F[:, 1] += cfg["face_back"] * w_fb
    if cfg.get("behind_ears", True):
        for e_s, e in ears.items():
            lat = smoothstep(0.045, 0.060, e_s * x)
            # in front of the ear's back edge and below its top + 15 mm: aim over the top and behind it
            front = 1 - smoothstep(e["y_back"] + 0.0, e["y_back"] + ears_cfg["back_mm"] / 1000 + 0.004, y)
            zone = lat * front * smoothstep(e["z_bot"] - 0.025, e["z_bot"] - 0.005, z) * \
                (1 - smoothstep(e["z_top"] + 0.010, e["z_top"] + 0.025, z)) * (y > e["y_front"] - 0.03)
            if cfg.get("_sideburn") is not None:
                sb = cfg["_sideburn"][ids]
                zone = zone * ~sb
                F[sb] = np.array([0.0, 0.4, -1.0])
            T = np.stack([np.full(len(P), e_s * ears_cfg["x_max_mm"] / 1000),
                          np.full(len(P), e["y_back"] + ears_cfg["back_mm"] / 1000 + 0.004),
                          np.full(len(P), e["z_top"] + ears_cfg["over_mm"] / 1000)], 1)
            F += 3.0 * zone[:, None] * _unit(T - P)
            # the temple sweep (10-29): above the ear and in front of its back edge - the temple and the side of the head -
            # the comb aims at the same tuck point behind the ear's top, which lies back and DOWN from there: hair from
            # above passes over the temple on its way behind the ear (a swept-back, tucked style covers the temple; the
            # straight-back sweep left it bare)
            tsw = cfg.get("temple_sweep", 0.0)
            if tsw > 0:
                lat2 = smoothstep(0.040, 0.055, e_s * x)
                front2 = 1 - smoothstep(e["y_back"] - 0.005, e["y_back"] + 0.010, y)
                band2 = smoothstep(e["z_top"] - 0.005, e["z_top"] + 0.010, z) *                     (1 - smoothstep(e["z_top"] + 0.050, e["z_top"] + 0.075, z)) * (y > e["y_front"] - 0.045)
                F += tsw * (lat2 * front2 * band2)[:, None] * _unit(T - P)
    Ff = np.zeros_like(P)
    # the hair mass: off the scalp, the upper layers (higher plateau) fan out a little (the strands do not collide with
    # each other; real hair stacks ~10-20 mm thick and falls wider than the head)
    if cfg.get("volume", 0) > 0 and cfg.get("_Hp") is not None:
        Hp = cfg["_Hp"][ids]
        off = smoothstep(Hp + 0.004, Hp + 0.014, sd)
        oh = np.stack([x, y - scalp.c[1] - 0.03, np.zeros_like(x)], 1)
        below = 1 - smoothstep(z_eb - 0.02, z_eb, z)
        Ff += (cfg["volume"] * (Hp / max(cfg["_Hp_max"], 1e-6)) * off * below)[:, None] * _unit(oh)
    # the front drape (10-29 night): strands rooted in front of the ears' line (_drape_w per strand) are drawn forward below
    # the ears' lower end until they hang in front of the shoulders (hair brought forward over the shoulders; with the
    # lengths behind, the drape is 0)
    dr = cfg.get("drape_front", 0.0)
    if dr > 0 and cfg.get("_drape_w") is not None:
        wd = cfg["_drape_w"][ids] * (1 - smoothstep(z_eb - 0.02, z_eb + 0.01, z)) *             smoothstep(scalp.c[1] - 0.075, scalp.c[1] - 0.035, y)
        Ff[:, 1] -= dr * wd
    # the face corridor: a point over the face skin is swept back and up, hard, on or off the scalp (a free force)
    if corr_cfg is not None:
        cm = corridor_mask(scalp, corr_cfg, Pv, sd, near_mm=40.0, margin_mm=-2.0)
        if cm.any():
            Ff[cm] += 6.0 * np.array([0.0, 1.0, 0.6])
    return F, Ff


def comb_gather(scalp, G, pull, P, s, ids, Pv, sd):
    """gathered styles: toward the gather point G - as a comb on the scalp and as a free pull (tension) off it, so a
    strand that leaves the scalp (at the sides, under the occiput) is still drawn to the tie, along the head"""
    u = _unit(G[None, :] - P)
    return pull * u, pull * u


def corridor_mask(scalp, corr_cfg, Pv, Psd, near_mm=30.0, margin_mm=0.0):
    """points over the face skin: their nearest skin vertex Pv outside the hairline by > outside_mm (+ margin_mm),
    within phi_deg of the front, above the ears' lower end, and within near_mm of the skin (Psd = their height)"""
    phv, _, _ = scalp.sph(Pv)
    vin = scalp.d_inside_mm(Pv)
    z_eb = min(e["z_bot"] for e in scalp.ears.values())
    m = (vin < -(corr_cfg["outside_mm"] + margin_mm)) & (np.abs(phv) < corr_cfg["phi_deg"]) & (Pv[:, 2] > z_eb) & \
        (Psd < near_mm / 1000)
    if corr_cfg.get("eyes_only"):
        m &= Pv[:, 2] > z_eb + 0.025
    return m


def ear_tuck(scalp, ears_cfg, P):
    """behind each ear: at the ear's height no point lateral of x_max (rising to x_top 40 mm above the ear's top) once
    behind its back edge"""
    out = P.copy()
    for e_s, e in scalp.ears.items():
        zc = out[:, 2]
        wz = smoothstep(e["z_bot"] - 0.020, e["z_bot"] - 0.002, zc) * (1 - smoothstep(e["z_top"] + 0.04,
                                                                                     e["z_top"] + 0.06, zc))
        xl = ears_cfg["x_max_mm"] / 1000 + (ears_cfg["x_top_mm"] - ears_cfg["x_max_mm"]) / 1000 * \
            smoothstep(e["z_top"] + 0.004, e["z_top"] + 0.04, zc)
        # behind the ear's back edge, or above its top (in its y range +- 15 mm)
        behind = np.maximum(smoothstep(e["y_back"] + 0.001, e["y_back"] + 0.006, out[:, 1]),
                            smoothstep(e["z_top"] + 0.002, e["z_top"] + 0.006, zc) *
                            smoothstep(e["y_front"] - 0.015, e["y_front"] - 0.005, out[:, 1]))
        over = np.maximum(0, e_s * out[:, 0] - xl) * wz * behind
        out[:, 0] -= e_s * over
    return out


def walk(surf, X, N, d0, L_max, step, stiffness, gravity, comb_strength, comb, h_fn, constraint=None, stop=None,
         ids=None, workers=4):
    """walk one batch: roots X (m, 3), root normals N, start directions d0, lengths L_max (m,). comb(P, s, ids) ->
    vectors; h_fn(s, ids[, j]) -> the height kept above the skin (m) - j the skin vertex under the point (a 3-argument h_fn
    lets the height follow the place: the stack); constraint(P) -> P; stop(P, s, ids) -> bool (m,)
    (the strand stops there, e.g. at the gather point). Returns the polylines (m, S + 1, 3) at arc k * step (a stopped
    or finished strand repeats its last point), the stop step (S where none), the last sd."""
    m = len(X)
    S = int(np.ceil(L_max.max() / step))
    ids = np.arange(m) if ids is None else ids
    out = np.empty((m, S + 1, 3))
    out[:, 0] = X
    P = X.copy()
    d = _unit(d0)
    s = np.zeros(m)
    alive = np.ones(m, bool)
    stop_at = np.full(m, S)
    sd, j = surf.sd(P)
    g = gravity * np.array([0.0, 0.0, -1.0])
    h3 = h_fn.__code__.co_argcount >= 3 if hasattr(h_fn, "__code__") else False
    hf = (lambda s_, i_, j_: h_fn(s_, i_, j_)) if h3 else (lambda s_, i_, j_: h_fn(s_, i_))  # noqa: E731
    for k in range(S):
        a = alive & (s < L_max)
        if not a.any():
            out[:, k + 1:] = P[:, None, :]
            break
        ia = np.nonzero(a)[0]
        Pa, da, sa = P[ia], d[ia], s[ia]
        n = surf.VN[j[ia]]
        h = hf(sa, ids[ia], j[ia])
        on = 1 - smoothstep(h + 0.002, h + 0.010, sd[ia])
        C = comb(Pa, sa, ids[ia], surf.V[j[ia]], sd[ia])
        C, Ff = C if isinstance(C, tuple) else (C, 0.0)
        C = C - np.einsum("ij,ij->i", C, n)[:, None] * n
        Ft = comb_strength * on[:, None] * C + Ff + g
        da = _unit(stiffness * da + (1 - stiffness) * _unit(Ft))
        Pn = Pa + step * da
        sdn, jn = surf.sd(Pn)
        hn = hf(sa + step, ids[ia], jn)
        need = np.maximum(hn - sdn, 0)
        Pn = Pn + need[:, None] * surf.VN[jn]
        if constraint is not None:
            Pn = constraint(Pn)
        dv = Pn - Pa
        Lv = np.linalg.norm(dv, axis=1, keepdims=True)
        dn = dv / np.maximum(Lv, 1e-12)
        Pn = Pa + dn * step
        sdn, jn = surf.sd(Pn)
        P[ia], d[ia], s[ia] = Pn, dn, sa + step
        sd[ia], j[ia] = sdn, jn
        if stop is not None:
            st = stop(Pn, s[ia], ids[ia])
            if st.any():
                stop_at[ia[st]] = k + 1
                alive[ia[st]] = False
        out[:, k + 1] = P
    return out, stop_at, sd


def resample_poly(poly, step, L, K):
    """polylines sampled at arc k * step (m, S + 1, 3) -> K points uniform over [0, L] (m,)"""
    m, S1 = poly.shape[:2]
    t = np.linspace(0, 1, K)[None, :] * L[:, None] / step
    i0 = np.minimum(np.floor(t).astype(np.int64), S1 - 2)
    w = (t - i0)[..., None]
    r = np.arange(m)[:, None]
    return poly[r, i0] * (1 - w) + poly[r, i0 + 1] * w


def first_below(poly, step, ztip):
    """arc length where each polyline first goes below ztip (m,); the full length where it never does"""
    z = poly[..., 2]
    below = z < ztip[:, None]
    has = below.any(1)
    k = np.argmax(below, 1)
    k = np.maximum(k, 1)
    r = np.arange(len(poly))
    z0, z1 = z[r, k - 1], z[r, k]
    f = np.clip((z0 - ztip) / np.maximum(z0 - z1, 1e-9), 0, 1)
    L = (k - 1 + f) * step
    return np.where(has, L, (poly.shape[1] - 1) * step), has


# ------------------------------------------------------------------------------------------------ gathering
def tie_frame(scalp, phi_deg, theta_deg, tie_r, tail_dir):
    """the gather point G on the scalp (along the ray from the centre at phi, theta), its normal, the tie centre
    T = G + n (tie_r + 1.5 mm), the tie axis A (the normal turned toward tail_dir) and a frame U, W across it"""
    ph, th = np.radians(phi_deg), np.radians(theta_deg)
    ray = np.array([np.sin(th) * np.sin(ph), -np.sin(th) * np.cos(ph), np.cos(th)])
    V = scalp.V
    w = V - scalp.c
    t = w @ ray
    dist = np.linalg.norm(w - t[:, None] * ray, axis=1) + 10 * (t < 0)
    j = int(np.argmin(dist))
    G = V[j]
    n = scalp.VN[j]
    T = G + n * (tie_r + 0.0015)
    A = _unit(n + np.asarray(tail_dir, np.float64))
    U = _unit(np.cross(A, Z) if abs(A @ Z) < 0.95 else np.cross(A, [1.0, 0, 0]))
    W = np.cross(A, U)
    return G, n, T, A, U, W


def bridge(Pa, da, D, A, step):
    """cubic Hermite from Pa (direction da) to D (direction A) at ~step spacing -> list of (m, k, 3) arrays (padded to
    the longest by repeating D), the lengths"""
    chord = np.linalg.norm(D - Pa, axis=1)
    k = int(np.ceil(max(chord.max(), step) * 1.3 / step)) + 1
    u = np.linspace(0, 1, k)[None, :, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    m0 = da[:, None, :] * chord[:, None, None]
    m1 = np.broadcast_to(A, Pa.shape)[:, None, :] * chord[:, None, None]
    B = h00 * Pa[:, None, :] + h10 * m0 + h01 * D[:, None, :] + h11 * m1
    Lb = np.linalg.norm(np.diff(B, axis=1), axis=2).sum(1)
    return B, Lb


def spine_walk(surf, T, A, length, step, stiffness, gravity, radius_fn, workers=4):
    """the tail's centre line from T along A, then hanging; kept radius_fn(s) off the body -> (S + 1, 3)"""
    S = int(np.ceil(length / step))
    P = np.array(T, np.float64)
    d = _unit(np.asarray(A, np.float64))
    out = [P.copy()]
    g = gravity * np.array([0.0, 0.0, -1.0])
    for k in range(S):
        d = _unit(stiffness * d + (1 - stiffness) * _unit(g + 1e-9))
        Pn = P + step * d
        sd, j = surf.sd(Pn[None])
        need = max(radius_fn((k + 1) * step) - float(sd[0]), 0)
        Pn = Pn + need * surf.VN[j[0]]
        dv = Pn - P
        d = dv / max(np.linalg.norm(dv), 1e-12)
        P = P + d * step
        out.append(P.copy())
    return np.array(out)


def coil_spine(T, A, U, W, length, step, r0, R, turns, rise, tuck):
    """a bun's rope centre line: from the tie at radius r0 out to R in the first quarter turn, `turns` turns about the
    axis A, each turn `rise` further out along A, the last `tuck` metres turning under toward the axis -> (S + 1, 3)"""
    S = int(np.ceil(length / step))
    # parametrise by angle, then resample by arc
    psi = np.linspace(0, 2 * np.pi * turns, 4000)
    Rr = r0 + (R - r0) * smoothstep(0, np.pi / 2, psi)
    h = rise * psi / (2 * np.pi)
    C = T + h[:, None] * A + Rr[:, None] * (np.cos(psi)[:, None] * U + np.sin(psi)[:, None] * W)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    # the tuck: the last `tuck` of arc spirals in toward the axis and back down to the tie's height
    Lc = arc[-1]
    f = smoothstep(Lc - tuck, Lc, arc)
    Rt = Rr * (1 - 0.65 * f)
    ht = h * (1 - 0.8 * f)
    C = T + ht[:, None] * A + Rt[:, None] * (np.cos(psi)[:, None] * U + np.sin(psi)[:, None] * W)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    t = np.minimum(np.arange(S + 1) * step, arc[-1])
    out = np.stack([np.interp(t, arc, C[:, k]) for k in range(3)], 1)
    # past the coil's end (strands longer than it): continue along the last direction, back toward the scalp
    over = np.arange(S + 1) * step - arc[-1]
    dl = _unit(C[-1] - C[-6])
    out += np.maximum(over, 0)[:, None] * (0.5 * dl - 0.5 * A)
    return out


def rmf(C, U0):
    """rotation-minimising frames along a polyline C (S, 3) starting with U0 -> T, U, W (S, 3) each (double
    reflection, Wang et al. 2008)"""
    S = len(C)
    T = np.gradient(C, axis=0)
    T = _unit(T)
    U = np.empty_like(C)
    U[0] = _unit(U0 - (U0 @ T[0]) * T[0])
    for i in range(S - 1):
        v1 = C[i + 1] - C[i]
        c1 = v1 @ v1
        if c1 < 1e-20:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2 / c1) * (v1 @ U[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        U[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-20 else rL
    W = np.cross(T, U)
    return T, U, W


def place(spine, Ts, Us, Ws, step, s_from, rho, theta, radius_fn, twist_pitch, L_tail):
    """strands along a spine: strand i at arc t in [0, L_tail_i] (step) sits at radius rho_i radius_fn(t) / rho_max and
    angle theta_i + 2 pi t / twist_pitch in the cross-section -> (m, S + 1, 3) (past its length a strand repeats its
    last point)"""
    S1 = len(spine)
    t = np.arange(S1) * step
    R = radius_fn(t)
    tw = 2 * np.pi * t / twist_pitch if twist_pitch > 0 else np.zeros_like(t)
    ang = theta[:, None] + tw[None, :]
    off = rho[:, None, None] * R[None, :, None] * (np.cos(ang)[..., None] * Us[None] + np.sin(ang)[..., None] * Ws[None])
    out = spine[None] + off
    kmax = np.minimum(np.ceil(L_tail / step).astype(np.int64), S1 - 1)
    idx = np.minimum(np.arange(S1)[None, :], kmax[:, None])
    return np.take_along_axis(out, idx[..., None], axis=1)
