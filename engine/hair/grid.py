"""Hair Studio grids (numpy + scipy, Apache-2.0): the body as a signed-distance grid and the hair as a fibre-volume
field, for the rod settle (DESIGN_v2 4.5, implementation step 3).

    SDFGrid    the body's lattice, built from its field by field_collider.FieldSDFGrid (mesh-free since 11-14)
        .sample(P, grad=False, fine=True) -> sd (m) [, gradient (..., 3)]   trilinear value, exact gradient of it
        .free_fraction(P, mm=None) -> a = clip(0.5 + sd / h, 0.2, 1)         free share of a cell centred at P
        .A, .a       float32 lattices: the signed distance (m) and a_v at the nodes
        .lo, .hi, .h, .shape, .report
    Splat(grid=None, mm=None, lo=None, dtype=np.float32)
        .field(Y, v) -> F     sample volumes v (m^3) at Y (m) spread trilinearly onto the lattice, cropped to the
                              samples (first node .c0), then blurred [1, 2, 1]: fibre volume per node cell (m^3)
        .free() -> a_v over that crop (1 away from the body)
        .phi(F) -> fibre fraction F / (a_v h^3)
        .adjoint(G, want_v=False) -> dE/dY (n, 3) [, dE/dv (n,)] for G = dE/dF on the crop of the last field()
        .pressure(Y, v, c, free_weight=True) -> E (J), dE/dY (n, 3), phi        the hair-hair term, f = c phi^3
    blur121(A) -> A blurred by [1, 2, 1] / 4 along each axis (zero outside: symmetric, so it is its own adjoint)
All positions in metres; arguments named *_mm in millimetres.

Why a grid for the body. The settle asks for ~0.5 M body distances in each of ~800 evaluations. A lattice filled once
from the body's field answers in well under a microsecond, and its gradient is the exact derivative of the value it
returns, which L-BFGS needs (a gradient that disagrees with its energy stalls the line search). Contact by a penalty on
a sampled signed distance is the level-set collision of film cloth and hair (Bridson, Marino & Fedkiw 2003, "Simulation
of clothing with folds and wrinkles"; Petrovic, Henne & Anderson 2005, Pixar).

How the grid is built: from the field (field_collider.FieldSDFGrid: the field at every node, the side of the nodes a
surface-band store holds no brick for from a flood fill). The lattice built from a triangle collider (exact triangle
distances near the skin, nearest surface samples beyond) went with the meshes (fs-hairclean 11-14, P14; the code is
archived with mesh.py). Fine crops (fine=[(lo, hi), ...] in metres, at fine_mm): a 3 mm grid can leak through an
ear's rim (DESIGN_v2 12), so boxes around the ears hold a 2 mm lattice. Inside a box the value is blended from the
coarse to the fine lattice over blend_mm at its faces (smoothstep), so value and gradient stay continuous and
consistent. Outside the grid's box the value is the box's edge value plus the distance out (the box is padded beyond
the body, so that far out the hair is free); the gradient follows.

Why the free-volume fraction. Next to the body part of a cell is skin and bone. Hair can fill only the free share of
it, a_v = clip(0.5 + SDF_v / h, 0.2, 1) (half free on the skin, the 0.2 floor keeps the fraction finite inside). Without
it the density gradient pulls hair into the head (DESIGN_v2 4.5, from the D3 groom prototype).

Why the splat. The hair-hair pressure needs the fibre fraction phi around every sample. Each sample's fibre volume is
spread onto the 8 nodes around it with trilinear weights (particle-to-grid transfer, Zhu & Bridson 2005 "Animating sand
as a fluid"; the volumetric hair of Petrovic 2005 and McAdams et al. 2009), then blurred [1, 2, 1] once per axis so the
field is smooth on the scale of a cell (cheaper than a Gaussian, and exactly symmetric). The adjoint runs the same
steps backwards: blur the energy's gradient on the grid, then gather it at each sample with the derivatives of its
trilinear weights. The pressure law: free energy density f(phi) = c phi^3 per unit free volume gives the swelling
pressure P = phi f' - f = 2 c phi^3 = k E phi^3 with c = k E / 2 (Van Wyk 1946; Goldstein, Warren & Ball 2012, PRL
108:078101).
"""
import numpy as np

_CORNERS = [(cx, cy, cz) for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)]


# ---------------------------------------------------------------- lattice helpers
def blur121(A):
    """[1, 2, 1] / 4 along each axis, zero outside the array. The operator is symmetric, so it is its own adjoint; it
    keeps the total when the outermost layer is empty (Splat crops with that margin)."""
    B = np.asarray(A)
    for ax in range(B.ndim):
        C = B * 2
        a = [slice(None)] * B.ndim
        b = [slice(None)] * B.ndim
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        C[tuple(a)] += B[tuple(b)]
        C[tuple(b)] += B[tuple(a)]
        C *= 0.25
        B = C
    return B


def _gather(flat, b, sx, sy, t, grad):
    """trilinear value of a flat lattice (strides sx, sy, 1) at cells with first node b and in-cell coordinates t
    (n, 3) in [0, 1], and the derivative of that interpolant with respect to t"""
    c000, c001, c010, c011 = flat[b], flat[b + 1], flat[b + sy], flat[b + sy + 1]
    c100, c101, c110, c111 = flat[b + sx], flat[b + sx + 1], flat[b + sx + sy], flat[b + sx + sy + 1]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    ux, uy, uz = 1 - tx, 1 - ty, 1 - tz
    c00 = c000 * ux + c100 * tx
    c01 = c001 * ux + c101 * tx
    c10 = c010 * ux + c110 * tx
    c11 = c011 * ux + c111 * tx
    c0 = c00 * uy + c10 * ty
    c1 = c01 * uy + c11 * ty
    v = c0 * uz + c1 * tz
    if not grad:
        return v
    g = np.empty((len(b), 3))
    g[:, 2] = c1 - c0
    g[:, 1] = (c10 - c00) * uz + (c11 - c01) * tz
    g[:, 0] = ((c100 - c000) * uy + (c110 - c010) * ty) * uz + ((c101 - c001) * uy + (c111 - c011) * ty) * tz
    return v, g


def _trilinear(A, lo, h, P, grad=False):
    """trilinear value of lattice A (first node lo, spacing h) at points P inside its box [and the exact gradient]"""
    shp = np.array(A.shape)
    f = (P - lo) / h
    i0 = np.clip(np.floor(f).astype(np.int64), 0, shp - 2)
    t = f - i0
    sx, sy = int(shp[1] * shp[2]), int(shp[2])
    b = i0[:, 0] * sx + i0[:, 1] * sy + i0[:, 2]
    r = _gather(A.ravel(), b, sx, sy, t, grad)
    if not grad:
        return r
    return r[0], r[1] / h


def _smooth01(t):
    return t * t * (3 - 2 * t)


class _Fine:
    """a finer lattice over a box, blended in over blend (m) at the box faces"""

    def __init__(self, lo, h, A, blend):
        self.lo, self.h, self.A = np.asarray(lo, np.float64), float(h), A
        self.hi = self.lo + self.h * (np.array(A.shape) - 1)
        self.blend = float(blend)

    def blend_into(self, P, v, g):
        inside = np.nonzero(np.all((P > self.lo) & (P < self.hi), 1))[0]
        if len(inside) == 0:
            return v, g
        Pi = P[inside]
        dlo, dhi = Pi - self.lo, self.hi - Pi
        d = np.minimum(dlo, dhi)
        t = np.clip(d / self.blend, 0, 1)
        s = _smooth01(t)
        w = s.prod(1)
        if g is None:
            fv = _trilinear(self.A, self.lo, self.h, Pi, False)
            v[inside] = (1 - w) * v[inside] + w * fv
            return v, g
        fv, fg = _trilinear(self.A, self.lo, self.h, Pi, True)
        ds = np.where((t > 0) & (t < 1), 6 * t * (1 - t) / self.blend, 0.0) * np.where(dlo < dhi, 1.0, -1.0)
        gw = np.stack([ds[:, 0] * s[:, 1] * s[:, 2], s[:, 0] * ds[:, 1] * s[:, 2], s[:, 0] * s[:, 1] * ds[:, 2]], 1)
        v0 = v[inside]
        g[inside] = (1 - w)[:, None] * g[inside] + w[:, None] * fg + (fv - v0)[:, None] * gw
        v[inside] = (1 - w) * v0 + w * fv
        return v, g


# ---------------------------------------------------------------- the body's signed-distance grid
class SDFGrid:
    """the body's signed distance (m, + outside) on a regular lattice: the reads. The lattice is built from the body's
    field by field_collider.FieldSDFGrid (which sets A, a, lo, hi, h, mm, shape, fine, key, report); the build from a
    triangle collider went with the meshes (P14, 11-14).

    A        float32 lattice (m), first node lo, spacing h = mm / 1000; hi its last node
    a        clip(0.5 + A / h, 0.2, 1): the free share of each node's cell
    fine     [_Fine]: finer lattices over boxes (the ears), blended in at their faces
    """

    def __init__(self, *a, **kw):
        raise TypeError("engine/hair is mesh-free (P14): build the body lattice from its field "
                        "(field_collider.FieldSDFGrid, or field_collider.install for generate / head)")

    def sample(self, P, grad=False, fine=True):
        """signed distance (m) at points P (..., 3) and, with grad, its gradient (the exact derivative of the
        trilinear value, blended across the fine boxes)"""
        P = np.asarray(P, np.float64)
        shp = P.shape[:-1]
        P = P.reshape(-1, 3)
        Pc = np.clip(P, self.lo, self.hi)
        o = P - Pc
        do = np.sqrt((o * o).sum(1))
        if grad:
            v, g = _trilinear(self.A, self.lo, self.h, Pc, True)
        else:
            v, g = _trilinear(self.A, self.lo, self.h, Pc, False), None
        out = do > 0
        if out.any():
            v = v + do
            if grad:
                g[out] = g[out] * (o[out] == 0) + o[out] / do[out, None]
        if fine:
            for fb in self.fine:
                v, g = fb.blend_into(P, v, g)
        if grad:
            return v.reshape(shp), g.reshape(shp + (3,))
        return v.reshape(shp)

    def free_fraction(self, P, mm=None):
        """the free share of a cell of mm (default the grid's) centred at P: clip(0.5 + sd / h, 0.2, 1)"""
        h = self.h if mm is None else mm / 1000.0
        return np.clip(0.5 + self.sample(P) / h, 0.2, 1.0)


# ---------------------------------------------------------------- the hair's fibre-volume field
class Splat:
    """trilinear splat of sample volumes onto a lattice cropped to the samples, a [1, 2, 1] blur, and the adjoint.

    grid   SDFGrid: the lattice is the grid's (same first node and spacing, so a_v is a slice of grid.a); with mm
           different from the grid's, the lattice keeps the grid's first node and a_v is sampled once
    mm     spacing without a grid (a_v = 1 everywhere: no body)
    dtype  float32 for the settle (DESIGN_v2); float64 for exact checks
    adjoint() uses the samples of the last field() call.
    """

    def __init__(self, grid=None, mm=None, lo=None, dtype=np.float32):
        self.dtype = dtype
        self._afull = None
        if grid is not None and (mm is None or abs(mm - grid.mm) < 1e-9):
            self.h, self.lo, self._afull = grid.h, grid.lo.copy(), grid.a
        elif grid is not None:
            self.h, self.lo = mm / 1000.0, grid.lo.copy()
            shape = tuple(int(s) for s in np.floor((grid.hi - grid.lo) / self.h + 1e-9).astype(int) + 1)
            n = int(np.prod(shape))
            a = np.empty(n, np.float32)
            gx = [self.lo[k] + self.h * np.arange(shape[k]) for k in range(3)]
            for b0 in range(0, n, 500000):
                ix, iy, iz = np.unravel_index(np.arange(b0, min(n, b0 + 500000)), shape)
                a[b0:b0 + len(ix)] = grid.free_fraction(np.stack([gx[0][ix], gx[1][iy], gx[2][iz]], 1), mm)
            self._afull = a.reshape(shape)
        else:
            if mm is None:
                raise ValueError("Splat needs a grid or a spacing mm")
            self.h = mm / 1000.0
            self.lo = np.zeros(3) if lo is None else np.asarray(lo, np.float64)
        self.vox = self.h ** 3
        self._st = None
        self.c0 = None
        self.shape = None

    def field(self, Y, v):
        """blurred fibre volume per node cell (m^3) on the crop around the samples Y (n, 3) of volume v (n,)"""
        Y = np.asarray(Y, np.float64).reshape(-1, 3)
        v = np.broadcast_to(np.asarray(v, np.float64), (len(Y),))
        f = (Y - self.lo) / self.h
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        if len(Y):
            c0 = i0.min(0) - 1                 # one empty layer each side: the blur loses nothing at the crop edge
            cs = i0.max(0) + 3 - c0
        else:
            c0, cs = np.zeros(3, np.int64), np.full(3, 3, np.int64)
        sx, sy = int(cs[1] * cs[2]), int(cs[2])
        li = i0 - c0
        b = li[:, 0] * sx + li[:, 1] * sy + li[:, 2]
        lins, wts = [], []
        for cx, cy, cz in _CORNERS:
            w = (t[:, 0] if cx else 1 - t[:, 0]) * (t[:, 1] if cy else 1 - t[:, 1]) * (t[:, 2] if cz else 1 - t[:, 2])
            lins.append(b + cx * sx + cy * sy + cz)
            wts.append(w * v)
        S = np.bincount(np.concatenate(lins), weights=np.concatenate(wts), minlength=int(np.prod(cs)))
        F = blur121(S.reshape(tuple(cs)).astype(self.dtype))
        self._st = (b, t, v, sx, sy)
        self.c0, self.shape = c0, tuple(int(s) for s in cs)
        return F

    def free(self):
        """a_v over the crop of the last field() (1 where the grid does not reach or with no grid)"""
        out = np.ones(self.shape, self.dtype)
        if self._afull is None:
            return out
        G = np.array(self._afull.shape)
        a0 = np.maximum(self.c0, 0)
        a1 = np.minimum(self.c0 + np.array(self.shape), G)
        if np.all(a1 > a0):
            d0, d1 = a0 - self.c0, a1 - self.c0
            out[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = self._afull[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
        return out

    def phi(self, F):
        """fibre fraction of the free volume of each cell"""
        return F / (self.free() * self.dtype(self.vox))

    def adjoint(self, G, want_v=False):
        """gradient with respect to the sample positions (and volumes) of a scalar E of the field F, given G = dE/dF
        on the crop of the last field(): the blur is its own adjoint, then each sample gathers the blurred G with the
        derivatives of its trilinear weights"""
        b, t, v, sx, sy = self._st
        H = blur121(np.asarray(G, self.dtype)).ravel()
        hv, ht = _gather(H, b, sx, sy, t, True)
        gY = v[:, None] * ht / self.h
        if want_v:
            return gY, np.asarray(hv, np.float64)
        return gY

    def pressure(self, Y, v, c, free_weight=True):
        """the hair-hair energy (J) of samples Y with fibre volumes v: E = sum_v (a_v h^3) c phi_v^3, the free energy
        density c phi^3 integrated over the volume hair can fill (P = 2 c phi^3 = k E phi^3 with c = k E / 2).
        free_weight False weighs every cell by h^3 (DESIGN_v2 4.5 as written), which adds a push away from the body
        of 1 / a_v on top of the free-volume fraction. Returns E, dE/dY (n, 3), phi on the crop."""
        F = self.field(Y, v)
        a = self.free()
        av = a * self.dtype(self.vox)
        ph = F / av
        p2 = ph * ph
        if free_weight:
            E = float(c) * float(np.sum(av * p2 * ph, dtype=np.float64))
            G = self.dtype(3 * c) * p2
        else:
            E = float(c) * self.vox * float(np.sum(p2 * ph, dtype=np.float64))
            G = self.dtype(3 * c * self.vox) * p2 / av
        return E, self.adjoint(G), ph
