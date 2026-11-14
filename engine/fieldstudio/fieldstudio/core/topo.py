"""fs-brush 11-11 (design D4: a stroke breaks nothing that existed before it): the topology of a set of lattice nodes
(inside = F < 0) as the brushes' keep-apart guard sees it: inside 26-connected, outside 6-connected (each inside node a
closed unit cube; cubes touching at a corner join).
  b0 = inside parts, b2 = enclosed cavities (outside parts not reaching the box's border), b1 = tunnels / handles
  (through openings: a pierced ear, a through nostril), from the Euler characteristic of the union of closed cubes:
  chi = V - E + F - C = b0 - b1 + b2.
Over a box holding a stroke's whole footprint (its border nodes unchanged) the counts are local: a fusion of touching
parts, a sealed opening, a capped pit or a fold-over pocket each changes one of them; a blind pit filled solid changes
none. Self-test: python -m fieldstudio.core.topo (ball, sealed shell, torus, pierced ball, blind pit, two parts)."""
import numpy as np


def _any_shift(a, axes):
    """OR of a over all 0/1 shifts along the given axes (a padded by one on each side first)"""
    out = a
    for ax in axes:
        out = out[(slice(None),) * ax + (slice(1, None),)] | out[(slice(None),) * ax + (slice(None, -1),)]
    return out


def euler(ins):
    """chi of the union of closed unit cubes at the True voxels"""
    a = np.pad(np.asarray(ins, bool), 1)
    C = int(a.sum())
    F = sum(int(_any_shift(a, (k,)).sum()) for k in range(3))
    E = sum(int(_any_shift(a, tuple(j for j in range(3) if j != k)).sum()) for k in range(3))
    V = int(_any_shift(a, (0, 1, 2)).sum())
    return V - E + F - C


def betti(ins):
    """-> {'b0', 'b1', 'b2', 'chi'} of a boolean node grid (True = inside)"""
    from scipy import ndimage as ndi
    ins = np.asarray(ins, bool)
    _, b0 = ndi.label(ins, structure=np.ones((3, 3, 3)))
    _, no = ndi.label(~np.pad(ins, 1))             # 6-connected outside; the padding is the box's border
    b2 = int(no - 1) if no else 0
    chi = euler(ins)
    return {'b0': int(b0), 'b1': int(b0 + b2 - chi), 'b2': b2, 'chi': int(chi)}


def cavities(ins, h, origin):
    """the enclosed outside parts (b2) with their places: [{'centroid_m', 'volume_m3', 'bbox_m': (lo, hi), 'nodes'}];
    h = the node spacing (m), origin = the world position of node (0, 0, 0)"""
    from scipy import ndimage as ndi
    ins = np.asarray(ins, bool)
    lab, no = ndi.label(~np.pad(ins, 1))
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))) - {0}
    org = np.asarray(origin, float)
    out = []
    for k in range(1, no + 1):
        if k in border:
            continue
        w = np.argwhere(lab == k) - 1                # back to the unpadded grid
        P = org + w * float(h)
        out.append({'centroid_m': P.mean(0).tolist(), 'volume_m3': float(len(w)) * float(h) ** 3,
                    'bbox_m': (P.min(0).tolist(), P.max(0).tolist()), 'nodes': int(len(w))})
    return out


def node_grid(scene, oid, lo, hi):
    """the object's lattice nodes (local multiples of its sculpt h; translation-only transforms) inside the world box
    lo..hi (m): -> (inside bool grid, h_world, origin_world)"""
    from ..tools.protected_merge import field_of
    o = scene.get(oid)
    s = float(o.transform.get('scale', 1.0)) if isinstance(o.transform.get('scale', 1.0), (int, float)) else 1.0
    if abs(s - 1.0) > 1e-9 or np.any(np.asarray(o.transform.get('rotate', (0, 0, 0)), float)):
        raise ValueError('node_grid: only translated objects (rotate 0, scale 1) are read on their lattice')
    hm = float(o.sculpt.h)
    T0 = np.asarray(o.transform.get('translate', (0, 0, 0)), float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    I0 = np.floor((lo - T0) / hm).astype(int)
    I1 = np.ceil((hi - T0) / hm).astype(int)
    ax = [np.arange(I0[a], I1[a] + 1) * hm + T0[a] for a in range(3)]
    ins = np.zeros([len(a) for a in ax], bool)
    Y, Z = np.meshgrid(ax[1], ax[2], indexing='ij')
    pk = scene.pack()
    for i0 in range(0, len(ax[0]), 16):
        xs = ax[0][i0:i0 + 16]
        P = np.stack([np.repeat(xs, Y.size), np.tile(Y.ravel(), len(xs)), np.tile(Z.ravel(), len(xs))], 1)
        ins[i0:i0 + 16] = (field_of(scene, pk, o, P) < 0).reshape(len(xs), len(ax[1]), len(ax[2]))
    return ins, hm, np.array([ax[0][0], ax[1][0], ax[2][0]])


def object_topology(scene, oid, lo, hi):
    """-> betti + {'cavities': [...]} of the object's lattice nodes over the world box lo..hi (m). CPU; seconds for a
    whole head at 1 mm (34 M nodes ~ 20 s), so for an on-request check, not the stroke path."""
    ins, h, org = node_grid(scene, oid, lo, hi)
    b = betti(ins)
    b['cavities'] = cavities(ins, h, org)
    b['nodes'] = int(ins.size)
    return b


def _selftest():
    n = 40
    I, J, K = np.meshgrid(*[np.arange(n) - n / 2 + 0.5] * 3, indexing='ij')
    r = np.sqrt(I ** 2 + J ** 2 + K ** 2)
    ball = r < 12
    shell = (r < 12) & (r > 7)
    torus = (np.sqrt(I ** 2 + J ** 2) - 10) ** 2 + K ** 2 < 16
    pierced = ball & ~((np.sqrt(I ** 2 + K ** 2) < 2.5))
    pit = ball & ~((np.sqrt(I ** 2 + K ** 2) < 2.5) & (J > 4))
    two = (np.sqrt((I - 6) ** 2 + J ** 2 + K ** 2) < 5) | (np.sqrt((I + 6) ** 2 + J ** 2 + K ** 2) < 5)
    ok = True
    for name, s, want in (('ball', ball, (1, 0, 0)), ('shell', shell, (1, 0, 1)), ('torus', torus, (1, 1, 0)),
                          ('pierced', pierced, (1, 1, 0)), ('pit', pit, (1, 0, 0)), ('two', two, (2, 0, 0))):
        b = betti(s)
        got = (b['b0'], b['b1'], b['b2'])
        ok &= got == want
        print('%-8s %s %s' % (name, got, 'ok' if got == want else 'WANT %s' % (want,)))
    cv = cavities(shell, 0.001, (0, 0, 0))
    print('shell cavity', len(cv), 'nodes', cv[0]['nodes'] if cv else None)
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
