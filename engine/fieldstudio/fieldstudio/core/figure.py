"""hand2 (2026-11-07): File > New figure: a standing full-body female start at figure scale (1680 mm, feet on y = 0,
facing +z, her right on -x), built from primitives as separate objects on a 1 mm sculpt lattice, to be shaped with
the deform brushes. Paired parts (upper arms, forearms, hands, thighs, shins, feet) are one object per pair, both
sides placed symmetrically about x = 0, so Mirror X (which mirrors in the object's own frame) mirrors her left onto
her right and never drops a phantom dab into empty space beside a one-sided limb.

Proportions are a blockout (E: about 7.3 heads; head 230 mm, shoulders 380 mm, hips 340 mm), not a measured body:
the sculptor sets them against the reference planes.
"""
import copy

H_FIG = 1.68
LATTICE = 0.001
ALBEDO = [0.66, 0.56, 0.49]

# (name, kind, params, centre [x, y, z] for a midline part / the right-side part of a pair, paired)
PARTS = [
    ('Head', 'ellipsoid', {'radii': [0.075, 0.112, 0.095]}, [0.0, 1.565, 0.0], False),
    ('Neck', 'cylinder', {'radius': 0.052, 'height': 0.14, 'round': 0.0}, [0.0, 1.43, -0.01], False),
    ('Torso', 'ellipsoid', {'radii': [0.165, 0.27, 0.11]}, [0.0, 1.17, 0.0], False),
    ('Pelvis', 'ellipsoid', {'radii': [0.17, 0.13, 0.115]}, [0.0, 0.91, -0.005], False),
    ('Upper arms', 'capsule', {'radius': 0.042, 'height': 0.22}, [0.205, 1.215, -0.01], True),
    ('Forearms', 'capsule', {'radius': 0.034, 'height': 0.19}, [0.215, 0.935, 0.0], True),
    ('Hands', 'ellipsoid', {'radii': [0.022, 0.085, 0.045]}, [0.222, 0.735, 0.01], True),
    ('Thighs', 'capsule', {'radius': 0.075, 'height': 0.30}, [0.092, 0.625, 0.0], True),
    ('Shins', 'capsule', {'radius': 0.05, 'height': 0.32}, [0.088, 0.27, -0.01], True),
    ('Feet', 'ellipsoid', {'radii': [0.042, 0.035, 0.115]}, [0.088, 0.035, 0.045], True),
]


def figure_specs(height=H_FIG, h=LATTICE, first_id=1000):
    """-> list of AddObject specs (one per part). Node ids are fixed per spec (deterministic replay)."""
    k = float(height) / H_FIG
    out = []
    nid = int(first_id)
    for name, kind, params, c, paired in PARTS:
        p = copy.deepcopy(params)
        for key, v in list(p.items()):
            if key in ('radius', 'height'):
                p[key] = float(v) * k
            elif key == 'radii':
                p[key] = [float(x) * k for x in v]
        c = [float(x) * k for x in c]
        spec = {'name': name, 'h': float(h), 'colour': {'albedo': list(ALBEDO), 'variation': 0.0}}
        if not paired:
            spec.update(kind=kind, params=p, translate=c)
        else:
            kids = []
            for sx in (1.0, -1.0):
                prim = {'id': nid + 2, 'kind': kind, 'name': None, 'params': p, 'children': []}
                kids.append({'id': nid + 1, 'kind': 'transform', 'name': 'left' if sx > 0 else 'right',
                             'params': {'translate': [sx * c[0], 0.0, 0.0], 'rotate': [0.0, 0.0, 0.0],
                                        'scale': [1.0, 1.0, 1.0]}, 'children': [prim]})
                nid += 2
            spec.update(base={'id': nid + 1, 'kind': 'union', 'name': name, 'params': {'blend': 0.0},
                              'children': kids}, translate=[0.0, c[1], c[2]])
            nid += 1
        nid += 3
        out.append(spec)
    return out
