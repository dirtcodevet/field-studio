"""Hair Studio core: a numpy-only strand generator driven by a STYLE file (Apache-2.0).

    style.py     the STYLE schema (defaults, ranges, notes), presets that extend one another, validation
    field_collider.py  the body and the scalp as fields: distance, push-out, report, the body lattice, the roots
    ram.py       this process's peak RAM (mesh-free since 11-14: the triangle helper mesh.py is archived)
    scalp.py     the scalp: spherical frame about a head centre, the hairline boundary theta_b(phi)
    flow.py      the comb: strands walked over the scalp along a direction field and hung under gravity, with
                 collision; gathering (ponytail / bun) after a tie
    form.py      layers, locks / sub-locks / bundles, clumping, waves, crown lift, wander, frizz, fly-aways
    colour.py    the fibre colour: eumelanin / pheomelanin / a neutral (ash) absorber -> absorption sigma_a, Chiang's
                 reflectance map, per-point colour
    generate.py  scalp + style -> strands (the npz format below), and the collision report
    preview.py   a free CPU coverage splat (numba) of strands through a pinhole camera over a z-buffer
"""
