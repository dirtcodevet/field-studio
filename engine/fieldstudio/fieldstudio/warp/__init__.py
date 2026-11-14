"""The one warp engine (fs-tools, 11-14; TOOLSET_DESIGN section 5 item 3, the approved design).

Moves are smooth one-to-one warps: flows of velocity fields, anchored or along a path, with an optional volume-keeping
(divergence-free) form, gated per point by contact. One core, built once, serving Shape, Transform (objects, masses,
regions), Place on surface, Draw out and, later, posing.

    profile.py   the shared profile: Radius, Hardness (the peak), Smoothing (the foot), the falloff choice
    flow.py      the kernels: velocity, RK4 forward / inverse maps, Jacobian (numba, deterministic)
    lens.py      what a lens sees: the depth images a flat tool takes at the press (Reach: Near side / Whole part)
    build.py     the warps: anchored, affine / rigid (a region), along a path; supports; the swept box
    bake.py      a warp applied to an object: F'(x) = F(phi^-1(x)) written into its sculpt lattice (a Command)

Patent routes (LIBRARIES_AND_PATENTS 6.2): the volume-keeping form is a vector-potential flow (von Funck et al. 2006),
never a Kelvinlet; the field is read back through the inverse map and baked into the lattice when the move ends (no
parametric surface, no long stored chain of maps).
"""
from .profile import weight, profile_curve, FALLOFF, PEAK_SHAPE, PEAK_SWELL
from .build import Support, Warp, anchored, affine, rigid, along_path
from .lens import depth_images, lens_frame, march
