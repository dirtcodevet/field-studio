"""fs-materials (2026-11-14): the materials library made real in Field Studio.

A shape picks a library material and IS that material: its look (colour from a volumetric composition field, roughness
and Fresnel from the profile's finish), its physics (density, friction, porosity, brittleness from the profile), and its
seeded variation as structure in the field (relief, chips, cracks: core/program.py OP_MATGEN, materials/fieldops.py).

Modules:
  fieldops   the field generators' evaluator (numba) and parameter blocks; fieldops.cuh is the card's copy
  library    the profiles (engine/materials/profiles) as the panel reads them: families, cards, finishes, physics
  generate   seeded generators: stone surface, chips on any form's edges, cracks; Vary across copies
  commands   the undoable edits (assign a material, add cracks, vary copies) built from the history's commands
  look       the card look: the per-volume material table and the material shading hook for the tracer
Licence: Apache-2.0 (LICENSE, NOTICE). Subject-free: no person's data here.
"""
