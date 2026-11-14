"""The approved Field Studio toolset (fs-tools, 11-14; engine/fieldstudio/design/TOOLSET_DESIGN_2026-11-13.md).

Importing this package registers its commands with core.history (so logs that hold them replay). The window reads the
tools from tools/registry.py and posts their jobs (toolset/jobs.py).

    cut.py       Cut: a line cut with a Length (through = two objects, exact), a shape cut into the container
    hollow.py    Hollow / Shell inward (the outside unchanged), Re-hollow
    copies.py    Mirror (exact, world or object planes), Array (line, radial, path; Vary by seed)
    connect.py   Connect: a connector morphing outline to outline, cut flush by both objects' fields
    moves.py     Transform of a region (soft edge, its sculpting carried), Bend / Twist / Taper of the whole object
    place.py     Place on surface: contact placement (the physics settle: a hook)
    mask.py      Mask by thickness / distance; Grow, Shrink, Soften, Invert, Clear
    drag.py      a 2-D drag to a 3-D move (the guide plane)
    jobs.py      the engine jobs (register(engine)) the window posts
    merge.py     Merge (an exact join; blend at contact, contact-following width), Group, Ungroup
    contact.py   contact between objects from their exact fields
    common.py    frames, boxes, lattice nodes, brick writes with exact undo
"""
from .cut import LineCut, CutAway, CutWithObject
from .hollow import HollowObject
from .copies import MirrorAcross, ArrayCopies
from .connect import ConnectObjects
from .moves import ClosedWarp, region_grid
from .place import Placer
from .mask import MaskBy, MaskOp
from .merge import JoinObjects, GroupObjects, UngroupObjects, NotTouching, group_members
