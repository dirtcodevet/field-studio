"""Field Studio core: scene (objects and layers), ops (base-tree nodes), brush (sculpt layer), history."""
from .scene import Scene, SceneObject, SculptGrid, DEFAULT_H, make_object
from .ops import Node, compile_tree, PRIMITIVES, BOOLEANS, MUTATIONS
from .history import (History, Command, AddObject, RemoveObject, SetBase, SetNodeParams, SetObjectProps,
                      BrushStroke, replay, save_project, load_project)
from . import brush
from .combine import CombineObjects, DuplicateObject, MirrorDuplicate, MaskEdit   # hand2f: registers them
from .attach import FuseRoot                                                       # 11-09: registers it
