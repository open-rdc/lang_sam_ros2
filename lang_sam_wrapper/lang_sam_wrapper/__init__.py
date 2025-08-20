# lang_sam_ros2 package initialization
try:
    from .lang_sam_tracker_node import LangSAMTrackerNode
except ImportError:
    # Fallback if tracker not available
    LangSAMTrackerNode = None

try:
    from .lang_sam_tracker_node_legacy import LangSAMTrackerNodeLegacy
except ImportError:
    # Fallback if legacy tracker not available
    LangSAMTrackerNodeLegacy = None

from .multi_view_node import MultiViewNode
from .debug_lang_segment_anything import LangSAMNode

# Allow importing the C++ CSRT module
try:
    from .csrt_native import *
except ImportError:
    # Fallback if C++ extension not available
    pass
