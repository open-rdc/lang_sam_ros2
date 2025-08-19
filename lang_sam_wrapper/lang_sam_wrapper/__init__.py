# lang_sam_ros2 package initialization
try:
    from .lang_sam_tracker_node_native import LangSAMTrackerNodeNative
except ImportError:
    # Fallback if native tracker not available
    LangSAMTrackerNodeNative = None

from .multi_view_node import MultiViewNode
from .debug_lang_segment_anything import LangSAMNode

# Allow importing the native C++ CSRT module
try:
    from .csrt_native import *
except ImportError:
    # Fallback if C++ extension not available
    pass
