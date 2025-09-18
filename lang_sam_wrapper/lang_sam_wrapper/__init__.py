# lang_sam_ros2 パッケージ初期化
try:
    from .lang_sam_tracker_node import LangSAMTrackerNode
except ImportError:
    # トラッカーが利用できない場合のフォールバック
    LangSAMTrackerNode = None

try:
    from .lang_sam_tracker_node_legacy import LangSAMTrackerNodeLegacy
except ImportError:
    # レガシートラッカーが利用できない場合のフォールバック
    LangSAMTrackerNodeLegacy = None

# マルチビューノードはlang_sam_executorパッケージのC++実装に移動
from .debug_lang_segment_anything import LangSAMNode

# C++ CSRTモジュールのインポートを許可
try:
    from .csrt_native import *
except ImportError:
    # C++拡張が利用できない場合のフォールバック
    pass
