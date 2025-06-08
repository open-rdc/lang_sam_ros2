import rclpy
from rclpy.executors import MultiThreadedExecutor

from lang_sam_ros2.lang_segment_anything import LangSAMNode
from lang_sam_ros2.lang_segment_anything_mask import LangSAMNode
from lang_sam_ros2.lang_segment_anything_optflow import OptFlowMaskNode


def main(args=None):
    rclpy.init(args=args)

    # ノードの初期化
    lang_sam_node = LangSAMNode()
    lang_sam_mask_node = LangSAMNode()
    optflow_node = OptFlowMaskNode()

    # マルチスレッドエグゼキュータに登録
    executor = MultiThreadedExecutor()
    executor.add_node(lang_sam_node)
    executor.add_node(lang_sam_mask_node)
    executor.add_node(optflow_node)

    try:
        executor.spin()
    finally:
        lang_sam_node.destroy_node()
        lang_sam_mask_node.destroy_node()
        optflow_node.destroy_node()
        rclpy.shutdown()
