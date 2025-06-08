import rclpy
from rclpy.executors import MultiThreadedExecutor

# LangSAMノードとOptFlowノードをインポート
from lang_sam_ros2.lang_sam_node import LangSAMNode
from lang_sam_ros2.optflow_node import OptFlowNode


def main(args=None):
    rclpy.init(args=args)

    # ノードの初期化
    lang_sam_node = LangSAMNode()
    optflow_node = OptFlowNode()

    # マルチスレッドエグゼキュータに登録
    executor = MultiThreadedExecutor()
    executor.add_node(lang_sam_node)
    executor.add_node(optflow_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # ノードの破棄
        lang_sam_node.destroy_node()
        optflow_node.destroy_node()

    # rclpyのシャットダウン
    rclpy.shutdown()


if __name__ == '__main__':
    main()
