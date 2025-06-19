#!/usr/bin/env python3
"""
LangSAMラッパーメインスクリプト

LangSAMノードとOptFlowノードをマルチスレッドで同時実行する。
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .lang_sam_node import LangSAMNode
from .optflow_node import OptFlowNode


def main(args=None):
    """メイン関数：LangSAMノードとOptFlowノードを同時実行"""
    rclpy.init(args=args)

    # ノードの初期化
    nodes = [LangSAMNode(), OptFlowNode()]

    # マルチスレッドエグゼキュータに登録
    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # ノードの破棄
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
