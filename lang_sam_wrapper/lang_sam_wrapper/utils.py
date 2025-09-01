"""ROS2関係の処理 - lang_sam_wrapperパッケージ用ユーティリティ"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.publisher import Publisher


class ParameterManager(ABC):
    """ROS2パラメータ管理基底クラス"""
    
    def __init__(self, node: Node):
        self.node = node
        
    def declare_parameter_with_default(self, name: str, default_value: Any) -> None:
        """型安全なパラメータ宣言"""
        try:
            self.node.declare_parameter(name, default_value)
        except Exception as e:
            self.node.get_logger().warn(f"Parameter declaration failed for '{name}': {e}")
    
    def get_parameter_safe(self, name: str, default_value: Any = None) -> Any:
        """安全なパラメータ取得（デフォルト値フォールバック付き）"""
        try:
            return self.node.get_parameter(name).value
        except Exception:
            self.node.get_logger().warn(f"Failed to get parameter '{name}', using default: {default_value}")
            return default_value
    
    def batch_declare_parameters(self, param_definitions: Dict[str, Dict[str, Any]]) -> None:
        """バッチパラメータ宣言"""
        for name, config in param_definitions.items():
            self.declare_parameter_with_default(name, config.get('default'))
    
    def batch_load_parameters(self, param_names: List[str]) -> Dict[str, Any]:
        """バッチパラメータ読み込み"""
        return {name: self.get_parameter_safe(name) for name in param_names}
    
    @abstractmethod
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """パラメータ定義を返す（継承クラスで実装）"""
        pass
    
    def initialize_parameters(self) -> Dict[str, Any]:
        """パラメータ初期化統一メソッド"""
        definitions = self.get_parameter_definitions()
        self.batch_declare_parameters(definitions)
        return self.batch_load_parameters(list(definitions.keys()))


class TrackerParameterManager(ParameterManager):
    """LangSAMTrackerNode用パラメータマネージャ"""
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            'sam_model': {'default': 'sam2.1_hiera_small'},
            'text_prompt': {'default': 'white line. red pylon. human. car.'},
            'box_threshold': {'default': 0.3},
            'text_threshold': {'default': 0.2},
            'tracking_targets': {'default': ['white line', 'red pylon', 'human', 'car']},
            'gdino_interval_seconds': {'default': 1.0},
            'input_topic': {'default': '/zed/zed_node/rgb/image_rect_color'},
            'gdino_topic': {'default': '/image_gdino'},
            'csrt_output_topic': {'default': '/image_csrt'},
            'sam_topic': {'default': '/image_sam'},
            'bbox_margin': {'default': 5},
            'bbox_min_size': {'default': 3},
        }


class MultiViewParameterManager(ParameterManager):
    """MultiViewNode用パラメータマネージャ"""
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            'upper_left_topic': {'default': '/zed/zed_node/rgb/image_rect_color'},
            'upper_right_topic': {'default': '/image_gdino'},
            'lower_left_topic': {'default': '/image_csrt'},
            'lower_right_topic': {'default': '/image_sam'},
            'output_fps': {'default': 30.0},
            'output_width': {'default': 1280},
            'output_height': {'default': 720},
            'border_width': {'default': 2},
            'border_color': {'default': [255, 255, 255]},
            'enable_labels': {'default': True},
            'label_font_scale': {'default': 0.7},
            'label_thickness': {'default': 2},
            'label_color': {'default': [0, 255, 0]},
        }


class ImagePublisher:
    """ROS2画像配信ユーティリティ"""
    
    def __init__(self, node: Node):
        self.node = node
        self.bridge = CvBridge()
    
    def publish_image(self, publisher: Publisher, image: np.ndarray) -> bool:
        """基本画像配信"""
        try:
            if image is None or image.size == 0:
                return False
            
            msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            publisher.publish(msg)
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"画像配信エラー: {e}")
            return False
    
    def publish_with_fallback(self, publisher: Publisher, primary_image: np.ndarray, 
                             fallback_image: Optional[np.ndarray] = None) -> bool:
        """フォールバック付き画像配信"""
        if self.publish_image(publisher, primary_image):
            return True
        
        if fallback_image is not None:
            return self.publish_image(publisher, fallback_image)
        
        return False
    
    def ros_image_to_numpy(self, ros_msg: Image) -> Optional[np.ndarray]:
        """安全なROS Image → NumPy変換"""
        try:
            return self.bridge.imgmsg_to_cv2(ros_msg, 'bgr8')
        except Exception as e:
            self.node.get_logger().error(f"画像変換エラー: {e}")
            return None


class FrequencyMonitor:
    """周波数監視ユーティリティ"""
    
    def __init__(self, window_size: int = 300):
        from collections import deque
        import time
        self.timestamps = deque(maxlen=window_size)
        self.frequency = 0.0
        self.time = time
    
    def update(self) -> float:
        """周波数更新・計算"""
        current_time = self.time.time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) >= 2:
            time_span = current_time - self.timestamps[0]
            if time_span > 0:
                self.frequency = (len(self.timestamps) - 1) / time_span
        
        return self.frequency
    


class ImageValidator:
    """画像データ検証ユーティリティ"""
    
    @staticmethod
    def validate_image_data(image: Optional[np.ndarray]) -> bool:
        """画像データ検証"""
        return (
            image is not None 
            and isinstance(image, np.ndarray) 
            and image.size > 0 
            and len(image.shape) >= 2
        )
    
    @staticmethod
    def ensure_bgr_format(image: np.ndarray) -> np.ndarray:
        """BGR形式確保（グレースケール→BGR変換含む）"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    @staticmethod
    def safe_resize(image: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
        """安全なリサイズ（黒画像フォールバック付き）"""
        if not ImageValidator.validate_image_data(image):
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        return cv2.resize(image, (width, height))


