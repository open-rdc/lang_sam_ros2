"""トラッキング設定管理"""


class TrackingConfig:
    """トラッキング設定管理"""
    
    def __init__(self, bbox_margin: int = 5, bbox_min_size: int = 20, 
                 tracker_min_size: int = 10):
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        self.tracker_min_size = tracker_min_size
    
    def update(self, **kwargs) -> None:
        """設定更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)