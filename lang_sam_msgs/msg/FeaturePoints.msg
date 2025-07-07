std_msgs/Header header
string[] labels                     # オブジェクトラベル
geometry_msgs/Point32[] points     # 全ての特徴点（フラット化）
uint32[] point_counts              # 各ラベルの点数
float64[] boxes                    # バウンディングボックス (x1, y1, x2, y2 for each detection)
float64[] probs                    # 各検出の確率
sensor_msgs/Image source_image     # 元画像