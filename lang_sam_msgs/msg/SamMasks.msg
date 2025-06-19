# SAM segmentation masks with labels
std_msgs/Header header
string[] labels
sensor_msgs/Image[] masks
float64[] boxes  # x1, y1, x2, y2 for each detection (4 values per detection)
float64[] probs  # confidence scores for each detection