# Detection results message for LangSAM system
std_msgs/Header header
int32 num_detections
string model_used
geometry_msgs/Polygon[] boxes
string[] labels
float64[] probabilities
sensor_msgs/Image[] masks