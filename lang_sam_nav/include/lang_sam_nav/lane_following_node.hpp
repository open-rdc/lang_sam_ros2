#ifndef LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_
#define LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <lang_sam_msgs/msg/detection_result.hpp>

#include <memory>
#include <vector>
#include <chrono>

namespace lang_sam_nav
{

// Lane pixel structure similar to reference code
struct LaneLines {
  std::vector<cv::Point> left_pixels;
  std::vector<cv::Point> right_pixels;
  std::vector<cv::Point> center_pixels;
};

class LaneFollowingNode : public rclcpp::Node
{
public:
  explicit LaneFollowingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // Callback functions
  void detection_callback(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg);
  void detection_fallback_callback(const std_msgs::msg::String::SharedPtr msg);
  void original_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void control_timer_callback();

  // Core processing functions
  std::vector<cv::Mat> extract_white_line_masks(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg);
  
  // New pixel-based approach functions
  cv::Mat preprocess_mask_with_skeleton(const cv::Mat& combined_mask);
  LaneLines extract_lane_pixels_bottom_up(const cv::Mat& skeleton_mask);
  std::pair<cv::Vec4f, cv::Vec4f> fit_lane_lines(const LaneLines& lane_pixels);
  cv::Point2f calculate_intersection_from_fitted_lines(const cv::Vec4f& left_line, const cv::Vec4f& right_line);
  
  // Existing control functions
  double calculate_centering_control(const cv::Point2f& intersection);
  double calculate_normalized_error_control(const cv::Point2f& intersection);
  
  // Legacy functions (kept for compatibility)
  std::pair<cv::Mat, cv::Mat> split_masks_left_right(const std::vector<cv::Mat>& masks);
  std::pair<std::vector<cv::Point>, std::vector<cv::Point>> find_lane_boundaries_bottom_up(const cv::Mat& combined_mask);
  std::vector<cv::Vec2f> detect_lane_lines_separated(const cv::Mat& left_mask, const cv::Mat& right_mask);
  std::vector<cv::Vec2f> filter_horizontal_lines(const std::vector<cv::Vec2f>& lines);
  cv::Vec2f select_closest_line_to_center(const std::vector<cv::Vec2f>& lines, bool is_left_region);
  cv::Point2f calculate_intersection(const cv::Vec2f& line1, const cv::Vec2f& line2);
  float calculate_x_at_y(const cv::Vec2f& line, float y);
  
  // Utility functions
  void publish_visualization_image(const cv::Mat& base_image, 
                                 const std_msgs::msg::Header& header,
                                 const std::vector<cv::Mat>& white_line_masks,
                                 const LaneLines& lane_pixels,
                                 const cv::Vec4f& left_line,
                                 const cv::Vec4f& right_line,
                                 const cv::Point2f& intersection);

  // ROS2 Publishers and Subscribers
  rclcpp::Subscription<lang_sam_msgs::msg::DetectionResult>::SharedPtr detection_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr detection_fallback_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr original_image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Parameters
  double linear_velocity_;
  double kp_, ki_, kd_;
  std::string original_image_topic_;
  
  // Hough transform parameters
  int hough_threshold_;
  double hough_rho_;
  double hough_theta_;
  int hough_min_line_length_;
  int hough_max_line_gap_;
  
  // Lane detection parameters
  double lane_split_ratio_;
  int roi_top_offset_;
  int roi_bottom_offset_;
  
  
  // Visualization parameters
  bool enable_visualization_;
  int line_thickness_;
  int circle_radius_;

  // State variables
  cv::Mat latest_original_image_;
  std::chrono::time_point<std::chrono::steady_clock> last_time_;
  double last_error_;
  double error_sum_;
  int image_width_, image_height_;
  
  // Lane detection continuity state
  cv::Vec2f last_left_line_;
  cv::Vec2f last_right_line_;
  cv::Point2f last_intersection_;
  bool has_valid_lines_;
  
  // Pixel-based lane tracking
  LaneLines last_lane_pixels_;
  cv::Vec4f last_fitted_left_line_;
  cv::Vec4f last_fitted_right_line_;
  
  // Pixel detection parameters (ROS configurable)
  int pixel_search_tolerance_;
  bool use_skeleton_;
  int min_pixels_for_line_fit_;
  int morph_kernel_size_;
  int gaussian_blur_size_;
  bool enable_adaptive_tolerance_;
  
  // CV Bridge
  cv_bridge::CvImagePtr cv_bridge_ptr_;
};

} // namespace lang_sam_nav

#endif // LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_