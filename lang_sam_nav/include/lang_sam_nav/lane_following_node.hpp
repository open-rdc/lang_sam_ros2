#ifndef LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_
#define LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <lang_sam_msgs/msg/detection_result.hpp>

#include <memory>
#include <vector>
#include <chrono>

namespace lang_sam_nav
{

class LaneFollowingNode : public rclcpp::Node
{
public:
  explicit LaneFollowingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // コールバック関数
  void detection_callback(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg);
  void original_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  // 処理関数
  std::vector<cv::Mat> extract_white_line_masks(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg);
  cv::Mat combine_masks(const std::vector<cv::Mat>& masks);
  void detectLanePixels(const cv::Mat& mask, std::vector<cv::Point>& left_pixels, std::vector<cv::Point>& right_pixels);
  cv::Vec4f fitLineFromPixels(const std::vector<cv::Point>& pixels);
  cv::Point2f calculateIntersection(const cv::Vec4f& left_line, const cv::Vec4f& right_line);
  double calculateControl(const cv::Point2f& intersection);
  void publishControl();
  
  // 可視化
  void publishVisualization(const std_msgs::msg::Header& header,
                           const std::vector<cv::Vec4f>& lane_lines,
                           const cv::Point2f& intersection);

  // ROS2 パブリッシャー・サブスクライバー
  rclcpp::Subscription<lang_sam_msgs::msg::DetectionResult>::SharedPtr detection_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr original_image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_pub_;

  // パラメータ
  double linear_velocity_;
  double kp_;
  double max_angular_velocity_;
  std::string original_image_topic_;
  bool enable_visualization_;
  int line_thickness_;
  int circle_radius_;

  // 状態変数
  cv::Mat latest_original_image_;
  cv::Vec4f last_left_line_;
  cv::Vec4f last_right_line_;
  cv::Point2f last_intersection_;
  bool has_valid_lines_;
  int image_width_;
  int image_height_;


  // CV Bridge
  cv_bridge::CvImagePtr cv_bridge_ptr_;
};

} // namespace lang_sam_nav

#endif // LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_