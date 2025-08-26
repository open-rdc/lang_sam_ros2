#ifndef LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_
#define LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <lang_sam_msgs/msg/detection_result.hpp>
#include "lang_sam_nav/lane_pixel_finder.hpp"

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

  // YOLOP Nav互換の処理関数
  std::vector<cv::Mat> extract_white_line_masks(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg);
  cv::Mat combine_masks(const std::vector<cv::Mat>& masks);
  cv::Mat skeletonize(const cv::Mat& mask);
  cv::Mat denoise(const cv::Mat& image);
  cv::Mat filterHorizontalLines(const cv::Mat& image);
  std::pair<cv::Vec4f, cv::Vec4f> fitLaneLines(const std::vector<cv::Point>& left_pixels, 
                                                const std::vector<cv::Point>& right_pixels);
  cv::Point2f calculateIntersection(const cv::Vec4f& left_line, const cv::Vec4f& right_line);
  double calculateControl(const cv::Point2f& intersection);
  
  // 可視化
  void publishVisualization(const cv::Mat& base_image,
                           const std_msgs::msg::Header& header,
                           const std::vector<cv::Point>& left_pixels,
                           const std::vector<cv::Point>& right_pixels,
                           const std::vector<cv::Point>& center_pixels,
                           const cv::Vec4f& left_line,
                           const cv::Vec4f& right_line,
                           const cv::Point2f& intersection);

  // ROS2 パブリッシャー・サブスクライバー
  rclcpp::Subscription<lang_sam_msgs::msg::DetectionResult>::SharedPtr detection_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr original_image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_pub_;

  // パラメータ（YOLOP Nav互換）
  double linear_velocity_;
  double kp_;  // P制御のみ使用
  double max_angular_velocity_;
  std::string original_image_topic_;
  int pixel_tolerance_;
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
  
  // LanePixelFinder
  std::unique_ptr<LanePixelFinder> pixel_finder_;
  
  // CV Bridge
  cv_bridge::CvImagePtr cv_bridge_ptr_;
};

} // namespace lang_sam_nav

#endif // LANG_SAM_NAV__LANE_FOLLOWING_NODE_HPP_