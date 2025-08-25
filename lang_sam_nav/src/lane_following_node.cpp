#include "lang_sam_nav/lane_following_node.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <regex>
#include <std_msgs/msg/string.hpp>

namespace lang_sam_nav
{

LaneFollowingNode::LaneFollowingNode(const rclcpp::NodeOptions & options)
: Node("lane_following_node", options),
  last_error_(0.0),
  error_sum_(0.0),
  image_width_(0),
  image_height_(0),
  last_left_line_(cv::Vec2f(0, 0)),
  last_right_line_(cv::Vec2f(0, 0)),
  last_intersection_(cv::Point2f(-1, -1)),
  has_valid_lines_(false)
{
  // Declare parameters - Control
  this->declare_parameter("linear_velocity", 1.0);
  this->declare_parameter("kp", 1.0);
  this->declare_parameter("ki", 0.1);
  this->declare_parameter("kd", 0.05);
  this->declare_parameter("original_image_topic", "/image_raw");
  
  // Declare parameters - Hough Transform
  this->declare_parameter("hough_threshold", 50);
  this->declare_parameter("hough_rho", 1.0);
  this->declare_parameter("hough_theta", 0.017453); // CV_PI/180
  this->declare_parameter("hough_min_line_length", 30);
  this->declare_parameter("hough_max_line_gap", 10);
  
  // Declare parameters - Lane Detection
  this->declare_parameter("lane_split_ratio", 0.5);
  this->declare_parameter("roi_top_offset", 50);
  this->declare_parameter("roi_bottom_offset", 0);
  
  
  // Declare parameters - Visualization
  this->declare_parameter("enable_visualization", true);
  this->declare_parameter("line_thickness", 3);
  this->declare_parameter("circle_radius", 10);
  
  // Declare parameters - Pixel-based Detection
  this->declare_parameter("pixel_search_tolerance", 100);
  this->declare_parameter("use_skeleton", false);
  this->declare_parameter("min_pixels_for_line_fit", 5);
  this->declare_parameter("morph_kernel_size", 3);
  this->declare_parameter("gaussian_blur_size", 5);
  this->declare_parameter("enable_adaptive_tolerance", true);

  // Get parameters - Control
  linear_velocity_ = this->get_parameter("linear_velocity").as_double();
  kp_ = this->get_parameter("kp").as_double();
  ki_ = this->get_parameter("ki").as_double();
  kd_ = this->get_parameter("kd").as_double();
  original_image_topic_ = this->get_parameter("original_image_topic").as_string();
  
  // Get parameters - Hough Transform
  hough_threshold_ = this->get_parameter("hough_threshold").as_int();
  hough_rho_ = this->get_parameter("hough_rho").as_double();
  hough_theta_ = this->get_parameter("hough_theta").as_double();
  hough_min_line_length_ = this->get_parameter("hough_min_line_length").as_int();
  hough_max_line_gap_ = this->get_parameter("hough_max_line_gap").as_int();
  
  // Get parameters - Lane Detection
  lane_split_ratio_ = this->get_parameter("lane_split_ratio").as_double();
  roi_top_offset_ = this->get_parameter("roi_top_offset").as_int();
  roi_bottom_offset_ = this->get_parameter("roi_bottom_offset").as_int();
  
  
  // Get parameters - Visualization
  enable_visualization_ = this->get_parameter("enable_visualization").as_bool();
  line_thickness_ = this->get_parameter("line_thickness").as_int();
  circle_radius_ = this->get_parameter("circle_radius").as_int();
  
  // Get parameters - Pixel-based Detection
  pixel_search_tolerance_ = this->get_parameter("pixel_search_tolerance").as_int();
  use_skeleton_ = this->get_parameter("use_skeleton").as_bool();
  min_pixels_for_line_fit_ = this->get_parameter("min_pixels_for_line_fit").as_int();
  morph_kernel_size_ = this->get_parameter("morph_kernel_size").as_int();
  gaussian_blur_size_ = this->get_parameter("gaussian_blur_size").as_int();
  enable_adaptive_tolerance_ = this->get_parameter("enable_adaptive_tolerance").as_bool();

  // Initialize CV Bridge
  cv_bridge_ptr_ = std::make_shared<cv_bridge::CvImage>();
  
  // Initialize tracking states
  has_valid_lines_ = false;

  // Create subscribers
  RCLCPP_INFO(this->get_logger(), "[INIT] Creating DetectionResult subscriber on /lang_sam_detections");
  detection_sub_ = this->create_subscription<lang_sam_msgs::msg::DetectionResult>(
    "/lang_sam_detections", 10,
    std::bind(&LaneFollowingNode::detection_callback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "[INIT] ✓ DetectionResult subscriber created successfully");

  // Create fallback subscriber for compatibility
  detection_fallback_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/lang_sam_detections_simple", 10,
    std::bind(&LaneFollowingNode::detection_fallback_callback, this, std::placeholders::_1));

  original_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    original_image_topic_, 10,
    std::bind(&LaneFollowingNode::original_image_callback, this, std::placeholders::_1));

  // Create publishers
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/lane_following_visualization", 10);

  // Create control timer (100Hz)
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(10),
    std::bind(&LaneFollowingNode::control_timer_callback, this));

  last_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(this->get_logger(), "Lane Following Node (C++) initialized");
}

void LaneFollowingNode::detection_callback(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Detection callback triggered with %zu labels and %zu masks", 
    msg->labels.size(), msg->masks.size());
  
  try {
    // Extract white line masks
    auto white_line_masks = extract_white_line_masks(msg);
    
    if (!white_line_masks.empty()) {
      // Combine all masks
      cv::Mat combined_mask = cv::Mat::zeros(image_height_, image_width_, CV_8UC1);
      for (const auto& mask : white_line_masks) {
        if (mask.size() == combined_mask.size()) {
          cv::bitwise_or(combined_mask, mask, combined_mask);
        }
      }
      
      // NEW: Preprocess with skeletonization
      cv::Mat skeleton_mask = preprocess_mask_with_skeleton(combined_mask);
      
      // NEW: Extract lane pixels using bottom-up scanning
      LaneLines lane_pixels = extract_lane_pixels_bottom_up(skeleton_mask);
      
      // NEW: Fit lines using cv::fitLine (parameterized minimum pixels)
      if (lane_pixels.left_pixels.size() >= static_cast<size_t>(min_pixels_for_line_fit_) && 
          lane_pixels.right_pixels.size() >= static_cast<size_t>(min_pixels_for_line_fit_)) {
        auto [left_line, right_line] = fit_lane_lines(lane_pixels);
        cv::Point2f intersection = calculate_intersection_from_fitted_lines(left_line, right_line);
        
        if (intersection.x >= 0 && intersection.y >= 0) {
          // Update stored results with new valid detection
          last_lane_pixels_ = lane_pixels;
          last_fitted_left_line_ = left_line;
          last_fitted_right_line_ = right_line;
          last_intersection_ = intersection;
          has_valid_lines_ = true;
          
          RCLCPP_INFO(this->get_logger(), "Pixel-based detection: %zu left, %zu right pixels, intersection at (%.1f, %.1f)", 
            lane_pixels.left_pixels.size(), lane_pixels.right_pixels.size(), 
            intersection.x, intersection.y);
        } else {
          RCLCPP_WARN(this->get_logger(), "Lines are parallel, keeping previous intersection");
        }
      } else {
        RCLCPP_WARN(this->get_logger(), "Insufficient pixels: left=%zu, right=%zu", 
          lane_pixels.left_pixels.size(), lane_pixels.right_pixels.size());
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "No white line detected, keeping previous result");
    }

    // Always use the stored lines/intersection for control (if available)
    if (has_valid_lines_) {
      // Calculate angular velocity to center the intersection
      double angular_velocity = calculate_centering_control(last_intersection_);

      // Publish control commands
      auto cmd_vel = geometry_msgs::msg::Twist();
      cmd_vel.linear.x = linear_velocity_;
      cmd_vel.angular.z = angular_velocity;
      cmd_vel_pub_->publish(cmd_vel);

      // Create and publish visualization
      if (enable_visualization_ && !latest_original_image_.empty()) {
        publish_visualization_image(latest_original_image_, msg->header, white_line_masks, last_lane_pixels_, 
                                   last_fitted_left_line_, last_fitted_right_line_, last_intersection_);
      }

      RCLCPP_INFO(this->get_logger(), 
        "Intersection at (%.0f, %.0f), Angular: %.3f",
        last_intersection_.x, last_intersection_.y, angular_velocity);
    } else {
      RCLCPP_WARN(this->get_logger(), "No valid lines available yet");
      if (enable_visualization_ && !latest_original_image_.empty()) {
        LaneLines empty_pixels;
        cv::Vec4f empty_left(0, 0, 0, 0);
        cv::Vec4f empty_right(0, 0, 0, 0);
        publish_visualization_image(latest_original_image_, msg->header, {}, empty_pixels, 
                                   empty_left, empty_right, cv::Point2f(-1, -1));
      }
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error in detection callback: %s", e.what());
  }
}

void LaneFollowingNode::detection_fallback_callback(const std_msgs::msg::String::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "[FALLBACK] Detection callback triggered");
  
  try {
    // Parse JSON data from fallback String message
    std::string json_data = msg->data;
    RCLCPP_DEBUG(this->get_logger(), "[FALLBACK] Received JSON: %s", json_data.substr(0, 200).c_str());
    
    // Simple JSON parsing without external library
    // Look for white_line_masks array
    std::regex white_line_regex(R"("white_line_masks":\s*\[([^\]]*)\])");
    std::smatch match;
    
    if (std::regex_search(json_data, match, white_line_regex)) {
      std::string masks_data = match[1].str();
      RCLCPP_INFO(this->get_logger(), "[FALLBACK] Found white line masks data: %s", masks_data.c_str());
      
      // Count number of masks by counting objects in the array
      int mask_count = 0;
      size_t pos = 0;
      while ((pos = masks_data.find("{", pos)) != std::string::npos) {
        mask_count++;
        pos++;
      }
      
      RCLCPP_INFO(this->get_logger(), "[FALLBACK] Detected %d white line masks", mask_count);
      
      if (mask_count > 0) {
        // Create dummy masks for skeleton detection
        // Since we can't transfer full mask data via String message,
        // we'll create simple rectangular masks based on the detection boxes
        
        // Extract boxes data
        std::regex boxes_regex(R"("boxes":\s*\[([^\]]*)\])");
        if (std::regex_search(json_data, match, boxes_regex)) {
          std::string boxes_data = match[1].str();
          RCLCPP_DEBUG(this->get_logger(), "[FALLBACK] Boxes data: %s", boxes_data.c_str());
          
          // Parse box coordinates
          std::vector<cv::Mat> dummy_masks;
          std::regex box_regex(R"(\[([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+)\])");
          std::sregex_iterator boxes_begin(boxes_data.begin(), boxes_data.end(), box_regex);
          std::sregex_iterator boxes_end;
          
          for (std::sregex_iterator i = boxes_begin; i != boxes_end; ++i) {
            std::smatch box_match = *i;
            
            float x1 = std::stof(box_match[1].str());
            float y1 = std::stof(box_match[2].str());
            float x2 = std::stof(box_match[3].str());
            float y2 = std::stof(box_match[4].str());
            
            RCLCPP_DEBUG(this->get_logger(), "[FALLBACK] Box: [%.1f, %.1f, %.1f, %.1f]", x1, y1, x2, y2);
            
            // Create rectangular mask from bounding box
            if (image_width_ > 0 && image_height_ > 0) {
              cv::Mat mask = cv::Mat::zeros(image_height_, image_width_, CV_8UC1);
              cv::Rect rect(
                static_cast<int>(std::max(0.0f, x1)),
                static_cast<int>(std::max(0.0f, y1)),
                static_cast<int>(std::min(static_cast<float>(image_width_), x2) - std::max(0.0f, x1)),
                static_cast<int>(std::min(static_cast<float>(image_height_), y2) - std::max(0.0f, y1))
              );
              
              if (rect.width > 0 && rect.height > 0) {
                mask(rect) = 255;
                dummy_masks.push_back(mask);
                RCLCPP_DEBUG(this->get_logger(), "[FALLBACK] Created mask with rect: %dx%d at (%d,%d)", 
                  rect.width, rect.height, rect.x, rect.y);
              }
            }
          }
          
          if (!dummy_masks.empty()) {
            // Split masks into left and right regions
            auto [left_mask, right_mask] = split_masks_left_right(dummy_masks);
            
            // Simple Hough line detection
            auto lane_lines = detect_lane_lines_separated(left_mask, right_mask);
            
            if (lane_lines.size() >= 2) {
              // Good detection - update stored lines and intersection
              cv::Vec2f left_line = lane_lines[0];
              cv::Vec2f right_line = lane_lines[1];
              cv::Point2f intersection = calculate_intersection(left_line, right_line);
              
              if (intersection.x >= 0 && intersection.y >= 0) {
                // Update stored results with new valid detection
                last_left_line_ = left_line;
                last_right_line_ = right_line;
                last_intersection_ = intersection;
                has_valid_lines_ = true;
                
                RCLCPP_INFO(this->get_logger(), "[FALLBACK] Updated lines with new detection");
                
                // Publish visualization with dummy masks (convert to pixel-based format)
                if (enable_visualization_ && !latest_original_image_.empty()) {
                  std_msgs::msg::Header header;
                  header.stamp = this->get_clock()->now();
                  header.frame_id = "camera_frame";
                  
                  // Note: fallback mode doesn't have pixel data, so use empty structures
                  LaneLines empty_pixels;
                  cv::Vec4f empty_left(0, 0, 0, 0);
                  cv::Vec4f empty_right(0, 0, 0, 0);
                  
                  publish_visualization_image(
                    latest_original_image_, header, dummy_masks, empty_pixels, 
                    empty_left, empty_right, intersection
                  );
                }
              } else {
                RCLCPP_WARN(this->get_logger(), "[FALLBACK] Lines are parallel, keeping previous intersection");
              }
            } else {
              RCLCPP_WARN(this->get_logger(), "[FALLBACK] Less than 2 lane lines detected: %zu", lane_lines.size());
            }
          } else {
            RCLCPP_WARN(this->get_logger(), "[FALLBACK] No valid masks created from boxes");
          }
        } else {
          RCLCPP_WARN(this->get_logger(), "[FALLBACK] No boxes data found in JSON");
        }
      } else {
        RCLCPP_WARN(this->get_logger(), "[FALLBACK] No white line masks detected");
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "[FALLBACK] No white_line_masks found in JSON data");
    }
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "[FALLBACK] Error in fallback detection callback: %s", e.what());
  }
}

void LaneFollowingNode::original_image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    latest_original_image_ = cv_ptr->image.clone();
    
    // Set image dimensions from first received image
    if (image_width_ == 0 || image_height_ == 0) {
      image_height_ = latest_original_image_.rows;
      image_width_ = latest_original_image_.cols;
    }
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
  }
}

void LaneFollowingNode::control_timer_callback()
{
  // Continuous control timer - ensure robot keeps moving
  auto cmd_vel = geometry_msgs::msg::Twist();
  cmd_vel.linear.x = linear_velocity_; // Always move forward
  cmd_vel.angular.z = 0.0; // Straight by default
  cmd_vel_pub_->publish(cmd_vel);
}

std::vector<cv::Mat> LaneFollowingNode::extract_white_line_masks(
  const lang_sam_msgs::msg::DetectionResult::SharedPtr msg)
{
  std::vector<cv::Mat> white_line_masks;
  
  RCLCPP_INFO(this->get_logger(), "Received %zu labels: ", msg->labels.size());
  for (size_t i = 0; i < msg->labels.size(); ++i) {
    RCLCPP_INFO(this->get_logger(), "  [%zu]: %s", i, msg->labels[i].c_str());
  }
  RCLCPP_INFO(this->get_logger(), "Received %zu masks", msg->masks.size());

  for (size_t i = 0; i < msg->labels.size() && i < msg->masks.size(); ++i) {
    const std::string& label = msg->labels[i];
    RCLCPP_INFO(this->get_logger(), "Processing label %zu: \"%s\"", i, label.c_str());
    
    // Check if this is a white line detection
    std::string lower_label = label;
    std::transform(lower_label.begin(), lower_label.end(), lower_label.begin(), ::tolower);
    
    if (lower_label.find("white line") != std::string::npos) {
      try {
        // Convert sensor_msgs/Image to OpenCV image
        auto cv_ptr = cv_bridge::toCvCopy(msg->masks[i], sensor_msgs::image_encodings::MONO8);
        white_line_masks.push_back(cv_ptr->image.clone());
        
        RCLCPP_INFO(this->get_logger(), 
          "Successfully extracted white line mask %zu, shape: %dx%d", 
          i, cv_ptr->image.cols, cv_ptr->image.rows);
        
        // Set image dimensions from first mask
        if (image_width_ == 0 || image_height_ == 0) {
          image_height_ = cv_ptr->image.rows;
          image_width_ = cv_ptr->image.cols;
        }
        
      } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "CV bridge exception for mask %zu: %s", i, e.what());
      }
    }
  }
  
  return white_line_masks;
}

std::pair<cv::Mat, cv::Mat> LaneFollowingNode::split_masks_left_right(const std::vector<cv::Mat>& masks)
{
  if (masks.empty() || image_width_ == 0 || image_height_ == 0) {
    cv::Mat empty_mask = cv::Mat::zeros(image_height_, image_width_, CV_8UC1);
    return {empty_mask, empty_mask};
  }

  // Combine all white line masks
  cv::Mat combined_mask = cv::Mat::zeros(image_height_, image_width_, CV_8UC1);
  for (const auto& mask : masks) {
    if (mask.size() == combined_mask.size()) {
      cv::bitwise_or(combined_mask, mask, combined_mask);
    }
  }

  // Apply ROI (Region of Interest) filtering
  cv::Mat roi_mask = cv::Mat::zeros(combined_mask.size(), CV_8UC1);
  int top = std::max(0, roi_top_offset_);
  int bottom = std::min(image_height_, image_height_ - roi_bottom_offset_);
  cv::rectangle(roi_mask, cv::Point(0, top), cv::Point(image_width_, bottom), cv::Scalar(255), -1);
  cv::bitwise_and(combined_mask, roi_mask, combined_mask);

  // Split image into left and right halves
  int split_x = static_cast<int>(image_width_ * lane_split_ratio_);
  
  cv::Mat left_mask = cv::Mat::zeros(combined_mask.size(), CV_8UC1);
  cv::Mat right_mask = cv::Mat::zeros(combined_mask.size(), CV_8UC1);
  
  // Left region: from 0 to split_x
  cv::Rect left_roi(0, 0, split_x, image_height_);
  combined_mask(left_roi).copyTo(left_mask(left_roi));
  
  // Right region: from split_x to image_width
  cv::Rect right_roi(split_x, 0, image_width_ - split_x, image_height_);
  combined_mask(right_roi).copyTo(right_mask(right_roi));
  
  RCLCPP_DEBUG(this->get_logger(), 
    "Split masks: left_pixels=%d, right_pixels=%d", 
    cv::countNonZero(left_mask), cv::countNonZero(right_mask));
  
  return {left_mask, right_mask};
}

std::pair<std::vector<cv::Point>, std::vector<cv::Point>> LaneFollowingNode::find_lane_boundaries_bottom_up(const cv::Mat& combined_mask)
{
  std::vector<cv::Point> left_pixels;
  std::vector<cv::Point> right_pixels;
  
  if (combined_mask.empty()) {
    return {left_pixels, right_pixels};
  }
  
  const int tolerance = 50; // Search tolerance in pixels
  int center_x = image_width_ / 2;
  
  // Start from bottom of image and scan upward
  for (int y = image_height_ - 1; y >= roi_top_offset_; y -= 2) {
    // Search center first
    bool found_center = false;
    if (y < combined_mask.rows && center_x < combined_mask.cols) {
      if (combined_mask.at<uchar>(y, center_x) > 0) {
        found_center = true;
      }
    }
    
    if (found_center) {
      // Expand left and right from center with tolerance
      int left_bound = std::max(0, center_x - tolerance);
      int right_bound = std::min(image_width_ - 1, center_x + tolerance);
      
      // Find leftmost pixel
      for (int x = center_x; x >= left_bound; x--) {
        if (combined_mask.at<uchar>(y, x) > 0) {
          left_pixels.push_back(cv::Point(x, y));
          break;
        }
      }
      
      // Find rightmost pixel
      for (int x = center_x; x <= right_bound; x++) {
        if (combined_mask.at<uchar>(y, x) > 0) {
          right_pixels.push_back(cv::Point(x, y));
          break;
        }
      }
    } else {
      // Search wider area if no center pixel found
      for (int x = std::max(0, center_x - tolerance); x < std::min(image_width_, center_x + tolerance); x++) {
        if (combined_mask.at<uchar>(y, x) > 0) {
          if (x < center_x) {
            left_pixels.push_back(cv::Point(x, y));
          } else {
            right_pixels.push_back(cv::Point(x, y));
          }
        }
      }
    }
  }
  
  RCLCPP_DEBUG(this->get_logger(), "Bottom-up scan found: %zu left pixels, %zu right pixels", 
    left_pixels.size(), right_pixels.size());
  
  return {left_pixels, right_pixels};
}

std::vector<cv::Vec2f> LaneFollowingNode::detect_lane_lines_separated(const cv::Mat& left_mask, const cv::Mat& right_mask)
{
  std::vector<cv::Vec2f> lane_lines;
  
  // Detect line in left region
  std::vector<cv::Vec2f> left_lines;
  if (cv::countNonZero(left_mask) > 0) {
    cv::HoughLines(left_mask, left_lines, hough_rho_, hough_theta_, hough_threshold_, 0, 0);
    
    // Filter horizontal lines
    left_lines = filter_horizontal_lines(left_lines);
    
    if (!left_lines.empty()) {
      // Select line closest to robot from left region
      cv::Vec2f best_left_line = select_closest_line_to_center(left_lines, true);
      lane_lines.push_back(best_left_line);
      RCLCPP_DEBUG(this->get_logger(), "Left line selected: rho=%.2f, theta=%.2f (from %zu candidates)", 
        best_left_line[0], best_left_line[1], left_lines.size());
    }
  }
  
  // Detect line in right region  
  std::vector<cv::Vec2f> right_lines;
  if (cv::countNonZero(right_mask) > 0) {
    cv::HoughLines(right_mask, right_lines, hough_rho_, hough_theta_, hough_threshold_, 0, 0);
    
    // Filter horizontal lines
    right_lines = filter_horizontal_lines(right_lines);
    
    if (!right_lines.empty()) {
      // Select line closest to robot from right region
      cv::Vec2f best_right_line = select_closest_line_to_center(right_lines, false);
      lane_lines.push_back(best_right_line);
      RCLCPP_DEBUG(this->get_logger(), "Right line selected: rho=%.2f, theta=%.2f (from %zu candidates)", 
        best_right_line[0], best_right_line[1], right_lines.size());
    }
  }
  
  RCLCPP_INFO(this->get_logger(), 
    "Lane lines detected: %zu total (%zu left, %zu right)", 
    lane_lines.size(), left_lines.size(), right_lines.size());
  
  return lane_lines;
}

std::vector<cv::Vec2f> LaneFollowingNode::filter_horizontal_lines(const std::vector<cv::Vec2f>& lines)
{
  std::vector<cv::Vec2f> filtered_lines;
  
  const double horizontal_threshold = 20.0 * CV_PI / 180.0; // 20 degrees in radians
  
  for (const auto& line : lines) {
    double theta = line[1];
    
    // Normalize theta to [0, PI]
    while (theta < 0) theta += CV_PI;
    while (theta >= CV_PI) theta -= CV_PI;
    
    // Check if line is too close to horizontal (0 or 180 degrees)
    bool is_near_zero = theta < horizontal_threshold;
    bool is_near_pi = theta > (CV_PI - horizontal_threshold);
    
    if (!is_near_zero && !is_near_pi) {
      filtered_lines.push_back(line);
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Filtered horizontal line: theta=%.2f degrees", 
        theta * 180.0 / CV_PI);
    }
  }
  
  RCLCPP_DEBUG(this->get_logger(), "Filtered %zu/%zu horizontal lines", 
    lines.size() - filtered_lines.size(), lines.size());
  
  return filtered_lines;
}

cv::Vec2f LaneFollowingNode::select_closest_line_to_center(const std::vector<cv::Vec2f>& lines, bool is_left_region)
{
  if (lines.empty()) {
    return cv::Vec2f(0, 0);
  }
  
  if (lines.size() == 1) {
    return lines[0];
  }
  
  // Image center position
  float center_x = static_cast<float>(image_width_) / 2.0f;
  float center_y = static_cast<float>(image_height_) / 2.0f;
  
  cv::Vec2f best_line = lines[0];
  float min_distance = std::numeric_limits<float>::max();
  
  for (const auto& line : lines) {
    float rho = line[0];
    float theta = line[1];
    
    // Check slope condition for mountain shape (山の形)
    // Left line should have negative slope (左肩下がり): theta > π/2
    // Right line should have negative slope (右肩下がり): theta < π/2
    bool slope_valid = false;
    
    if (is_left_region) {
      // Left region: line should slope down from left to right (theta > π/2)
      slope_valid = (theta > M_PI / 2.0 && theta < M_PI);
    } else {
      // Right region: line should slope down from left to right (theta < π/2)
      slope_valid = (theta > 0 && theta < M_PI / 2.0);
    }
    
    if (!slope_valid) {
      RCLCPP_DEBUG(this->get_logger(), 
        "%s line rejected: rho=%.1f, theta=%.3f (%.1f deg) - invalid slope",
        is_left_region ? "Left" : "Right",
        rho, theta, theta * 180.0 / M_PI);
      continue;
    }
    
    // Calculate perpendicular distance from image center to line
    // Line equation: x*cos(theta) + y*sin(theta) = rho
    // Distance = |center_x*cos(theta) + center_y*sin(theta) - rho|
    float distance_to_center = std::abs(center_x * std::cos(theta) + center_y * std::sin(theta) - rho);
    
    // Also check if the line passes through reasonable region
    // Find x coordinate of line at center_y level
    bool position_valid = true;
    if (std::abs(std::cos(theta)) > 1e-6) { // Avoid near-vertical lines
      float x_at_center_y = (rho - center_y * std::sin(theta)) / std::cos(theta);
      
      // For left region, line should be reasonably left of center
      // For right region, line should be reasonably right of center  
      if (is_left_region && x_at_center_y > center_x + 50) {
        position_valid = false; // Too far right for left region
      } else if (!is_left_region && x_at_center_y < center_x - 50) {
        position_valid = false; // Too far left for right region
      }
      
      RCLCPP_DEBUG(this->get_logger(), 
        "%s line: rho=%.1f, theta=%.3f (%.1f deg), x_at_center=%.1f, dist_to_center=%.1f, valid=%s",
        is_left_region ? "Left" : "Right",
        rho, theta, theta * 180.0 / M_PI, x_at_center_y, distance_to_center,
        position_valid ? "yes" : "no");
    }
    
    if (position_valid && distance_to_center < min_distance) {
      min_distance = distance_to_center;
      best_line = line;
    }
  }
  
  return best_line;
}

// Calculate x coordinate at given y position
float LaneFollowingNode::calculate_x_at_y(const cv::Vec2f& line, float y)
{
  float rho = line[0];
  float theta = line[1];
  
  if (std::abs(std::cos(theta)) < 1e-6) {
    return -1; // Vertical line
  }
  
  // x*cos(theta) + y*sin(theta) = rho
  // x = (rho - y*sin(theta)) / cos(theta)
  return (rho - y * std::sin(theta)) / std::cos(theta);
}

cv::Point2f LaneFollowingNode::calculate_intersection(const cv::Vec2f& line1, const cv::Vec2f& line2)
{
  float rho1 = line1[0], theta1 = line1[1];
  float rho2 = line2[0], theta2 = line2[1];
  
  // Convert to ax + by = c format
  float a1 = std::cos(theta1);
  float b1 = std::sin(theta1);
  float c1 = rho1;
  
  float a2 = std::cos(theta2);
  float b2 = std::sin(theta2);
  float c2 = rho2;
  
  // Calculate determinant
  float determinant = a1 * b2 - a2 * b1;
  if (std::abs(determinant) < 1e-6) {
    return cv::Point2f(-1, -1); // Parallel lines
  }
  
  float x = (c1 * b2 - c2 * b1) / determinant;
  float y = (a1 * c2 - a2 * c1) / determinant;
  
  return cv::Point2f(x, y);
}

double LaneFollowingNode::calculate_centering_control(const cv::Point2f& intersection)
{
  // If no valid intersection, use previous intersection for stability
  if (intersection.x < 0 || intersection.y < 0) {
    if (last_intersection_.x > 0 && last_intersection_.y > 0) {
      return calculate_normalized_error_control(last_intersection_);
    }
    return 0.0; // No valid intersection
  }
  
  // Update last valid intersection
  last_intersection_ = intersection;
  
  return calculate_normalized_error_control(intersection);
}

double LaneFollowingNode::calculate_normalized_error_control(const cv::Point2f& intersection)
{
  float current_x = intersection.x;
  float target_x = image_width_ / 2.0f;
  float pixel_error = current_x - target_x;
  
  // Normalize error to [-1.0, 1.0] range based on half image width
  float normalized_error = pixel_error / (image_width_ / 2.0f);
  normalized_error = std::max(-1.0f, std::min(1.0f, normalized_error));
  
  // PID control with normalized error
  auto current_time = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(current_time - last_time_).count();
  
  if (dt <= 0) dt = 0.01; // Avoid division by zero
  if (dt > 0.1) dt = 0.1; // Prevent large time jumps
  
  // Proportional term
  double p_term = kp_ * normalized_error;
  
  // Integral term with anti-windup
  error_sum_ += normalized_error * dt;
  // Clamp integral sum to prevent windup
  error_sum_ = std::max(-1.0, std::min(1.0, error_sum_));
  double i_term = ki_ * error_sum_;
  
  // Derivative term
  double d_term = kd_ * (normalized_error - last_error_) / dt;
  
  // Calculate angular velocity
  double angular_velocity = -(p_term + i_term + d_term);
  
  // Clamp angular velocity
  angular_velocity = std::max(-1.0, std::min(1.0, angular_velocity));
  
  // Update for next iteration
  last_error_ = normalized_error;
  last_time_ = current_time;
  
  return angular_velocity;
}

void LaneFollowingNode::publish_visualization_image(const cv::Mat& base_image,
                                                  const std_msgs::msg::Header& header,
                                                  const std::vector<cv::Mat>& white_line_masks,
                                                  const LaneLines& lane_pixels,
                                                  const cv::Vec4f& left_line,
                                                  const cv::Vec4f& right_line,
                                                  const cv::Point2f& intersection)
{
  try {
    cv::Mat viz_image = base_image.clone();
    
    // Draw detected pixels from bottom-up scanning
    for (const auto& pt : lane_pixels.left_pixels) {
      cv::circle(viz_image, pt, 2, cv::Scalar(255, 100, 100), -1); // Blue for left
    }
    
    for (const auto& pt : lane_pixels.right_pixels) {
      cv::circle(viz_image, pt, 2, cv::Scalar(100, 100, 255), -1); // Red for right
    }
    
    for (const auto& pt : lane_pixels.center_pixels) {
      cv::circle(viz_image, pt, 2, cv::Scalar(100, 255, 100), -1); // Green for center
    }
    
    // Draw fitted lines using Vec4f format (vx, vy, x0, y0)
    if (left_line[0] != 0 || left_line[1] != 0) {
      float vx = left_line[0], vy = left_line[1];
      float x0 = left_line[2], y0 = left_line[3];
      
      // Calculate line endpoints
      cv::Point pt1, pt2;
      pt1.x = cvRound(x0 - 1000 * vx);
      pt1.y = cvRound(y0 - 1000 * vy);
      pt2.x = cvRound(x0 + 1000 * vx);
      pt2.y = cvRound(y0 + 1000 * vy);
      
      cv::line(viz_image, pt1, pt2, cv::Scalar(0, 255, 0), line_thickness_); // Green for left line
    }
    
    if (right_line[0] != 0 || right_line[1] != 0) {
      float vx = right_line[0], vy = right_line[1];
      float x0 = right_line[2], y0 = right_line[3];
      
      // Calculate line endpoints
      cv::Point pt1, pt2;
      pt1.x = cvRound(x0 - 1000 * vx);
      pt1.y = cvRound(y0 - 1000 * vy);
      pt2.x = cvRound(x0 + 1000 * vx);
      pt2.y = cvRound(y0 + 1000 * vy);
      
      cv::line(viz_image, pt1, pt2, cv::Scalar(0, 255, 255), line_thickness_); // Yellow for right line
    }
    
    // Draw intersection point
    if (intersection.x >= 0 && intersection.y >= 0) {
      cv::circle(viz_image, intersection, circle_radius_, cv::Scalar(0, 0, 255), -1);
      
      // Draw error visualization
      cv::Point center(image_width_ / 2, intersection.y);
      cv::line(viz_image, intersection, center, cv::Scalar(255, 0, 255), 2);
    }
    
    // Draw image center for reference
    cv::Point center_point(image_width_ / 2, image_height_ / 2);
    cv::circle(viz_image, center_point, circle_radius_ / 2, cv::Scalar(255, 0, 0), -1);
    
    // Add text overlay with statistics
    std::string stats_text = "L:" + std::to_string(lane_pixels.left_pixels.size()) + 
                           " R:" + std::to_string(lane_pixels.right_pixels.size()) +
                           " C:" + std::to_string(lane_pixels.center_pixels.size());
    
    cv::putText(viz_image, stats_text, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    if (intersection.x >= 0 && intersection.y >= 0) {
      float error_pixels = intersection.x - (image_width_ / 2.0f);
      std::string error_text = "Error: " + std::to_string(static_cast<int>(error_pixels)) + "px";
      cv::putText(viz_image, error_text, cv::Point(10, 60), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    
    // Convert to ROS message and publish
    auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_image).toImageMsg();
    visualization_pub_->publish(*viz_msg);
    
    RCLCPP_DEBUG(this->get_logger(), "Published visualization: L:%zu R:%zu C:%zu pixels", 
      lane_pixels.left_pixels.size(), lane_pixels.right_pixels.size(), lane_pixels.center_pixels.size());
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Visualization error: %s", e.what());
  }
}

// New pixel-based processing functions
cv::Mat LaneFollowingNode::preprocess_mask_with_skeleton(const cv::Mat& combined_mask)
{
  if (combined_mask.empty()) {
    return combined_mask;
  }
  
  cv::Mat processed_mask;
  
  // Apply Gaussian blur to reduce noise (parameterized)
  int blur_size = std::max(1, gaussian_blur_size_);  // Ensure odd number
  if (blur_size % 2 == 0) blur_size += 1;
  cv::GaussianBlur(combined_mask, processed_mask, cv::Size(blur_size, blur_size), 0);
  
  // Morphological close operation to connect nearby regions (parameterized kernel size)
  int kernel_size = std::max(1, morph_kernel_size_);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
  cv::morphologyEx(processed_mask, processed_mask, cv::MORPH_CLOSE, kernel);
  
  // Threshold to ensure binary image
  cv::threshold(processed_mask, processed_mask, 127, 255, cv::THRESH_BINARY);
  
  // Optional: Apply skeletonization (thinning) using Zhang-Suen algorithm
  // Note: Skeletonization reduces pixel count significantly but provides cleaner lines
  
  if (use_skeleton_) {
    cv::Mat skeleton;
    cv::ximgproc::thinning(processed_mask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
    
    RCLCPP_DEBUG(this->get_logger(), "Skeleton pixels: %d (from %d processed)", 
      cv::countNonZero(skeleton), cv::countNonZero(processed_mask));
    
    return skeleton;
  } else {
    // Return processed mask without skeletonization (more pixels but thicker lines)
    RCLCPP_DEBUG(this->get_logger(), "Processed pixels: %d (skeletonization skipped)", 
      cv::countNonZero(processed_mask));
    
    return processed_mask;
  }
}

LaneLines LaneFollowingNode::extract_lane_pixels_bottom_up(const cv::Mat& skeleton_mask)
{
  LaneLines lane_pixels;
  
  if (skeleton_mask.empty()) {
    return lane_pixels;
  }
  
  int center_x = image_width_ / 2;
  int tolerance = pixel_search_tolerance_;
  
  // Bottom-up scanning from reference code
  for (int y = image_height_ - 1; y >= roi_top_offset_; y -= 1) {
    bool found_left = false;
    bool found_right = false;
    bool found_center = false;
    
    // Check center pixel first
    if (skeleton_mask.at<uchar>(y, center_x) > 0) {
      found_center = true;
      lane_pixels.center_pixels.push_back(cv::Point(center_x, y));
    }
    
    // Search left from center with tolerance
    for (int x = center_x - 1; x >= std::max(0, center_x - tolerance); x--) {
      if (skeleton_mask.at<uchar>(y, x) > 0) {
        if (!found_left) {
          lane_pixels.left_pixels.push_back(cv::Point(x, y));
          found_left = true;
        }
      }
    }
    
    // Search right from center with tolerance
    for (int x = center_x + 1; x <= std::min(image_width_ - 1, center_x + tolerance); x++) {
      if (skeleton_mask.at<uchar>(y, x) > 0) {
        if (!found_right) {
          lane_pixels.right_pixels.push_back(cv::Point(x, y));
          found_right = true;
        }
      }
    }
    
    // Adaptive tolerance: expand search if no pixels found (parameterized)
    if (enable_adaptive_tolerance_ && !found_left && !found_right && !found_center) {
      int extended_tolerance = tolerance * 2;
      
      for (int x = std::max(0, center_x - extended_tolerance); 
           x <= std::min(image_width_ - 1, center_x + extended_tolerance); x++) {
        if (skeleton_mask.at<uchar>(y, x) > 0) {
          if (x < center_x && !found_left) {
            lane_pixels.left_pixels.push_back(cv::Point(x, y));
            found_left = true;
          } else if (x > center_x && !found_right) {
            lane_pixels.right_pixels.push_back(cv::Point(x, y));
            found_right = true;
          }
        }
      }
    }
  }
  
  RCLCPP_DEBUG(this->get_logger(), "Extracted pixels: left=%zu, right=%zu, center=%zu", 
    lane_pixels.left_pixels.size(), 
    lane_pixels.right_pixels.size(), 
    lane_pixels.center_pixels.size());
  
  return lane_pixels;
}

std::pair<cv::Vec4f, cv::Vec4f> LaneFollowingNode::fit_lane_lines(const LaneLines& lane_pixels)
{
  cv::Vec4f left_line(0, 0, 0, 0);
  cv::Vec4f right_line(0, 0, 0, 0);
  
  // Fit line to left pixels using cv::fitLine
  if (lane_pixels.left_pixels.size() > 5) {
    cv::fitLine(lane_pixels.left_pixels, left_line, cv::DIST_L2, 0, 0.01, 0.01);
    RCLCPP_DEBUG(this->get_logger(), "Left line fitted: vx=%.2f, vy=%.2f, x0=%.2f, y0=%.2f", 
      left_line[0], left_line[1], left_line[2], left_line[3]);
  }
  
  // Fit line to right pixels
  if (lane_pixels.right_pixels.size() > 5) {
    cv::fitLine(lane_pixels.right_pixels, right_line, cv::DIST_L2, 0, 0.01, 0.01);
    RCLCPP_DEBUG(this->get_logger(), "Right line fitted: vx=%.2f, vy=%.2f, x0=%.2f, y0=%.2f", 
      right_line[0], right_line[1], right_line[2], right_line[3]);
  }
  
  return {left_line, right_line};
}

cv::Point2f LaneFollowingNode::calculate_intersection_from_fitted_lines(const cv::Vec4f& left_line, const cv::Vec4f& right_line)
{
  // Extract line parameters (vx, vy, x0, y0)
  float vx1 = left_line[0], vy1 = left_line[1];
  float x1 = left_line[2], y1 = left_line[3];
  
  float vx2 = right_line[0], vy2 = right_line[1];
  float x2 = right_line[2], y2 = right_line[3];
  
  // Check for parallel lines
  float det = vx1 * vy2 - vx2 * vy1;
  if (std::abs(det) < 1e-6) {
    return cv::Point2f(-1, -1);
  }
  
  // Calculate intersection using parametric line equations
  float t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
  
  float intersection_x = x1 + t * vx1;
  float intersection_y = y1 + t * vy1;
  
  // Validate intersection is within image bounds
  if (intersection_x < 0 || intersection_x >= image_width_ || 
      intersection_y < 0 || intersection_y >= image_height_) {
    // Try to find intersection at a specific y-coordinate (e.g., middle of image)
    float target_y = image_height_ * 0.6f;
    
    // Calculate x coordinates at target_y for both lines
    float t1 = (target_y - y1) / vy1;
    float t2 = (target_y - y2) / vy2;
    
    float x1_at_target = x1 + t1 * vx1;
    float x2_at_target = x2 + t2 * vx2;
    
    // Use midpoint as intersection
    intersection_x = (x1_at_target + x2_at_target) / 2.0f;
    intersection_y = target_y;
  }
  
  return cv::Point2f(intersection_x, intersection_y);
}

} // namespace lang_sam_nav

// Main function
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lang_sam_nav::LaneFollowingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}