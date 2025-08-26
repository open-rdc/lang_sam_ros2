#include "lang_sam_nav/lane_following_node.hpp"
#include <algorithm>
#include <cmath>

namespace lang_sam_nav
{

LaneFollowingNode::LaneFollowingNode(const rclcpp::NodeOptions & options)
: Node("lane_following_node", options),
  image_width_(0),
  image_height_(0),
  last_left_line_(cv::Vec4f(0, 0, 0, 0)),
  last_right_line_(cv::Vec4f(0, 0, 0, 0)),
  last_intersection_(cv::Point2f(-1, -1)),
  has_valid_lines_(false)
{
  // パラメータ宣言（YOLOP Nav互換）
  this->declare_parameter("linear_velocity", 1.0);
  this->declare_parameter("kp", 1.0);
  this->declare_parameter("max_angular_velocity", 1.0);
  this->declare_parameter("original_image_topic", "/zed/zed_node/rgb/image_rect_color");
  this->declare_parameter("pixel_tolerance", 50);
  this->declare_parameter("enable_visualization", true);
  this->declare_parameter("line_thickness", 3);
  this->declare_parameter("circle_radius", 10);

  // パラメータ取得
  linear_velocity_ = this->get_parameter("linear_velocity").as_double();
  kp_ = this->get_parameter("kp").as_double();
  max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
  original_image_topic_ = this->get_parameter("original_image_topic").as_string();
  pixel_tolerance_ = this->get_parameter("pixel_tolerance").as_int();
  enable_visualization_ = this->get_parameter("enable_visualization").as_bool();
  line_thickness_ = this->get_parameter("line_thickness").as_int();
  circle_radius_ = this->get_parameter("circle_radius").as_int();

  // CV Bridge初期化
  cv_bridge_ptr_ = std::make_shared<cv_bridge::CvImage>();
  
  // LanePixelFinder初期化
  pixel_finder_ = std::make_unique<LanePixelFinder>(pixel_tolerance_);

  // サブスクライバー作成
  RCLCPP_INFO(this->get_logger(), "DetectionResultサブスクライバー作成: /lang_sam_detections");
  detection_sub_ = this->create_subscription<lang_sam_msgs::msg::DetectionResult>(
    "/lang_sam_detections", 10,
    std::bind(&LaneFollowingNode::detection_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "画像サブスクライバー作成: %s", original_image_topic_.c_str());
  original_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    original_image_topic_, 10,
    std::bind(&LaneFollowingNode::original_image_callback, this, std::placeholders::_1));

  // パブリッシャー作成
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/lane_following_visualization", 10);

  RCLCPP_INFO(this->get_logger(), "Lane Following Node (YOLOP Nav互換) 初期化完了");
}

void LaneFollowingNode::detection_callback(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "検出コールバック開始: ラベル数=%zu, マスク数=%zu", 
              msg->labels.size(), msg->masks.size());
  
  try {
    // 白線マスクを抽出
    std::vector<cv::Mat> white_line_masks = extract_white_line_masks(msg);
    
    if (white_line_masks.empty()) {
      RCLCPP_WARN(this->get_logger(), "白線が検出されませんでした");
      // 空の可視化を送信
      if (enable_visualization_ && !latest_original_image_.empty()) {
        std::vector<cv::Point> empty_pixels;
        cv::Vec4f empty_line(0, 0, 0, 0);
        publishVisualization(latest_original_image_, msg->header,
                            empty_pixels, empty_pixels, empty_pixels,
                            empty_line, empty_line, cv::Point2f(-1, -1));
      }
      return;
    }
    
    RCLCPP_INFO(this->get_logger(), "白線マスク数: %zu", white_line_masks.size());
    
    // マスクを結合
    cv::Mat combined_mask = combine_masks(white_line_masks);
    
    // スケルトン化
    cv::Mat skeletonized = skeletonize(combined_mask);
    
    // ノイズ除去をスキップして、スケルトン化した画像を直接使用
    // cv::Mat denoised = denoise(skeletonized);
    
    // 水平線フィルタリングもスキップ
    // cv::Mat filtered = filterHorizontalLines(denoised);
    
    // 車線ピクセル検出（スケルトン化後の画像を直接使用）
    std::vector<cv::Point> left_pixels, right_pixels, center_pixels;
    pixel_finder_->findLanePixels(skeletonized, left_pixels, right_pixels, center_pixels);
    
    RCLCPP_INFO(this->get_logger(), "ピクセル検出結果: 左=%zu, 右=%zu, 中央=%zu",
                left_pixels.size(), right_pixels.size(), center_pixels.size());
    
    // ピクセル数チェック
    if (left_pixels.size() < 5 || right_pixels.size() < 5) {
      RCLCPP_WARN(this->get_logger(), "不十分なピクセル数: 左=%zu, 右=%zu", 
                  left_pixels.size(), right_pixels.size());
      // 前回の有効な結果を使用
      if (!has_valid_lines_) {
        return;
      }
    } else {
      // 直線フィッティング
      auto [left_line, right_line] = fitLaneLines(left_pixels, right_pixels);
      
      // 交点計算
      cv::Point2f intersection = calculateIntersection(left_line, right_line);
      
      if (intersection.x >= 0 && intersection.y >= 0) {
        // 有効な検出結果を保存
        last_left_line_ = left_line;
        last_right_line_ = right_line;
        last_intersection_ = intersection;
        has_valid_lines_ = true;
      }
    }
    
    // 制御指令を生成（前回の有効な結果を使用）
    if (has_valid_lines_) {
      double angular_velocity = calculateControl(last_intersection_);
      
      auto cmd_vel = geometry_msgs::msg::Twist();
      cmd_vel.linear.x = linear_velocity_;
      cmd_vel.angular.z = angular_velocity;
      cmd_vel_pub_->publish(cmd_vel);
      
      RCLCPP_INFO(this->get_logger(), "制御: 交点=(%.0f, %.0f), 角速度=%.3f",
                  last_intersection_.x, last_intersection_.y, angular_velocity);
    }
    
    // 可視化
    if (enable_visualization_ && !latest_original_image_.empty()) {
      publishVisualization(latest_original_image_, msg->header,
                          left_pixels, right_pixels, center_pixels,
                          last_left_line_, last_right_line_, last_intersection_);
    }
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "検出コールバックエラー: %s", e.what());
  }
}

void LaneFollowingNode::original_image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    latest_original_image_ = cv_ptr->image;
    
    if (image_width_ == 0 || image_height_ == 0) {
      image_width_ = latest_original_image_.cols;
      image_height_ = latest_original_image_.rows;
      RCLCPP_INFO_ONCE(this->get_logger(), "画像サイズ設定: %dx%d", image_width_, image_height_);
    }
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge例外: %s", e.what());
  }
}

std::vector<cv::Mat> LaneFollowingNode::extract_white_line_masks(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg)
{
  std::vector<cv::Mat> white_line_masks;
  
  for (size_t i = 0; i < msg->labels.size(); ++i) {
    if (msg->labels[i] == "white line") {
      if (i < msg->masks.size()) {
        try {
          cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->masks[i], sensor_msgs::image_encodings::MONO8);
          white_line_masks.push_back(cv_ptr->image);
        } catch (cv_bridge::Exception& e) {
          RCLCPP_ERROR(this->get_logger(), "マスク変換エラー: %s", e.what());
        }
      }
    }
  }
  
  return white_line_masks;
}

cv::Mat LaneFollowingNode::combine_masks(const std::vector<cv::Mat>& masks)
{
  if (masks.empty()) {
    return cv::Mat();
  }
  
  cv::Mat combined = cv::Mat::zeros(masks[0].size(), CV_8UC1);
  for (const auto& mask : masks) {
    cv::bitwise_or(combined, mask, combined);
  }
  
  int non_zero = cv::countNonZero(combined);
  RCLCPP_INFO(this->get_logger(), "結合マスク: サイズ=%dx%d, 非ゼロピクセル=%d",
              combined.cols, combined.rows, non_zero);
  
  return combined;
}

cv::Mat LaneFollowingNode::skeletonize(const cv::Mat& mask)
{
  cv::Mat skeleton;
  cv::ximgproc::thinning(mask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
  
  int non_zero = cv::countNonZero(skeleton);
  RCLCPP_INFO(this->get_logger(), "スケルトン化後: 非ゼロピクセル=%d", non_zero);
  
  return skeleton;
}

cv::Mat LaneFollowingNode::denoise(const cv::Mat& image)
{
  cv::Mat denoised;
  // モルフォロジー演算でノイズ除去
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(image, denoised, cv::MORPH_OPEN, kernel);
  
  int non_zero = cv::countNonZero(denoised);
  RCLCPP_INFO(this->get_logger(), "ノイズ除去後: 非ゼロピクセル=%d", non_zero);
  
  return denoised;
}

cv::Mat LaneFollowingNode::filterHorizontalLines(const cv::Mat& image)
{
  cv::Mat filtered = image.clone();
  
  // ハフ変換で水平線を検出して除去
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(filtered, lines, 1, CV_PI/180, 30);
  
  for (const auto& line : lines) {
    float theta = line[1];
    // 水平線（±10度）を除去
    if (std::abs(theta - CV_PI/2) < CV_PI/18) {
      float rho = line[0];
      double a = std::cos(theta);
      double b = std::sin(theta);
      double x0 = a * rho;
      double y0 = b * rho;
      cv::Point pt1(cvRound(x0 + 1000*(-b)), cvRound(y0 + 1000*(a)));
      cv::Point pt2(cvRound(x0 - 1000*(-b)), cvRound(y0 - 1000*(a)));
      cv::line(filtered, pt1, pt2, cv::Scalar(0), 3);
    }
  }
  
  return filtered;
}

std::pair<cv::Vec4f, cv::Vec4f> LaneFollowingNode::fitLaneLines(const std::vector<cv::Point>& left_pixels,
                                                                const std::vector<cv::Point>& right_pixels)
{
  cv::Vec4f left_line(0, 0, 0, 0);
  cv::Vec4f right_line(0, 0, 0, 0);
  
  if (left_pixels.size() >= 5) {
    cv::fitLine(left_pixels, left_line, cv::DIST_L2, 0, 0.01, 0.01);
  }
  
  if (right_pixels.size() >= 5) {
    cv::fitLine(right_pixels, right_line, cv::DIST_L2, 0, 0.01, 0.01);
  }
  
  return {left_line, right_line};
}

cv::Point2f LaneFollowingNode::calculateIntersection(const cv::Vec4f& left_line, const cv::Vec4f& right_line)
{
  // Vec4f: (vx, vy, x0, y0) - 方向ベクトルと通過点
  float vx1 = left_line[0], vy1 = left_line[1];
  float x1 = left_line[2], y1 = left_line[3];
  
  float vx2 = right_line[0], vy2 = right_line[1];
  float x2 = right_line[2], y2 = right_line[3];
  
  // 平行チェック
  float det = vx1 * vy2 - vx2 * vy1;
  if (std::abs(det) < 1e-6) {
    return cv::Point2f(-1, -1);
  }
  
  // 交点計算
  float t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
  float intersection_x = x1 + t * vx1;
  float intersection_y = y1 + t * vy1;
  
  return cv::Point2f(intersection_x, intersection_y);
}

double LaneFollowingNode::calculateControl(const cv::Point2f& intersection)
{
  if (intersection.x < 0 || intersection.y < 0) {
    return 0.0;
  }
  
  // 画像中心からのエラー計算
  float center_x = image_width_ / 2.0f;
  float error = (intersection.x - center_x) / center_x;  // -1.0 ~ 1.0に正規化
  
  // P制御のみ（YOLOP Nav互換）
  double angular_velocity = -kp_ * error;
  
  // 最大角速度で制限
  angular_velocity = std::max(-max_angular_velocity_, 
                              std::min(max_angular_velocity_, angular_velocity));
  
  return angular_velocity;
}

void LaneFollowingNode::publishVisualization(const cv::Mat& base_image,
                                            const std_msgs::msg::Header& header,
                                            const std::vector<cv::Point>& left_pixels,
                                            const std::vector<cv::Point>& right_pixels,
                                            const std::vector<cv::Point>& center_pixels,
                                            const cv::Vec4f& left_line,
                                            const cv::Vec4f& right_line,
                                            const cv::Point2f& intersection)
{
  cv::Mat viz_image = base_image.clone();
  
  // ピクセルを描画
  for (const auto& pt : left_pixels) {
    cv::circle(viz_image, pt, 2, cv::Scalar(255, 100, 100), -1);  // 青
  }
  
  for (const auto& pt : right_pixels) {
    cv::circle(viz_image, pt, 2, cv::Scalar(100, 100, 255), -1);  // 赤
  }
  
  for (const auto& pt : center_pixels) {
    cv::circle(viz_image, pt, 2, cv::Scalar(100, 255, 100), -1);  // 緑
  }
  
  // フィッティングされた線を描画
  if (left_line[0] != 0 || left_line[1] != 0) {
    float vx = left_line[0], vy = left_line[1];
    float x0 = left_line[2], y0 = left_line[3];
    cv::Point pt1(cvRound(x0 - 1000 * vx), cvRound(y0 - 1000 * vy));
    cv::Point pt2(cvRound(x0 + 1000 * vx), cvRound(y0 + 1000 * vy));
    cv::line(viz_image, pt1, pt2, cv::Scalar(255, 255, 0), line_thickness_);  // 黄色
  }
  
  if (right_line[0] != 0 || right_line[1] != 0) {
    float vx = right_line[0], vy = right_line[1];
    float x0 = right_line[2], y0 = right_line[3];
    cv::Point pt1(cvRound(x0 - 1000 * vx), cvRound(y0 - 1000 * vy));
    cv::Point pt2(cvRound(x0 + 1000 * vx), cvRound(y0 + 1000 * vy));
    cv::line(viz_image, pt1, pt2, cv::Scalar(255, 255, 0), line_thickness_);  // 黄色
  }
  
  // 交点を描画
  if (intersection.x >= 0 && intersection.y >= 0) {
    cv::circle(viz_image, cv::Point(cvRound(intersection.x), cvRound(intersection.y)), 
              circle_radius_, cv::Scalar(0, 0, 255), -1);  // 赤
  }
  
  // 画像中心線を描画
  int center_x = image_width_ / 2;
  cv::line(viz_image, cv::Point(center_x, 0), cv::Point(center_x, image_height_),
          cv::Scalar(0, 255, 0), 2);  // 緑
  
  // ROS2メッセージとして配信
  cv_bridge_ptr_->image = viz_image;
  cv_bridge_ptr_->encoding = sensor_msgs::image_encodings::BGR8;
  cv_bridge_ptr_->header = header;
  visualization_pub_->publish(*cv_bridge_ptr_->toImageMsg());
}

} // namespace lang_sam_nav

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lang_sam_nav::LaneFollowingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}