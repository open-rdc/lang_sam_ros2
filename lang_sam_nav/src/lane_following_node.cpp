#include "lang_sam_nav/lane_following_node.hpp"
#include <algorithm>
#include <cmath>

namespace lang_sam_nav
{

LaneFollowingNode::LaneFollowingNode(const rclcpp::NodeOptions & options)
: Node("lane_following_node", options),
  last_left_line_(cv::Vec4f(0, 0, 0, 0)),
  last_right_line_(cv::Vec4f(0, 0, 0, 0)),
  last_intersection_(cv::Point2f(-1, -1)),
  has_valid_lines_(false),
  image_width_(0),
  image_height_(0)
{
  // パラメータ宣言
  this->declare_parameter("linear_velocity", 1.0);
  this->declare_parameter("kp", 1.0);
  this->declare_parameter("max_angular_velocity", 1.0);
  this->declare_parameter("original_image_topic", "/zed/zed_node/rgb/image_rect_color");
  this->declare_parameter("enable_visualization", true);
  this->declare_parameter("line_thickness", 3);
  this->declare_parameter("circle_radius", 10);

  // パラメータ取得
  linear_velocity_ = this->get_parameter("linear_velocity").as_double();
  kp_ = this->get_parameter("kp").as_double();
  max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
  original_image_topic_ = this->get_parameter("original_image_topic").as_string();
  enable_visualization_ = this->get_parameter("enable_visualization").as_bool();
  line_thickness_ = this->get_parameter("line_thickness").as_int();
  circle_radius_ = this->get_parameter("circle_radius").as_int();

  // CV Bridge初期化
  cv_bridge_ptr_ = std::make_shared<cv_bridge::CvImage>();

  // サブスクライバー作成
  detection_sub_ = this->create_subscription<lang_sam_msgs::msg::DetectionResult>(
    "/lang_sam_detections", 10,
    std::bind(&LaneFollowingNode::detection_callback, this, std::placeholders::_1));

  original_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    original_image_topic_, 10,
    std::bind(&LaneFollowingNode::original_image_callback, this, std::placeholders::_1));

  // パブリッシャー作成
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/lane_following_visualization", 10);

  RCLCPP_INFO(this->get_logger(), "Lane Following Node 初期化完了");
}

void LaneFollowingNode::detection_callback(const lang_sam_msgs::msg::DetectionResult::SharedPtr msg)
{
  try {
    if (image_width_ == 0 || image_height_ == 0) {
      RCLCPP_WARN(this->get_logger(), "画像サイズが未初期化です");
      return;
    }

    // 白線マスクを抽出
    std::vector<cv::Mat> white_line_masks = extract_white_line_masks(msg);
    RCLCPP_INFO(this->get_logger(), "マスク数=%zu", white_line_masks.size());

    if (white_line_masks.empty()) {
      RCLCPP_WARN(this->get_logger(), "白線が検出されませんでした");
      return;
    }

    // マスクを結合
    cv::Mat combined_mask = combine_masks(white_line_masks);
    if (combined_mask.empty()) {
      RCLCPP_WARN(this->get_logger(), "マスク結合に失敗しました");
      return;
    }

    // ハフ変換で直線検出
    std::vector<cv::Vec4f> lines = detectLinesWithHough(combined_mask);

    if (lines.size() >= 2) {
      // 左右の線を分類
      auto [hough_left_line, hough_right_line, left_idx, right_idx] = classifyLeftRightLines(lines);

      if (left_idx != -1 && right_idx != -1) {
        // ハフ線を線形近似で画像境界まで延長
        cv::Vec4f left_line = extendLineWithFitting(hough_left_line);
        cv::Vec4f right_line = extendLineWithFitting(hough_right_line);

        // 前回の結果を保存
        last_left_line_ = left_line;
        last_right_line_ = right_line;

        // 交点計算と制御
        cv::Point2f intersection = calculateIntersection(left_line, right_line);

        if (intersection.x >= 0 && intersection.y >= 0) {
          last_intersection_ = intersection;
          has_valid_lines_ = true;
          publishControl();
          publishVisualization(msg->header, {left_line, right_line}, intersection);
          return;
        }
      }
    }

    // 検出失敗時は前回の交点で制御継続
    if (has_valid_lines_) {
      publishControl();
      publishVisualization(msg->header, std::vector<cv::Vec4f>(), last_intersection_);
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "エラー: %s", e.what());
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

  RCLCPP_INFO(this->get_logger(), "結合マスク: サイズ=%dx%d, 非ゼロピクセル=%d",
              combined.cols, combined.rows, cv::countNonZero(combined));

  return combined;
}

cv::Vec4f LaneFollowingNode::extendLineWithFitting(const cv::Vec4f& line)
{
  // ハフ線の2点を取得
  std::vector<cv::Point> points;
  points.push_back(cv::Point(line[0], line[1]));
  points.push_back(cv::Point(line[2], line[3]));

  // 2点から線形近似を実行
  cv::Vec4f line_params;
  cv::fitLine(points, line_params, cv::DIST_L2, 0, 0.01, 0.01);

  float vx = line_params[0];
  float vy = line_params[1];
  float x0 = line_params[2];
  float y0 = line_params[3];

  // 画像境界まで延長
  int x1, y1, x2, y2;

  if (std::abs(vy) > std::abs(vx)) {
    // 垂直に近い線
    y1 = 0;
    x1 = static_cast<int>(x0 - (y0 / vy) * vx);
    y2 = image_height_ - 1;
    x2 = static_cast<int>(x0 + ((image_height_ - y0) / vy) * vx);

    x1 = std::max(0, std::min(image_width_ - 1, x1));
    x2 = std::max(0, std::min(image_width_ - 1, x2));
  } else {
    // 水平に近い線
    x1 = 0;
    y1 = static_cast<int>(y0 - (x0 / vx) * vy);
    x2 = image_width_ - 1;
    y2 = static_cast<int>(y0 + ((image_width_ - x0) / vx) * vy);

    y1 = std::max(0, std::min(image_height_ - 1, y1));
    y2 = std::max(0, std::min(image_height_ - 1, y2));
  }

  RCLCPP_INFO(this->get_logger(), "線延長: (%.0f,%.0f)-(%.0f,%.0f) → (%d,%d)-(%d,%d)",
              line[0], line[1], line[2], line[3], x1, y1, x2, y2);

  return cv::Vec4f(x1, y1, x2, y2);
}

std::vector<cv::Vec4f> LaneFollowingNode::detectLinesWithHough(const cv::Mat& mask)
{
  std::vector<cv::Vec4f> result_lines;
  std::vector<cv::Vec4i> lines;

  // ハフ変換で直線検出
  cv::HoughLinesP(mask, lines, 1, CV_PI/180, 50, 50, 10);

  // Vec4iからVec4fへ変換
  for (const auto& line : lines) {
    result_lines.push_back(cv::Vec4f(line[0], line[1], line[2], line[3]));
  }

  RCLCPP_INFO(this->get_logger(), "ハフ変換検出線数: %zu", result_lines.size());
  return result_lines;
}

std::tuple<cv::Vec4f, cv::Vec4f, int, int> LaneFollowingNode::classifyLeftRightLines(const std::vector<cv::Vec4f>& lines)
{
  cv::Vec4f left_line(0, 0, 0, 0);
  cv::Vec4f right_line(0, 0, 0, 0);
  int left_idx = -1;
  int right_idx = -1;

  float center_x = image_width_ / 2.0f;

  float best_left_score = -1;
  float best_right_score = -1;

  for (size_t i = 0; i < lines.size(); ++i) {
    const auto& line = lines[i];

    // 線の中点を計算
    float mid_x = (line[0] + line[2]) / 2.0f;

    // 垂直性を計算（垂直に近いほどスコアが高い）
    float dx = std::abs(line[2] - line[0]);
    float dy = std::abs(line[3] - line[1]);
    float verticality = dy / (dx + 1e-6);

    // 線の長さ
    float length = std::sqrt(dx * dx + dy * dy);

    // スコア計算（垂直性と長さを考慮）
    float score = verticality * length;

    // 左右に分類
    if (mid_x < center_x) {
      // 左側の線
      if (score > best_left_score) {
        best_left_score = score;
        left_line = line;
        left_idx = i;
      }
    } else {
      // 右側の線
      if (score > best_right_score) {
        best_right_score = score;
        right_line = line;
        right_idx = i;
      }
    }
  }

  RCLCPP_INFO(this->get_logger(), "左線インデックス: %d, 右線インデックス: %d", left_idx, right_idx);

  return {left_line, right_line, left_idx, right_idx};
}

cv::Point2f LaneFollowingNode::calculateIntersection(const cv::Vec4f& left_line, const cv::Vec4f& right_line)
{
  RCLCPP_INFO(this->get_logger(), "交点計算開始");
  RCLCPP_INFO(this->get_logger(), "左線: (%.1f,%.1f)-(%.1f,%.1f)",
              left_line[0], left_line[1], left_line[2], left_line[3]);
  RCLCPP_INFO(this->get_logger(), "右線: (%.1f,%.1f)-(%.1f,%.1f)",
              right_line[0], right_line[1], right_line[2], right_line[3]);

  // HoughLinesP の結果: (x1, y1, x2, y2)
  if (left_line[0] == 0 && left_line[1] == 0 && left_line[2] == 0 && left_line[3] == 0) {
    RCLCPP_WARN(this->get_logger(), "左線が無効: すべて0");
    return cv::Point2f(-1, -1);
  }
  if (right_line[0] == 0 && right_line[1] == 0 && right_line[2] == 0 && right_line[3] == 0) {
    RCLCPP_WARN(this->get_logger(), "右線が無効: すべて0");
    return cv::Point2f(-1, -1);
  }

  float x1 = left_line[0], y1 = left_line[1], x2 = left_line[2], y2 = left_line[3];
  float x3 = right_line[0], y3 = right_line[1], x4 = right_line[2], y4 = right_line[3];

  // 直線の交点計算
  float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  RCLCPP_INFO(this->get_logger(), "分母計算: denom=%.6f", denom);

  if (std::abs(denom) < 1e-6) {
    RCLCPP_WARN(this->get_logger(), "平行線: denom=%.6f", denom);
    return cv::Point2f(-1, -1);  // 平行線
  }

  float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  RCLCPP_INFO(this->get_logger(), "パラメータt=%.6f", t);

  float intersection_x = x1 + t * (x2 - x1);
  float intersection_y = y1 + t * (y2 - y1);

  RCLCPP_INFO(this->get_logger(), "計算された交点: (%.1f, %.1f)", intersection_x, intersection_y);

  return cv::Point2f(intersection_x, intersection_y);
}



void LaneFollowingNode::publishControl()
{
  if (has_valid_lines_) {
    double angular_velocity = calculateControl(last_intersection_);

    auto cmd_vel = geometry_msgs::msg::Twist();
    cmd_vel.linear.x = linear_velocity_;
    cmd_vel.angular.z = angular_velocity;
    cmd_vel_pub_->publish(cmd_vel);

    RCLCPP_INFO(this->get_logger(), "制御: 交点=(%.0f, %.0f), 角速度=%.3f",
                last_intersection_.x, last_intersection_.y, angular_velocity);
  }
}

double LaneFollowingNode::calculateControl(const cv::Point2f& intersection)
{
  if (intersection.x < 0 || intersection.y < 0) {
    return 0.0;
  }

  // 画像中心からのエラー計算
  float center_x = image_width_ / 2.0f;
  float error = (intersection.x - center_x) / center_x;  // -1.0 ~ 1.0に正規化

  // P制御
  double angular_velocity = -kp_ * error;

  // 最大角速度で制限
  angular_velocity = std::max(-max_angular_velocity_,
                              std::min(max_angular_velocity_, angular_velocity));

  return angular_velocity;
}

void LaneFollowingNode::publishVisualization(const std_msgs::msg::Header& header,
                                            const std::vector<cv::Vec4f>& lane_lines,
                                            const cv::Point2f& intersection)
{
  if (!enable_visualization_ || latest_original_image_.empty()) {
    return;
  }

  cv::Mat viz_image = latest_original_image_.clone();

  // レーン線を描画（青色、太い線）
  for (const auto& line : lane_lines) {
    if (line[0] != 0 || line[1] != 0 || line[2] != 0 || line[3] != 0) {
      cv::Point pt1(cvRound(line[0]), cvRound(line[1]));
      cv::Point pt2(cvRound(line[2]), cvRound(line[3]));
      cv::line(viz_image, pt1, pt2, cv::Scalar(255, 0, 0), line_thickness_);  // 青色

      // 線の端点を青の円で表示
      cv::circle(viz_image, pt1, 3, cv::Scalar(255, 0, 0), -1);  // 青
      cv::circle(viz_image, pt2, 3, cv::Scalar(255, 0, 0), -1);  // 青
    }
  }

  // 交点を描画（座標テキストなし）
  if (intersection.x >= 0 && intersection.y >= 0) {
    cv::circle(viz_image, cv::Point(cvRound(intersection.x), cvRound(intersection.y)),
              circle_radius_, cv::Scalar(0, 0, 255), -1);  // 赤
  }


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