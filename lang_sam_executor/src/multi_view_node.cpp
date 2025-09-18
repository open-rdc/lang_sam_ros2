/*
 * マルチビューノード C++実装
 * 4分割画像表示ノード（高性能C++版）
 * 
 * 技術的目的:
 * - GroundingDINO/CSRT/SAM2の結果を統合表示する目的で使用
 * - 高速画像合成とリアルタイム配信を実現する目的でC++実装
 * - ROS2メッセージングでモジュラーシステムを構築する目的で実装
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include <memory>
#include <string>
#include <map>
#include <chrono>
#include <deque>

// 频度モニタリングクラス
// 目的: 各視覚化トピックのフレームレートをリアルタイムで計測
class FrequencyMonitor {
public:
    FrequencyMonitor(size_t window_size = 300) : window_size_(window_size) {}
    
    void update() {
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        timestamps_.push_back(now);
        
        // 古いタイムスタンプを削除
        // 目的: メモリ使用量を制限して安定動作を維持
        if (timestamps_.size() > window_size_) {
            timestamps_.pop_front();
        }
        
        // 周波数計算
        // 目的: リアルタイムでシステムのパフォーマンスを監視
        if (timestamps_.size() >= 2) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamps_.back() - timestamps_.front()).count();
            if (duration > 0) {
                frequency_ = static_cast<double>(timestamps_.size() - 1) * 1000.0 / duration;
            }
        }
    }
    
    double getFrequency() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frequency_;
    }
    
private:
    size_t window_size_;
    std::deque<std::chrono::steady_clock::time_point> timestamps_;
    double frequency_ = 0.0;
    mutable std::mutex mutex_;
};

class MultiViewNode : public rclcpp::Node {
public:
    MultiViewNode() : Node("multi_view_node") {
        // パラメータ宣言
        declare_parameters();
        load_parameters();
        
        // 周波数監視初期化
        frequency_monitors_["upper_left"] = std::make_unique<FrequencyMonitor>();
        frequency_monitors_["upper_right"] = std::make_unique<FrequencyMonitor>();
        frequency_monitors_["lower_left"] = std::make_unique<FrequencyMonitor>();
        frequency_monitors_["lower_right"] = std::make_unique<FrequencyMonitor>();
        
        // サブスクライバー作成
        setup_communication();
        
        // タイマー作成（指定FPSで統合画像配信）
        timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / output_fps_)),
            std::bind(&MultiViewNode::publish_multi_view, this)
        );
        
        RCLCPP_INFO(get_logger(), "MultiViewNode (C++) initialized successfully");
    }

private:
    void declare_parameters() {
        // 4分割画像配置設定
        declare_parameter("upper_left_topic", "/zed/zed_node/rgb/image_rect_color");
        declare_parameter("upper_right_topic", "/image_gdino");
        declare_parameter("lower_left_topic", "/image_csrt");
        declare_parameter("lower_right_topic", "/image_sam");
        
        // 出力設定
        declare_parameter("output_fps", 30.0);
        declare_parameter("output_width", 1280);
        declare_parameter("output_height", 720);
        declare_parameter("border_width", 2);
        declare_parameter("border_color", std::vector<int64_t>{255, 255, 255});
        
        // ラベル設定
        declare_parameter("enable_labels", true);
        declare_parameter("label_font_scale", 0.7);
        declare_parameter("label_thickness", 2);
        declare_parameter("label_color", std::vector<int64_t>{0, 255, 0});
    }
    
    void load_parameters() {
        upper_left_topic_ = get_parameter("upper_left_topic").as_string();
        upper_right_topic_ = get_parameter("upper_right_topic").as_string();
        lower_left_topic_ = get_parameter("lower_left_topic").as_string();
        lower_right_topic_ = get_parameter("lower_right_topic").as_string();
        
        output_fps_ = get_parameter("output_fps").as_double();
        output_width_ = get_parameter("output_width").as_int();
        output_height_ = get_parameter("output_height").as_int();
        border_width_ = get_parameter("border_width").as_int();
        
        auto border_color_param = get_parameter("border_color").as_integer_array();
        border_color_ = cv::Scalar(border_color_param[0], border_color_param[1], border_color_param[2]);
        
        enable_labels_ = get_parameter("enable_labels").as_bool();
        label_font_scale_ = get_parameter("label_font_scale").as_double();
        label_thickness_ = get_parameter("label_thickness").as_int();
        
        auto label_color_param = get_parameter("label_color").as_integer_array();
        label_color_ = cv::Scalar(label_color_param[0], label_color_param[1], label_color_param[2]);
        
        // 計算値
        single_width_ = (output_width_ - 3 * border_width_) / 2;
        single_height_ = (output_height_ - 3 * border_width_) / 2;
    }
    
    void setup_communication() {
        // サブスクライバー
        upper_left_sub_ = create_subscription<sensor_msgs::msg::Image>(
            upper_left_topic_, 10,
            std::bind(&MultiViewNode::upper_left_callback, this, std::placeholders::_1)
        );
        
        upper_right_sub_ = create_subscription<sensor_msgs::msg::Image>(
            upper_right_topic_, 10,
            std::bind(&MultiViewNode::upper_right_callback, this, std::placeholders::_1)
        );
        
        lower_left_sub_ = create_subscription<sensor_msgs::msg::Image>(
            lower_left_topic_, 10,
            std::bind(&MultiViewNode::lower_left_callback, this, std::placeholders::_1)
        );
        
        lower_right_sub_ = create_subscription<sensor_msgs::msg::Image>(
            lower_right_topic_, 10,
            std::bind(&MultiViewNode::lower_right_callback, this, std::placeholders::_1)
        );
        
        // パブリッシャー
        multi_view_pub_ = create_publisher<sensor_msgs::msg::Image>("/multi_view", 10);
    }
    
    void upper_left_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            images_["upper_left"] = cv_ptr->image.clone();
            frequency_monitors_["upper_left"]->update();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void upper_right_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            images_["upper_right"] = cv_ptr->image.clone();
            frequency_monitors_["upper_right"]->update();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void lower_left_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            images_["lower_left"] = cv_ptr->image.clone();
            frequency_monitors_["lower_left"]->update();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void lower_right_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            images_["lower_right"] = cv_ptr->image.clone();
            frequency_monitors_["lower_right"]->update();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    cv::Mat resize_image(const cv::Mat& image) const {
        if (image.empty()) {
            return cv::Mat::zeros(single_height_, single_width_, CV_8UC3);
        }
        
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(single_width_, single_height_));
        return resized;
    }
    
    void add_label(cv::Mat& image, const std::string& text, const std::string& position) const {
        if (!enable_labels_) return;
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                           label_font_scale_, label_thickness_, &baseline);
        
        cv::Point org;
        if (position == "top_left") {
            org = cv::Point(10, 30);
        } else if (position == "top_right") {
            org = cv::Point(single_width_ - text_size.width - 10, 30);
        } else if (position == "bottom_left") {
            org = cv::Point(10, single_height_ - 10);
        } else { // bottom_right
            org = cv::Point(single_width_ - text_size.width - 10, single_height_ - 10);
        }
        
        cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, 
                   label_font_scale_, label_color_, label_thickness_);
    }
    
    void publish_multi_view() {
        try {
            cv::Mat multi_view;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                
                // 各画像をリサイズ
                cv::Mat upper_left = resize_image(images_["upper_left"]);
                cv::Mat upper_right = resize_image(images_["upper_right"]);
                cv::Mat lower_left = resize_image(images_["lower_left"]);
                cv::Mat lower_right = resize_image(images_["lower_right"]);
                
                // ラベル追加（小数第1位まで表示）
                char ul_freq[20], ur_freq[20], ll_freq[20], lr_freq[20];
                sprintf(ul_freq, "%.1f", frequency_monitors_["upper_left"]->getFrequency());
                sprintf(ur_freq, "%.1f", frequency_monitors_["upper_right"]->getFrequency());
                sprintf(ll_freq, "%.1f", frequency_monitors_["lower_left"]->getFrequency());
                sprintf(lr_freq, "%.1f", frequency_monitors_["lower_right"]->getFrequency());
                
                std::string ul_label = upper_left_topic_ + " (" + ul_freq + "Hz)";
                std::string ur_label = upper_right_topic_ + " (" + ur_freq + "Hz)";
                std::string ll_label = lower_left_topic_ + " (" + ll_freq + "Hz)";
                std::string lr_label = lower_right_topic_ + " (" + lr_freq + "Hz)";
                
                add_label(upper_left, ul_label, "top_left");
                add_label(upper_right, ur_label, "top_right");
                add_label(lower_left, ll_label, "bottom_left");
                add_label(lower_right, lr_label, "bottom_right");
                
                // 統合画像作成
                multi_view = cv::Mat(output_height_, output_width_, CV_8UC3, border_color_);
                
                // 2x2グリッド配置
                int x1 = border_width_;
                int y1 = border_width_;
                int x2 = border_width_ + single_width_ + border_width_;
                int y2 = border_width_ + single_height_ + border_width_;
                
                upper_left.copyTo(multi_view(cv::Rect(x1, y1, single_width_, single_height_)));
                upper_right.copyTo(multi_view(cv::Rect(x2, y1, single_width_, single_height_)));
                lower_left.copyTo(multi_view(cv::Rect(x1, y2, single_width_, single_height_)));
                lower_right.copyTo(multi_view(cv::Rect(x2, y2, single_width_, single_height_)));
            }
            
            // ROS2メッセージに変換して配信
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", multi_view).toImageMsg();
            msg->header.stamp = now();
            msg->header.frame_id = "multi_view";
            multi_view_pub_->publish(*msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "MultiView publish error: %s", e.what());
        }
    }
    
private:
    // パラメータ
    std::string upper_left_topic_, upper_right_topic_, lower_left_topic_, lower_right_topic_;
    double output_fps_;
    int output_width_, output_height_, border_width_;
    int single_width_, single_height_;
    cv::Scalar border_color_, label_color_;
    bool enable_labels_;
    double label_font_scale_;
    int label_thickness_;
    
    // ROS2通信
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr upper_left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr upper_right_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr lower_left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr lower_right_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr multi_view_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // データ管理
    std::map<std::string, cv::Mat> images_;
    std::map<std::string, std::unique_ptr<FrequencyMonitor>> frequency_monitors_;
    std::mutex mutex_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiViewNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}