#include "image_processor.hpp"
#include <algorithm>
#include <chrono>
#include <functional>

namespace fast_processing {

// ImageCache implementation
ImageCache::ImageCache(size_t max_size) : max_cache_size_(max_size) {}

size_t ImageCache::compute_hash(const cv::Mat& image) const {
    // 高速ハッシュ計算（画像データの一部をサンプリング）
    std::hash<std::string> hasher;
    std::string data_sample;
    
    if (!image.empty() && image.isContinuous()) {
        size_t step = std::max(1, static_cast<int>(image.total() / 1000));
        const uchar* ptr = image.ptr<uchar>();
        
        for (size_t i = 0; i < image.total(); i += step) {
            data_sample.push_back(static_cast<char>(ptr[i]));
        }
        
        // サイズと型も含める
        data_sample += std::to_string(image.rows) + "x" + std::to_string(image.cols) 
                      + "t" + std::to_string(image.type());
    }
    
    return hasher(data_sample);
}

void ImageCache::cleanup_cache() {
    while (access_order_.size() > max_cache_size_) {
        size_t oldest_hash = access_order_.front();
        access_order_.erase(access_order_.begin());
        
        bgr_cache_.erase(oldest_hash);
        rgb_cache_.erase(oldest_hash);
    }
}

cv::Mat ImageCache::get_rgb_image(const cv::Mat& bgr_image) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    size_t hash = compute_hash(bgr_image);
    
    // キャッシュヒット確認
    auto it = rgb_cache_.find(hash);
    if (it != rgb_cache_.end()) {
        // アクセス順序を更新
        auto order_it = std::find(access_order_.begin(), access_order_.end(), hash);
        if (order_it != access_order_.end()) {
            access_order_.erase(order_it);
        }
        access_order_.push_back(hash);
        return it->second;
    }
    
    // キャッシュミス：新しい変換
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // キャッシュに保存
    bgr_cache_[hash] = bgr_image.clone();
    rgb_cache_[hash] = rgb_image.clone();
    access_order_.push_back(hash);
    
    // キャッシュサイズ管理
    cleanup_cache();
    
    return rgb_image;
}

void ImageCache::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    bgr_cache_.clear();
    rgb_cache_.clear();
    access_order_.clear();
}

size_t ImageCache::cache_size() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return bgr_cache_.size();
}

// FastImageProcessor implementation
FastImageProcessor::FastImageProcessor() 
    : image_cache_(std::make_unique<ImageCache>(15)),
      default_color_(0, 255, 0),  // 緑色
      default_thickness_(2) {}

FastImageProcessor::~FastImageProcessor() = default;

cv::Mat FastImageProcessor::bgr_to_rgb_cached(const cv::Mat& bgr_image) {
    return image_cache_->get_rgb_image(bgr_image);
}

cv::Mat FastImageProcessor::rgb_to_bgr_cached(const cv::Mat& rgb_image) {
    cv::Mat bgr_result;
    cv::cvtColor(rgb_image, bgr_result, cv::COLOR_RGB2BGR);
    return bgr_result;
}

std::vector<cv::Rect> FastImageProcessor::convert_detections_fast(
    const std::vector<std::vector<int>>& detections) {
    
    std::vector<cv::Rect> boxes;
    boxes.reserve(detections.size());
    
    for (const auto& det : detections) {
        if (det.size() >= 4) {
            // xyxy形式からxywh形式に変換
            int x = det[0];
            int y = det[1];
            int width = det[2] - det[0];
            int height = det[3] - det[1];
            
            boxes.emplace_back(x, y, width, height);
        }
    }
    
    return boxes;
}

cv::Mat FastImageProcessor::draw_boxes_fast(
    const cv::Mat& image,
    const std::vector<BoundingBox>& boxes,
    const cv::Scalar& color,
    int thickness) {
    
    if (image.empty()) return cv::Mat();
    
    cv::Mat result = image.clone();
    
    for (const auto& box : boxes) {
        // バウンディングボックス描画
        cv::Rect rect(box.x, box.y, box.width, box.height);
        cv::rectangle(result, rect, color, thickness);
        
        // ラベル描画（空でない場合）
        if (!box.label.empty()) {
            std::string display_text = box.label;
            if (box.confidence > 0.0f && box.confidence < 1.0f) {
                display_text += " (" + std::to_string(static_cast<int>(box.confidence * 100)) + "%)";
            }
            
            // テキスト背景
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(display_text, font_face, font_scale, 1, &baseline);
            
            cv::Point text_origin(box.x, box.y - 5);
            if (text_origin.y < text_size.height) {
                text_origin.y = box.y + box.height + text_size.height + 5;
            }
            
            // 背景矩形
            cv::rectangle(result, 
                         text_origin + cv::Point(0, baseline),
                         text_origin + cv::Point(text_size.width, -text_size.height),
                         color, cv::FILLED);
            
            // テキスト
            cv::putText(result, display_text, text_origin, font_face, font_scale, 
                       cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
    
    return result;
}

std::vector<cv::Mat> FastImageProcessor::process_image_batch(
    const std::vector<cv::Mat>& images, bool to_rgb) {
    
    std::vector<cv::Mat> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        if (to_rgb) {
            results.push_back(bgr_to_rgb_cached(image));
        } else {
            results.push_back(image.clone());
        }
    }
    
    return results;
}

cv::Mat FastImageProcessor::resize_optimized(
    const cv::Mat& image, const cv::Size& target_size, int interpolation) {
    
    if (image.empty()) return cv::Mat();
    if (image.size() == target_size) return image.clone();
    
    cv::Mat resized;
    cv::resize(image, resized, target_size, 0, 0, interpolation);
    return resized;
}

size_t FastImageProcessor::get_cache_size() const {
    return image_cache_->cache_size();
}

void FastImageProcessor::clear_all_cache() {
    image_cache_->clear_cache();
}

// Utility functions
std::vector<std::vector<float>> boxes_to_xyxy(const std::vector<BoundingBox>& boxes) {
    std::vector<std::vector<float>> xyxy_data;
    xyxy_data.reserve(boxes.size());
    
    for (const auto& box : boxes) {
        xyxy_data.push_back({
            static_cast<float>(box.x),                    // x1
            static_cast<float>(box.y),                    // y1
            static_cast<float>(box.x + box.width),        // x2
            static_cast<float>(box.y + box.height)        // y2
        });
    }
    
    return xyxy_data;
}

std::vector<BoundingBox> xyxy_to_boxes(const std::vector<std::vector<float>>& xyxy_data,
                                       const std::vector<std::string>& labels) {
    std::vector<BoundingBox> boxes;
    boxes.reserve(xyxy_data.size());
    
    for (size_t i = 0; i < xyxy_data.size(); ++i) {
        if (xyxy_data[i].size() >= 4) {
            int x1 = static_cast<int>(xyxy_data[i][0]);
            int y1 = static_cast<int>(xyxy_data[i][1]);
            int x2 = static_cast<int>(xyxy_data[i][2]);
            int y2 = static_cast<int>(xyxy_data[i][3]);
            
            std::string label = (i < labels.size()) ? labels[i] : "";
            
            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1, label);
        }
    }
    
    return boxes;
}

} // namespace fast_processing