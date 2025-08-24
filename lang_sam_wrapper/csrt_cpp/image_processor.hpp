#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace fast_processing {

struct BoundingBox {
    int x, y, width, height;
    std::string label;
    float confidence;
    
    BoundingBox(int x, int y, int w, int h, const std::string& lbl = "", float conf = 1.0f)
        : x(x), y(y), width(w), height(h), label(lbl), confidence(conf) {}
};

class ImageCache {
private:
    mutable std::mutex cache_mutex_;
    std::unordered_map<size_t, cv::Mat> bgr_cache_;
    std::unordered_map<size_t, cv::Mat> rgb_cache_;
    std::vector<size_t> access_order_;
    size_t max_cache_size_;
    
    size_t compute_hash(const cv::Mat& image) const;
    void cleanup_cache();
    
public:
    explicit ImageCache(size_t max_size = 10);
    cv::Mat get_rgb_image(const cv::Mat& bgr_image);
    void clear_cache();
    size_t cache_size() const;
};

class FastImageProcessor {
private:
    std::unique_ptr<ImageCache> image_cache_;
    cv::Scalar default_color_;
    int default_thickness_;
    
public:
    FastImageProcessor();
    ~FastImageProcessor();
    
    // 高速画像変換
    cv::Mat bgr_to_rgb_cached(const cv::Mat& bgr_image);
    cv::Mat rgb_to_bgr_cached(const cv::Mat& rgb_image);
    
    // バウンディングボックス変換の高速化
    std::vector<cv::Rect> convert_detections_fast(
        const std::vector<std::vector<int>>& detections);
    
    // 可視化描画の高速化
    cv::Mat draw_boxes_fast(
        const cv::Mat& image,
        const std::vector<BoundingBox>& boxes,
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int thickness = 2);
    
    // バッチ処理用
    std::vector<cv::Mat> process_image_batch(
        const std::vector<cv::Mat>& images,
        bool to_rgb = true);
    
    // メモリ最適化されたリサイズ
    cv::Mat resize_optimized(
        const cv::Mat& image,
        const cv::Size& target_size,
        int interpolation = cv::INTER_LINEAR);
    
    // 統計情報
    size_t get_cache_size() const;
    void clear_all_cache();
};

// NumPy配列互換の高速変換関数
std::vector<std::vector<float>> boxes_to_xyxy(const std::vector<BoundingBox>& boxes);
std::vector<BoundingBox> xyxy_to_boxes(const std::vector<std::vector<float>>& xyxy_data,
                                       const std::vector<std::string>& labels = {});

} // namespace fast_processing