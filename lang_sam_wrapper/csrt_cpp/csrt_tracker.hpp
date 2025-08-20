#ifndef CSRT_TRACKER_HPP
#define CSRT_TRACKER_HPP

#include <opencv2/opencv.hpp>
// Include tracking headers in correct order for OpenCV 4.5+
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace csrt_native {

struct CSRTParams {
    // Core CSRT parameters from config.yaml
    bool use_hog = true;
    bool use_color_names = true;
    bool use_gray = true;
    bool use_rgb = false;
    bool use_channel_weights = true;
    bool use_segmentation = true;
    
    // Window function parameters
    std::string window_function = "hann";
    float kaiser_alpha = 2.0f;
    float cheb_attenuation = 45.0f;
    
    // Template parameters
    float template_size = 200.0f;
    float gsl_sigma = 1.0f;
    int hog_orientations = 9;
    float hog_clip = 0.2f;
    
    // Filter parameters
    float padding = 3.0f;
    float filter_lr = 0.02f;
    float weights_lr = 0.02f;
    
    // Advanced parameters
    int num_hog_channels_used = 18;
    int admm_iterations = 4;
    int histogram_bins = 16;
    float histogram_lr = 0.04f;
    
    // Segmentation parameters
    int background_ratio = 2;
    int number_of_scales = 33;
    float scale_sigma_factor = 0.25f;
    float scale_model_max_area = 512.0f;
    float scale_lr = 0.025f;
    float scale_step = 1.02f;
    float psr_threshold = 0.035f;
};

class CSRTTrackerNative {
public:
    CSRTTrackerNative(const std::string& tracker_id, const CSRTParams& params = CSRTParams{});
    ~CSRTTrackerNative();
    
    bool initialize(const cv::Mat& image, const cv::Rect2d& bbox);
    bool update(const cv::Mat& image, cv::Rect2d& bbox);
    bool is_initialized() const { return initialized_; }
    std::string get_tracker_id() const { return tracker_id_; }
    
    void set_params(const CSRTParams& params);
    CSRTParams get_params() const { return params_; }
    
private:
    std::string tracker_id_;
    // Use standard cv::Tracker interface for OpenCV 4.5+
    cv::Ptr<cv::Tracker> tracker_;
    CSRTParams params_;
    bool initialized_;
    
    void create_tracker_with_params();
};

class CSRTManagerNative {
public:
    CSRTManagerNative(const CSRTParams& default_params = CSRTParams{});
    ~CSRTManagerNative();
    
    // Main interface
    std::vector<cv::Rect2d> process_detections(
        const cv::Mat& image, 
        const std::vector<cv::Rect2d>& detections,
        const std::vector<std::string>& labels
    );
    
    std::vector<cv::Rect2d> update_trackers(const cv::Mat& image);
    
    // Management
    void clear_trackers();
    size_t get_tracker_count() const { return trackers_.size(); }
    std::vector<std::string> get_tracker_labels() const;
    
    // Configuration
    void set_default_params(const CSRTParams& params) { default_params_ = params; }
    void set_tracker_params(const std::string& tracker_id, const CSRTParams& params);
    
    // Bounding box utilities
    void set_bbox_min_size(int min_size) { bbox_min_size_ = min_size; }
    void set_bbox_margin(int margin) { bbox_margin_ = margin; }
    
private:
    std::unordered_map<std::string, std::shared_ptr<CSRTTrackerNative>> trackers_;
    std::unordered_map<std::string, std::string> tracker_labels_;
    std::vector<std::string> tracker_order_;  // Maintain insertion order
    CSRTParams default_params_;
    int next_tracker_id_;
    
    // Configuration
    int bbox_min_size_;
    int bbox_margin_;
    
    // Utility methods
    cv::Rect2d clip_bbox(const cv::Rect2d& bbox, const cv::Size& image_size);
    bool is_valid_bbox(const cv::Rect2d& bbox, const cv::Size& image_size);
    std::string generate_tracker_id(const std::string& label);
    std::string extract_clean_label(const std::string& tracker_id) const;
};

} // namespace csrt_native

#endif // CSRT_TRACKER_HPP