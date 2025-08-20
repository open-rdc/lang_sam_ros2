#include "csrt_tracker.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>

namespace csrt_native {

CSRTTrackerNative::CSRTTrackerNative(const std::string& tracker_id, const CSRTParams& params)
    : tracker_id_(tracker_id), params_(params), initialized_(false) {
    create_tracker_with_params();
}

CSRTTrackerNative::~CSRTTrackerNative() = default;

void CSRTTrackerNative::create_tracker_with_params() {
    try {
        // Create CSRT parameters object
        cv::TrackerCSRT::Params csrt_params;
        
        // Apply custom parameters from CSRTParams struct
        csrt_params.use_hog = params_.use_hog;
        csrt_params.use_color_names = params_.use_color_names;
        csrt_params.use_gray = params_.use_gray;
        csrt_params.use_rgb = params_.use_rgb;
        csrt_params.use_channel_weights = params_.use_channel_weights;
        csrt_params.use_segmentation = params_.use_segmentation;
        
        // Window function settings
        csrt_params.window_function = params_.window_function;
        csrt_params.kaiser_alpha = params_.kaiser_alpha;
        csrt_params.cheb_attenuation = params_.cheb_attenuation;
        
        // Template and filter parameters
        csrt_params.template_size = params_.template_size;
        csrt_params.gsl_sigma = params_.gsl_sigma;
        csrt_params.hog_orientations = params_.hog_orientations;
        csrt_params.hog_clip = params_.hog_clip;
        csrt_params.padding = params_.padding;
        csrt_params.filter_lr = params_.filter_lr;
        csrt_params.weights_lr = params_.weights_lr;
        
        // Advanced parameters
        csrt_params.num_hog_channels_used = params_.num_hog_channels_used;
        csrt_params.admm_iterations = params_.admm_iterations;
        csrt_params.histogram_bins = params_.histogram_bins;
        csrt_params.histogram_lr = params_.histogram_lr;
        
        // Scale and segmentation parameters
        csrt_params.background_ratio = params_.background_ratio;
        csrt_params.number_of_scales = params_.number_of_scales;
        csrt_params.scale_sigma_factor = params_.scale_sigma_factor;
        csrt_params.scale_model_max_area = params_.scale_model_max_area;
        csrt_params.scale_lr = params_.scale_lr;
        csrt_params.scale_step = params_.scale_step;
        csrt_params.psr_threshold = params_.psr_threshold;
        
        // Create tracker with parameters
        tracker_ = cv::TrackerCSRT::create(csrt_params);
        
        if (tracker_) {
            std::cout << "[" << tracker_id_ << "] CSRT tracker created with custom parameters" << std::endl;
        } else {
            // Fallback to legacy API with parameters
            cv::legacy::TrackerCSRT::Params legacy_params;
            
            // Copy parameters to legacy format
            legacy_params.use_hog = params_.use_hog;
            legacy_params.use_color_names = params_.use_color_names;
            legacy_params.use_gray = params_.use_gray;
            legacy_params.use_rgb = params_.use_rgb;
            legacy_params.use_channel_weights = params_.use_channel_weights;
            legacy_params.use_segmentation = params_.use_segmentation;
            legacy_params.window_function = params_.window_function;
            legacy_params.kaiser_alpha = params_.kaiser_alpha;
            legacy_params.cheb_attenuation = params_.cheb_attenuation;
            legacy_params.template_size = params_.template_size;
            legacy_params.gsl_sigma = params_.gsl_sigma;
            legacy_params.hog_orientations = params_.hog_orientations;
            legacy_params.hog_clip = params_.hog_clip;
            legacy_params.padding = params_.padding;
            legacy_params.filter_lr = params_.filter_lr;
            legacy_params.weights_lr = params_.weights_lr;
            legacy_params.num_hog_channels_used = params_.num_hog_channels_used;
            legacy_params.admm_iterations = params_.admm_iterations;
            legacy_params.histogram_bins = params_.histogram_bins;
            legacy_params.histogram_lr = params_.histogram_lr;
            legacy_params.background_ratio = params_.background_ratio;
            legacy_params.number_of_scales = params_.number_of_scales;
            legacy_params.scale_sigma_factor = params_.scale_sigma_factor;
            legacy_params.scale_model_max_area = params_.scale_model_max_area;
            legacy_params.scale_lr = params_.scale_lr;
            legacy_params.scale_step = params_.scale_step;
            legacy_params.psr_threshold = params_.psr_threshold;
            
            auto legacy_tracker = cv::legacy::TrackerCSRT::create(legacy_params);
            if (legacy_tracker) {
                tracker_ = cv::legacy::upgradeTrackingAPI(legacy_tracker);
                std::cout << "[" << tracker_id_ << "] CSRT tracker created with legacy API and custom parameters" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << tracker_id_ << "] Exception creating CSRT tracker: " << e.what() << std::endl;
        tracker_ = nullptr;
    }
}

bool CSRTTrackerNative::initialize(const cv::Mat& image, const cv::Rect2d& bbox) {
    if (!tracker_ || image.empty()) {
        return false;
    }
    
    try {
        // Validate bbox
        if (bbox.width <= 2 || bbox.height <= 2 || 
            bbox.x < 0 || bbox.y < 0 ||
            bbox.x + bbox.width > image.cols || 
            bbox.y + bbox.height > image.rows) {
            std::cerr << "[" << tracker_id_ << "] Invalid bbox: " 
                      << bbox.x << "," << bbox.y << " " << bbox.width << "x" << bbox.height << std::endl;
            return false;
        }
        
        // Initialize tracker (OpenCV 4.5+ init() returns void)
        tracker_->init(image, bbox);
        initialized_ = true;
        
        return initialized_;
    } catch (const std::exception& e) {
        std::cerr << "[" << tracker_id_ << "] Initialization error: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}

bool CSRTTrackerNative::update(const cv::Mat& image, cv::Rect2d& bbox) {
    if (!initialized_ || !tracker_ || image.empty()) {
        return false;
    }
    
    try {
        // OpenCV 4.5+ update() requires cv::Rect (int) not cv::Rect2d (double)
        cv::Rect int_bbox(static_cast<int>(bbox.x), static_cast<int>(bbox.y), 
                         static_cast<int>(bbox.width), static_cast<int>(bbox.height));
        bool success = tracker_->update(image, int_bbox);
        
        // Convert back to double precision
        bbox.x = static_cast<double>(int_bbox.x);
        bbox.y = static_cast<double>(int_bbox.y);
        bbox.width = static_cast<double>(int_bbox.width);
        bbox.height = static_cast<double>(int_bbox.height);
        
        if (success) {
            // Validate updated bbox
            if (bbox.width > 2 && bbox.height > 2 &&
                bbox.x >= 0 && bbox.y >= 0) {
                
                // Clip to image boundaries
                bbox.x = std::max(0.0, bbox.x);
                bbox.y = std::max(0.0, bbox.y);
                bbox.width = std::min(bbox.width, static_cast<double>(image.cols) - bbox.x);
                bbox.height = std::min(bbox.height, static_cast<double>(image.rows) - bbox.y);
                
                return (bbox.width > 2 && bbox.height > 2);
            }
        }
        
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[" << tracker_id_ << "] Update error: " << e.what() << std::endl;
        return false;
    }
}

void CSRTTrackerNative::set_params(const CSRTParams& params) {
    params_ = params;
    
    // Log parameter changes for verification
    std::cout << "[" << tracker_id_ << "] CSRT Parameters Updated:" << std::endl;
    std::cout << "  use_hog: " << params_.use_hog << std::endl;
    std::cout << "  use_color_names: " << params_.use_color_names << std::endl;
    std::cout << "  use_gray: " << params_.use_gray << std::endl;
    std::cout << "  use_rgb: " << params_.use_rgb << std::endl;
    std::cout << "  use_channel_weights: " << params_.use_channel_weights << std::endl;
    std::cout << "  use_segmentation: " << params_.use_segmentation << std::endl;
    std::cout << "  window_function: " << params_.window_function << std::endl;
    std::cout << "  kaiser_alpha: " << params_.kaiser_alpha << std::endl;
    std::cout << "  cheb_attenuation: " << params_.cheb_attenuation << std::endl;
    std::cout << "  template_size: " << params_.template_size << std::endl;
    std::cout << "  gsl_sigma: " << params_.gsl_sigma << std::endl;
    std::cout << "  hog_orientations: " << params_.hog_orientations << std::endl;
    std::cout << "  hog_clip: " << params_.hog_clip << std::endl;
    std::cout << "  padding: " << params_.padding << std::endl;
    std::cout << "  filter_lr: " << params_.filter_lr << std::endl;
    std::cout << "  weights_lr: " << params_.weights_lr << std::endl;
    std::cout << "  num_hog_channels_used: " << params_.num_hog_channels_used << std::endl;
    std::cout << "  admm_iterations: " << params_.admm_iterations << std::endl;
    std::cout << "  histogram_bins: " << params_.histogram_bins << std::endl;
    std::cout << "  histogram_lr: " << params_.histogram_lr << std::endl;
    std::cout << "  background_ratio: " << params_.background_ratio << std::endl;
    std::cout << "  number_of_scales: " << params_.number_of_scales << std::endl;
    std::cout << "  scale_sigma_factor: " << params_.scale_sigma_factor << std::endl;
    std::cout << "  scale_model_max_area: " << params_.scale_model_max_area << std::endl;
    std::cout << "  scale_lr: " << params_.scale_lr << std::endl;
    std::cout << "  scale_step: " << params_.scale_step << std::endl;
    std::cout << "  psr_threshold: " << params_.psr_threshold << std::endl;
    
    // For OpenCV 4.5+, parameters need to be set during creation
    // Re-create the tracker with new parameters
    create_tracker_with_params();
    std::cout << "[" << tracker_id_ << "] Native C++ CSRT tracker recreated with updated parameters" << std::endl;
}

// CSRTManagerNative Implementation
CSRTManagerNative::CSRTManagerNative(const CSRTParams& default_params)
    : default_params_(default_params), next_tracker_id_(1), bbox_min_size_(3), bbox_margin_(5) {
    std::cout << "CSRTManagerNative initialized with OpenCV 4.5+ compatible API" << std::endl;
}

CSRTManagerNative::~CSRTManagerNative() = default;

std::vector<cv::Rect2d> CSRTManagerNative::process_detections(
    const cv::Mat& image, 
    const std::vector<cv::Rect2d>& detections,
    const std::vector<std::string>& labels) {
    
    if (image.empty()) {
        return {};
    }
    
    // Clear existing trackers and create new ones
    trackers_.clear();
    tracker_labels_.clear();
    tracker_order_.clear();  // Clear order tracking
    
    std::vector<cv::Rect2d> results;
    
    std::cout << "[C++ CSRTManager] Processing " << detections.size() << " detections with labels: ";
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "'" << labels[i] << "'" << (i < labels.size()-1 ? ", " : "");
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& bbox = detections[i];
        std::string label = (i < labels.size()) ? labels[i] : "unknown";
        
        // Clip and validate bbox
        cv::Rect2d clipped_bbox = clip_bbox(bbox, image.size());
        if (!is_valid_bbox(clipped_bbox, image.size())) {
            std::cout << "[C++ CSRTManager] Skipping invalid bbox for label: " << label << std::endl;
            continue;
        }
        
        // Create tracker with consistent ID generation
        std::string tracker_id = generate_tracker_id(label);
        auto tracker = std::make_shared<CSRTTrackerNative>(tracker_id, default_params_);
        
        if (tracker->initialize(image, clipped_bbox)) {
            trackers_[tracker_id] = tracker;
            tracker_labels_[tracker_id] = label;
            tracker_order_.push_back(tracker_id);  // Maintain order
            results.push_back(clipped_bbox);
            
            std::cout << "[C++ CSRTManager] Created tracker: " << tracker_id << " for label: " << label << std::endl;
        }
    }
    
    std::cout << "[C++ CSRTManager] Successfully created " << trackers_.size() << " CSRT trackers" << std::endl;
    return results;
}

std::vector<cv::Rect2d> CSRTManagerNative::update_trackers(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    
    std::vector<cv::Rect2d> results;
    std::vector<std::string> failed_trackers;
    
    // Process trackers in order to maintain label consistency
    for (const std::string& tracker_id : tracker_order_) {
        auto tracker_it = trackers_.find(tracker_id);
        if (tracker_it == trackers_.end()) {
            continue;  // Tracker was already removed
        }
        
        auto& tracker = tracker_it->second;
        cv::Rect2d bbox;
        
        if (tracker->update(image, bbox)) {
            results.push_back(bbox);
        } else {
            failed_trackers.push_back(tracker_id);
        }
    }
    
    // Remove failed trackers and maintain order
    for (const auto& tracker_id : failed_trackers) {
        trackers_.erase(tracker_id);
        tracker_labels_.erase(tracker_id);
        
        // Remove from order tracking
        tracker_order_.erase(
            std::remove(tracker_order_.begin(), tracker_order_.end(), tracker_id),
            tracker_order_.end()
        );
        
        std::cout << "[C++ CSRTManager] Removed failed tracker: " << tracker_id << std::endl;
    }
    
    if (!tracker_order_.empty()) {
        std::cout << "[C++ CSRTManager] Updated " << tracker_order_.size() << " trackers, results: " << results.size() << std::endl;
    }
    
    return results;
}

void CSRTManagerNative::clear_trackers() {
    trackers_.clear();
    tracker_labels_.clear();
    tracker_order_.clear();
}

std::vector<std::string> CSRTManagerNative::get_tracker_labels() const {
    std::vector<std::string> labels;
    
    // Return labels in the same order as trackers were created
    for (const std::string& tracker_id : tracker_order_) {
        auto label_it = tracker_labels_.find(tracker_id);
        if (label_it != tracker_labels_.end()) {
            // Extract clean label from tracker_id (remove _number suffix)
            std::string clean_label = extract_clean_label(tracker_id);
            labels.push_back(clean_label);
        }
    }
    
    std::cout << "[C++ CSRTManager] Returning " << labels.size() << " labels in order: ";
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "'" << labels[i] << "'" << (i < labels.size()-1 ? ", " : "");
    }
    std::cout << std::endl;
    
    return labels;
}

void CSRTManagerNative::set_tracker_params(const std::string& tracker_id, const CSRTParams& params) {
    auto it = trackers_.find(tracker_id);
    if (it != trackers_.end()) {
        it->second->set_params(params);
    }
}

cv::Rect2d CSRTManagerNative::clip_bbox(const cv::Rect2d& bbox, const cv::Size& image_size) {
    cv::Rect2d clipped = bbox;
    
    // Apply clipping
    clipped.x = std::max(0.0, std::min(clipped.x, static_cast<double>(image_size.width - 3)));
    clipped.y = std::max(0.0, std::min(clipped.y, static_cast<double>(image_size.height - 3)));
    clipped.width = std::min(clipped.width, static_cast<double>(image_size.width) - clipped.x);
    clipped.height = std::min(clipped.height, static_cast<double>(image_size.height) - clipped.y);
    
    return clipped;
}

bool CSRTManagerNative::is_valid_bbox(const cv::Rect2d& bbox, const cv::Size& image_size) {
    return (bbox.width >= bbox_min_size_ && bbox.height >= bbox_min_size_ &&
            bbox.x >= 0 && bbox.y >= 0 &&
            bbox.x + bbox.width <= image_size.width &&
            bbox.y + bbox.height <= image_size.height);
}

std::string CSRTManagerNative::generate_tracker_id(const std::string& label) {
    return label + "_" + std::to_string(next_tracker_id_++);
}

std::string CSRTManagerNative::extract_clean_label(const std::string& tracker_id) const {
    // Extract label from tracker_id by removing the "_number" suffix
    size_t last_underscore = tracker_id.find_last_of('_');
    if (last_underscore != std::string::npos) {
        std::string suffix = tracker_id.substr(last_underscore + 1);
        // Check if suffix is a number
        bool is_number = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
        if (is_number) {
            return tracker_id.substr(0, last_underscore);
        }
    }
    // If no valid numeric suffix found, return the whole string
    return tracker_id;
}

} // namespace csrt_native