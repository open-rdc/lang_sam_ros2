#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "csrt_tracker.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    
    // Handle different image formats
    if (buf_info.ndim == 3 && buf_info.shape[2] == 3) {
        // BGR image
        cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
        return mat.clone();  // Make a copy for safety
    } else if (buf_info.ndim == 2) {
        // Grayscale image
        cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC1, (unsigned char*)buf_info.ptr);
        return mat.clone();
    }
    
    // Default case
    cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
    return mat.clone();
}

// Helper function to convert cv::Rect2d to tuple
py::tuple rect_to_tuple(const cv::Rect2d& rect) {
    return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
}

// Helper function to convert tuple to cv::Rect2d
cv::Rect2d tuple_to_rect(py::tuple t) {
    return cv::Rect2d(t[0].cast<double>(), t[1].cast<double>(), 
                      t[2].cast<double>(), t[3].cast<double>());
}

PYBIND11_MODULE(csrt_native, m) {
    m.doc() = "Native C++ CSRT tracker with OpenCV 4.5+ compatibility";
    
    // CSRTParams struct
    py::class_<csrt_native::CSRTParams>(m, "CSRTParams")
        .def(py::init<>())
        .def_readwrite("use_hog", &csrt_native::CSRTParams::use_hog)
        .def_readwrite("use_color_names", &csrt_native::CSRTParams::use_color_names)
        .def_readwrite("use_gray", &csrt_native::CSRTParams::use_gray)
        .def_readwrite("use_rgb", &csrt_native::CSRTParams::use_rgb)
        .def_readwrite("use_channel_weights", &csrt_native::CSRTParams::use_channel_weights)
        .def_readwrite("use_segmentation", &csrt_native::CSRTParams::use_segmentation)
        .def_readwrite("window_function", &csrt_native::CSRTParams::window_function)
        .def_readwrite("kaiser_alpha", &csrt_native::CSRTParams::kaiser_alpha)
        .def_readwrite("cheb_attenuation", &csrt_native::CSRTParams::cheb_attenuation)
        .def_readwrite("template_size", &csrt_native::CSRTParams::template_size)
        .def_readwrite("gsl_sigma", &csrt_native::CSRTParams::gsl_sigma)
        .def_readwrite("hog_orientations", &csrt_native::CSRTParams::hog_orientations)
        .def_readwrite("hog_clip", &csrt_native::CSRTParams::hog_clip)
        .def_readwrite("padding", &csrt_native::CSRTParams::padding)
        .def_readwrite("filter_lr", &csrt_native::CSRTParams::filter_lr)
        .def_readwrite("weights_lr", &csrt_native::CSRTParams::weights_lr)
        .def_readwrite("num_hog_channels_used", &csrt_native::CSRTParams::num_hog_channels_used)
        .def_readwrite("admm_iterations", &csrt_native::CSRTParams::admm_iterations)
        .def_readwrite("histogram_bins", &csrt_native::CSRTParams::histogram_bins)
        .def_readwrite("histogram_lr", &csrt_native::CSRTParams::histogram_lr)
        .def_readwrite("background_ratio", &csrt_native::CSRTParams::background_ratio)
        .def_readwrite("number_of_scales", &csrt_native::CSRTParams::number_of_scales)
        .def_readwrite("scale_sigma_factor", &csrt_native::CSRTParams::scale_sigma_factor)
        .def_readwrite("scale_model_max_area", &csrt_native::CSRTParams::scale_model_max_area)
        .def_readwrite("scale_lr", &csrt_native::CSRTParams::scale_lr)
        .def_readwrite("scale_step", &csrt_native::CSRTParams::scale_step)
        .def_readwrite("psr_threshold", &csrt_native::CSRTParams::psr_threshold);
    
    // CSRTTrackerNative
    py::class_<csrt_native::CSRTTrackerNative>(m, "CSRTTrackerNative")
        .def(py::init<const std::string&, const csrt_native::CSRTParams&>())
        .def("initialize", [](csrt_native::CSRTTrackerNative& self, py::array_t<uint8_t> image, py::tuple bbox) {
            cv::Mat mat = numpy_to_mat(image);
            cv::Rect2d rect = tuple_to_rect(bbox);
            return self.initialize(mat, rect);
        })
        .def("update", [](csrt_native::CSRTTrackerNative& self, py::array_t<uint8_t> image) -> py::object {
            cv::Mat mat = numpy_to_mat(image);
            cv::Rect2d bbox;
            bool success = self.update(mat, bbox);
            if (success) {
                // Fix: Return tuple directly instead of using py::cast
                return rect_to_tuple(bbox);
            }
            return py::none();
        })
        .def("is_initialized", &csrt_native::CSRTTrackerNative::is_initialized)
        .def("get_tracker_id", &csrt_native::CSRTTrackerNative::get_tracker_id)
        .def("set_params", &csrt_native::CSRTTrackerNative::set_params)
        .def("get_params", &csrt_native::CSRTTrackerNative::get_params);
    
    // CSRTManagerNative
    py::class_<csrt_native::CSRTManagerNative>(m, "CSRTManagerNative")
        .def(py::init<const csrt_native::CSRTParams&>())
        .def("process_detections", [](csrt_native::CSRTManagerNative& self, 
                                     py::array_t<uint8_t> image, 
                                     py::list detections, 
                                     py::list labels) -> py::list {
            cv::Mat mat = numpy_to_mat(image);
            
            std::vector<cv::Rect2d> detection_rects;
            for (auto item : detections) {
                detection_rects.push_back(tuple_to_rect(item.cast<py::tuple>()));
            }
            
            std::vector<std::string> label_strs;
            for (auto item : labels) {
                label_strs.push_back(item.cast<std::string>());
            }
            
            auto results = self.process_detections(mat, detection_rects, label_strs);
            
            py::list py_results;
            for (const auto& rect : results) {
                py_results.append(rect_to_tuple(rect));
            }
            return py_results;
        })
        .def("update_trackers", [](csrt_native::CSRTManagerNative& self, py::array_t<uint8_t> image) -> py::list {
            cv::Mat mat = numpy_to_mat(image);
            auto results = self.update_trackers(mat);
            
            py::list py_results;
            for (const auto& rect : results) {
                py_results.append(rect_to_tuple(rect));
            }
            return py_results;
        })
        .def("clear_trackers", &csrt_native::CSRTManagerNative::clear_trackers)
        .def("get_tracker_count", &csrt_native::CSRTManagerNative::get_tracker_count)
        .def("get_tracker_labels", &csrt_native::CSRTManagerNative::get_tracker_labels)
        .def("set_default_params", &csrt_native::CSRTManagerNative::set_default_params)
        .def("set_tracker_params", &csrt_native::CSRTManagerNative::set_tracker_params)
        .def("set_bbox_min_size", &csrt_native::CSRTManagerNative::set_bbox_min_size)
        .def("set_bbox_margin", &csrt_native::CSRTManagerNative::set_bbox_margin);
}