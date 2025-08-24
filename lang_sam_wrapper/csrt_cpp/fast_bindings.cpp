#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "image_processor.hpp"
#include "async_processor.hpp"

namespace py = pybind11;
using namespace fast_processing;

// Helper function to convert cv::Rect to Python tuple
py::tuple rect_to_tuple(const cv::Rect& rect) {
    return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
}

// Helper function to convert Python tuple to cv::Rect
cv::Rect tuple_to_rect(py::tuple t) {
    return cv::Rect(t[0].cast<int>(), t[1].cast<int>(), 
                    t[2].cast<int>(), t[3].cast<int>());
}

PYBIND11_MODULE(fast_processing, m) {
    m.doc() = "Fast Processing C++ Implementation with Image Processing and Async Support";
    
    // OpenCV enum bindings
    py::enum_<cv::InterpolationFlags>(m, "InterpolationFlags")
        .value("INTER_NEAREST", cv::INTER_NEAREST)
        .value("INTER_LINEAR", cv::INTER_LINEAR)
        .value("INTER_CUBIC", cv::INTER_CUBIC)
        .value("INTER_AREA", cv::INTER_AREA)
        .value("INTER_LANCZOS4", cv::INTER_LANCZOS4)
        .export_values();
    
    // BoundingBox struct
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<int, int, int, int, const std::string&, float>(),
             py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), 
             py::arg("label") = "", py::arg("confidence") = 1.0f)
        .def_readwrite("x", &BoundingBox::x)
        .def_readwrite("y", &BoundingBox::y)
        .def_readwrite("width", &BoundingBox::width)
        .def_readwrite("height", &BoundingBox::height)
        .def_readwrite("label", &BoundingBox::label)
        .def_readwrite("confidence", &BoundingBox::confidence);
    
    // FastImageProcessor
    py::class_<FastImageProcessor>(m, "FastImageProcessor")
        .def(py::init<>())
        .def("bgr_to_rgb_cached", [](FastImageProcessor& self, py::array_t<uint8_t> input) -> py::array_t<uint8_t> {
            py::buffer_info buf_info = input.request();
            cv::Mat image(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
            cv::Mat result = self.bgr_to_rgb_cached(image);
            return py::array_t<uint8_t>(
                {result.rows, result.cols, result.channels()},
                {sizeof(uint8_t)*result.cols*result.channels(), sizeof(uint8_t)*result.channels(), sizeof(uint8_t)},
                result.data
            );
        }, "Convert BGR to RGB with caching")
        .def("rgb_to_bgr_cached", [](FastImageProcessor& self, py::array_t<uint8_t> input) -> py::array_t<uint8_t> {
            py::buffer_info buf_info = input.request();
            cv::Mat image(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
            cv::Mat result = self.rgb_to_bgr_cached(image);
            return py::array_t<uint8_t>(
                {result.rows, result.cols, result.channels()},
                {sizeof(uint8_t)*result.cols*result.channels(), sizeof(uint8_t)*result.channels(), sizeof(uint8_t)},
                result.data
            );
        }, "Convert RGB to BGR with caching")
        .def("convert_detections_fast", &FastImageProcessor::convert_detections_fast,
             "Fast conversion from detection coordinates to cv::Rect")
        .def("draw_boxes_fast", 
             [](FastImageProcessor& self, const cv::Mat& image, 
                const std::vector<BoundingBox>& boxes, 
                py::tuple color = py::make_tuple(0, 255, 0), 
                int thickness = 2) {
                 cv::Scalar cv_color;
                 if (py::len(color) >= 3) {
                     cv_color = cv::Scalar(color[0].cast<double>(), 
                                          color[1].cast<double>(), 
                                          color[2].cast<double>());
                 } else {
                     cv_color = cv::Scalar(0, 255, 0);
                 }
                 return self.draw_boxes_fast(image, boxes, cv_color, thickness);
             },
             py::arg("image"), py::arg("boxes"), 
             py::arg("color") = py::make_tuple(0, 255, 0), 
             py::arg("thickness") = 2,
             "Fast bounding box drawing with labels")
        .def("process_image_batch", &FastImageProcessor::process_image_batch,
             py::arg("images"), py::arg("to_rgb") = true,
             "Process multiple images in batch")
        .def("resize_optimized", &FastImageProcessor::resize_optimized,
             py::arg("image"), py::arg("target_size"), 
             py::arg("interpolation") = cv::INTER_LINEAR,
             "Memory-optimized image resizing")
        .def("get_cache_size", &FastImageProcessor::get_cache_size,
             "Get current cache size")
        .def("clear_all_cache", &FastImageProcessor::clear_all_cache,
             "Clear all caches");
    
    // ProcessingResult
    py::class_<ProcessingResult>(m, "ProcessingResult")
        .def_readonly("frame_id", &ProcessingResult::frame_id)
        .def_readonly("task_type", &ProcessingResult::task_type)
        .def_readonly("result_image", &ProcessingResult::result_image)
        .def_readonly("success", &ProcessingResult::success)
        .def_readonly("error_message", &ProcessingResult::error_message);
    
    // AsyncProcessor
    py::class_<AsyncProcessor>(m, "AsyncProcessor")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("num_workers") = 3, py::arg("max_queue") = 10, py::arg("max_results") = 20,
             "Initialize async processor with worker threads")
        .def("submit_grounding_dino_task", [](AsyncProcessor& self, int frame_id, const cv::Mat& image,
                                             const std::string& text_prompt, double box_threshold, double text_threshold) -> bool {
            return self.submit_grounding_dino_task(frame_id, image, text_prompt, box_threshold, text_threshold);
        }, "Submit GroundingDINO processing task")
        .def("submit_sam2_task", [](AsyncProcessor& self, int frame_id, const cv::Mat& image,
                                   py::list boxes_py, py::list labels_py) -> bool {
            // Convert Python lists to C++ vectors
            std::vector<cv::Rect> boxes;
            std::vector<std::string> labels;
            
            for (auto item : boxes_py) {
                boxes.push_back(tuple_to_rect(item.cast<py::tuple>()));
            }
            
            for (auto item : labels_py) {
                labels.push_back(item.cast<std::string>());
            }
            
            return self.submit_sam2_task(frame_id, image, boxes, labels);
        }, "Submit SAM2 processing task")
        .def("submit_visualization_task", [](AsyncProcessor& self, int frame_id, const cv::Mat& image,
                                            py::list boxes_py, py::list labels_py, const std::string& topic) -> bool {
            std::vector<cv::Rect> boxes;
            std::vector<std::string> labels;
            
            for (auto item : boxes_py) {
                boxes.push_back(tuple_to_rect(item.cast<py::tuple>()));
            }
            
            for (auto item : labels_py) {
                labels.push_back(item.cast<std::string>());
            }
            
            return self.submit_visualization_task(frame_id, image, boxes, labels, topic);
        }, "Submit visualization processing task")
        .def("get_result", &AsyncProcessor::get_result,
             py::return_value_policy::take_ownership,
             "Get processing result by frame ID and task type")
        .def("has_result", &AsyncProcessor::has_result,
             "Check if result is available")
        .def("get_queue_size", &AsyncProcessor::get_queue_size,
             "Get current task queue size")
        .def("get_results_size", &AsyncProcessor::get_results_size,
             "Get current results cache size")
        .def("get_active_workers", &AsyncProcessor::get_active_workers,
             "Get number of active worker threads")
        .def("clear_all_results", &AsyncProcessor::clear_all_results,
             "Clear all cached results")
        .def("shutdown", &AsyncProcessor::shutdown,
             "Shutdown all worker threads");
    
    // ThreadPool
    py::class_<ThreadPool>(m, "ThreadPool")
        .def(py::init<size_t>(), py::arg("num_threads"),
             "Initialize thread pool")
        .def("get_queue_size", &ThreadPool::get_queue_size,
             "Get current task queue size");
    
    // Utility functions
    m.def("boxes_to_xyxy", [](py::list boxes_py) -> py::list {
        std::vector<BoundingBox> boxes;
        for (auto item : boxes_py) {
            boxes.push_back(item.cast<BoundingBox>());
        }
        
        auto xyxy_data = boxes_to_xyxy(boxes);
        
        py::list result;
        for (const auto& xyxy : xyxy_data) {
            py::list coords;
            for (float coord : xyxy) {
                coords.append(coord);
            }
            result.append(coords);
        }
        return result;
    }, "Convert BoundingBox list to xyxy format");
    
    m.def("xyxy_to_boxes", [](py::list xyxy_py, py::list labels_py = py::list()) -> py::list {
        std::vector<std::vector<float>> xyxy_data;
        for (auto item : xyxy_py) {
            py::list coords = item.cast<py::list>();
            std::vector<float> coord_vec;
            for (auto coord : coords) {
                coord_vec.push_back(coord.cast<float>());
            }
            xyxy_data.push_back(coord_vec);
        }
        
        std::vector<std::string> labels;
        for (auto item : labels_py) {
            labels.push_back(item.cast<std::string>());
        }
        
        auto boxes = xyxy_to_boxes(xyxy_data, labels);
        
        py::list result;
        for (const auto& box : boxes) {
            result.append(box);
        }
        return result;
    }, "Convert xyxy format to BoundingBox list",
       py::arg("xyxy_data"), py::arg("labels") = py::list());
    
    // Helper functions for Python integration
    m.def("rect_to_tuple", &rect_to_tuple, "Convert cv::Rect to Python tuple");
    m.def("tuple_to_rect", &tuple_to_rect, "Convert Python tuple to cv::Rect");
}