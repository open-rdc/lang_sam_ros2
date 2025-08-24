#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "image_processor.hpp"

namespace py = pybind11;
using namespace fast_processing;

PYBIND11_MODULE(fast_processing, m) {
    m.doc() = "Fast Processing C++ Implementation - Image Processing Only (Sync)";
    
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
    
    // FastImageProcessor - シンプルな同期処理のみ
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
        .def("get_cache_size", &FastImageProcessor::get_cache_size,
             "Get current cache size")
        .def("clear_cache", &FastImageProcessor::clear_all_cache,
             "Clear color conversion cache");
}