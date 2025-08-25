cmake_minimum_required(VERSION 3.8)
project(lang_sam_wrapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(csrt_cpp)
include_directories(/usr/include/python3.10)

# Install Python modules using ament_python
ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(lang_sam)

# Create pybind11 module for native CSRT tracking
add_library(csrt_native MODULE
  csrt_cpp/csrt_tracker.cpp
  csrt_cpp/python_bindings.cpp
)

# Create pybind11 module for fast processing (sync only)
add_library(fast_processing MODULE
  csrt_cpp/image_processor.cpp
  csrt_cpp/fast_bindings.cpp
)

# Set pybind11 properties for csrt_native
target_link_libraries(csrt_native PRIVATE pybind11::module)
set_target_properties(csrt_native PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
pybind11_extension(csrt_native)

# Set pybind11 properties for fast_processing
target_link_libraries(fast_processing PRIVATE pybind11::module)
set_target_properties(fast_processing PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                 SUFFIX "${PYTHON_MODULE_EXTENSION}")
pybind11_extension(fast_processing)

# Link OpenCV libraries
target_link_libraries(csrt_native PRIVATE ${OpenCV_LIBS})
target_link_libraries(fast_processing PRIVATE ${OpenCV_LIBS} Threads::Threads)

# Compiler definitions
target_compile_definitions(csrt_native PRIVATE VERSION_INFO="dev")
target_compile_definitions(fast_processing PRIVATE VERSION_INFO="dev")

# Install the pybind11 modules
install(TARGETS csrt_native fast_processing
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# Install Python scripts as executables
install(PROGRAMS
  lang_sam_wrapper/debug_lang_segment_anything.py
  lang_sam_wrapper/lang_sam_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()