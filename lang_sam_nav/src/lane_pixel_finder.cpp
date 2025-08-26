#include "lang_sam_nav/lane_pixel_finder.hpp"

namespace lang_sam_nav
{

LanePixelFinder::LanePixelFinder(int tolerance)
: tolerance_(tolerance)
{
}

void LanePixelFinder::findLanePixels(const cv::Mat& mask,
                                    std::vector<cv::Point>& left_pixels,
                                    std::vector<cv::Point>& right_pixels,
                                    std::vector<cv::Point>& center_pixels)
{
  // Clear output vectors
  left_pixels.clear();
  right_pixels.clear();
  center_pixels.clear();
  
  if (mask.empty()) {
    return;
  }
  
  int rows = mask.rows;
  int cols = mask.cols;
  
  // デバッグ：最初の非ゼロピクセルを検索
  int first_nonzero_row = -1;
  for (int row = rows - 1; row >= 0; --row) {
    const uchar* row_ptr = mask.ptr<uchar>(row);
    for (int col = 0; col < cols; ++col) {
      if (row_ptr[col] > 0) {
        first_nonzero_row = row;
        break;
      }
    }
    if (first_nonzero_row >= 0) break;
  }
  
  // Start from bottom of image
  for (int row = rows - 1; row >= 0; --row) {
    const uchar* row_ptr = mask.ptr<uchar>(row);
    
    // Find initial center
    int center = cols / 2;
    
    // Update search boundaries based on previous detections
    int left_bound = std::max(0, center - tolerance_);
    int right_bound = std::min(cols - 1, center + tolerance_);
    
    // If we have previous pixels, use them to guide search
    if (!center_pixels.empty()) {
      center = center_pixels.back().x;
      left_bound = std::max(0, center - tolerance_);
      right_bound = std::min(cols - 1, center + tolerance_);
    }
    
    // Search for left pixel
    int left = -1;
    bool found_left = false;
    for (int col = center; col >= left_bound; --col) {
      if (row_ptr[col] > 0) {
        left = col;
        found_left = true;
        break;
      }
    }
    
    // Search for right pixel
    int right = -1;
    bool found_right = false;
    for (int col = center; col <= right_bound; ++col) {
      if (row_ptr[col] > 0) {
        right = col;
        found_right = true;
        break;
      }
    }
    
    // Add found pixels
    if (found_left) {
      left_pixels.push_back(cv::Point(left, row));
    }
    
    if (found_right) {
      right_pixels.push_back(cv::Point(right, row));
    }
    
    // Calculate center if both left and right found
    if (found_left && found_right) {
      int center_x = (left + right) / 2;
      center_pixels.push_back(cv::Point(center_x, row));
    }
  }
}

} // namespace lang_sam_nav