#ifndef LANG_SAM_NAV__LANE_PIXEL_FINDER_HPP_
#define LANG_SAM_NAV__LANE_PIXEL_FINDER_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

namespace lang_sam_nav
{

/**
 * YOLOP Nav互換のLanePixelFinderクラス
 * バイナリマスクから車線ピクセルを検出
 */
class LanePixelFinder
{
public:
  explicit LanePixelFinder(int tolerance = 50);
  
  /**
   * 車線ピクセルを検出
   * @param mask バイナリマスク画像
   * @param left_pixels 左車線ピクセル（出力）
   * @param right_pixels 右車線ピクセル（出力）
   * @param center_pixels 中央ピクセル（出力）
   */
  void findLanePixels(const cv::Mat& mask,
                     std::vector<cv::Point>& left_pixels,
                     std::vector<cv::Point>& right_pixels,
                     std::vector<cv::Point>& center_pixels);
  
  // パラメータ設定
  void setTolerance(int tolerance) { tolerance_ = tolerance; }
  int getTolerance() const { return tolerance_; }

private:
  int tolerance_;  // ピクセル探索範囲
};

} // namespace lang_sam_nav

#endif // LANG_SAM_NAV__LANE_PIXEL_FINDER_HPP_