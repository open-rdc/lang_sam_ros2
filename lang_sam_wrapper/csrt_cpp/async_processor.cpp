#include "async_processor.hpp"
#include <iostream>
#include <algorithm>

namespace fast_processing {

// AsyncProcessor implementation
AsyncProcessor::AsyncProcessor(size_t num_workers, size_t max_queue, size_t max_results)
    : max_queue_size_(max_queue), max_results_size_(max_results) {
    
    // ワーカースレッド起動
    worker_threads_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
        worker_threads_.emplace_back(&AsyncProcessor::worker_loop, this);
    }
}

AsyncProcessor::~AsyncProcessor() {
    shutdown();
}

void AsyncProcessor::worker_loop() {
    active_workers_.fetch_add(1);
    
    while (!shutdown_flag_.load()) {
        std::unique_ptr<AsyncTask> task;
        
        // タスク取得
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            queue_condition_.wait(lock, [this] {
                return !task_queue_.empty() || shutdown_flag_.load();
            });
            
            if (shutdown_flag_.load() && task_queue_.empty()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        if (!task) continue;
        
        // タスク実行
        try {
            cv::Mat result_image;
            bool success = true;
            std::string error_msg;
            
            if (task->task_type == "GDINO") {
                auto* gdino_task = static_cast<GroundingDINOTask*>(task.get());
                
                // GroundingDINO処理のモック（実際のPython呼び出しが必要）
                // 現在は画像をそのまま返す
                result_image = task->image.clone();
                
                // TODO: 実際のGroundingDINO推論をPythonで実行
                // Python GIL問題のため、ここではPythonスクリプト呼び出しが必要
                
            } else if (task->task_type == "SAM2") {
                auto* sam2_task = static_cast<SAM2Task*>(task.get());
                
                // SAM2処理のモック（実際のPython呼び出しが必要）
                // ここではバウンディングボックス描画のみ実装
                result_image = task->image.clone();
                cv::Scalar color(0, 255, 255); // 黄色
                
                for (const auto& box : sam2_task->bounding_boxes) {
                    cv::rectangle(result_image, box, color, 2);
                }
                
            } else if (task->task_type == "VIZ") {
                auto* viz_task = static_cast<VisualizationTask*>(task.get());
                
                // 可視化処理
                result_image = task->image.clone();
                cv::Scalar color(0, 255, 0); // 緑色
                
                for (size_t i = 0; i < viz_task->bounding_boxes.size(); ++i) {
                    const auto& box = viz_task->bounding_boxes[i];
                    cv::rectangle(result_image, box, color, 2);
                    
                    // ラベル描画
                    if (i < viz_task->labels.size() && !viz_task->labels[i].empty()) {
                        cv::Point text_pos(box.x, box.y - 5);
                        if (text_pos.y < 20) text_pos.y = box.y + box.height + 20;
                        
                        cv::putText(result_image, viz_task->labels[i], text_pos,
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    }
                }
            }
            
            // 結果保存
            {
                std::lock_guard<std::mutex> results_lock(results_mutex_);
                
                int key = task->frame_id;
                results_[key] = std::make_unique<ProcessingResult>(
                    task->frame_id, task->task_type, result_image, success, error_msg
                );
                
                // 古い結果をクリーンアップ
                cleanup_old_results();
            }
            
        } catch (const std::exception& e) {
            // エラー処理
            std::lock_guard<std::mutex> results_lock(results_mutex_);
            results_[task->frame_id] = std::make_unique<ProcessingResult>(
                task->frame_id, task->task_type, cv::Mat(), false, e.what()
            );
        }
    }
    
    active_workers_.fetch_sub(1);
}

void AsyncProcessor::cleanup_old_results() {
    if (results_.size() <= max_results_size_) return;
    
    // 古い結果を削除（フレームID基準）
    std::vector<int> frame_ids;
    frame_ids.reserve(results_.size());
    
    for (const auto& pair : results_) {
        frame_ids.push_back(pair.first);
    }
    
    std::sort(frame_ids.begin(), frame_ids.end());
    
    // 古いものから削除
    size_t to_remove = results_.size() - max_results_size_;
    for (size_t i = 0; i < to_remove && i < frame_ids.size(); ++i) {
        results_.erase(frame_ids[i]);
    }
}

bool AsyncProcessor::submit_grounding_dino_task(int frame_id, const cv::Mat& image,
                                               const std::string& text_prompt,
                                               double box_threshold, double text_threshold) {
    if (shutdown_flag_.load()) return false;
    
    auto task = std::make_unique<GroundingDINOTask>(frame_id, image, text_prompt, box_threshold, text_threshold);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (task_queue_.size() >= max_queue_size_) {
            if (!task_queue_.empty()) {
                task_queue_.pop();
            }
        }
        
        task_queue_.push(std::move(task));
    }
    
    queue_condition_.notify_one();
    return true;
}

bool AsyncProcessor::submit_sam2_task(int frame_id, const cv::Mat& image,
                                     const std::vector<cv::Rect>& boxes,
                                     const std::vector<std::string>& labels) {
    if (shutdown_flag_.load()) return false;
    
    auto task = std::make_unique<SAM2Task>(frame_id, image, boxes, labels);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // キューサイズ制限
        if (task_queue_.size() >= max_queue_size_) {
            // 古いタスクを削除
            if (!task_queue_.empty()) {
                task_queue_.pop();
            }
        }
        
        task_queue_.push(std::move(task));
    }
    
    queue_condition_.notify_one();
    return true;
}

bool AsyncProcessor::submit_visualization_task(int frame_id, const cv::Mat& image,
                                              const std::vector<cv::Rect>& boxes,
                                              const std::vector<std::string>& labels,
                                              const std::string& output_topic) {
    if (shutdown_flag_.load()) return false;
    
    auto task = std::make_unique<VisualizationTask>(frame_id, image, boxes, labels, output_topic);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (task_queue_.size() >= max_queue_size_) {
            if (!task_queue_.empty()) {
                task_queue_.pop();
            }
        }
        
        task_queue_.push(std::move(task));
    }
    
    queue_condition_.notify_one();
    return true;
}

std::unique_ptr<ProcessingResult> AsyncProcessor::get_result(int frame_id, const std::string& task_type) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    auto it = results_.find(frame_id);
    if (it != results_.end() && it->second->task_type == task_type) {
        auto result = std::move(it->second);
        results_.erase(it);
        return result;
    }
    
    return nullptr;
}

bool AsyncProcessor::has_result(int frame_id, const std::string& task_type) const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    auto it = results_.find(frame_id);
    return (it != results_.end() && it->second->task_type == task_type);
}

size_t AsyncProcessor::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

size_t AsyncProcessor::get_results_size() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_.size();
}

int AsyncProcessor::get_active_workers() const {
    return active_workers_.load();
}

void AsyncProcessor::clear_all_results() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    results_.clear();
}

void AsyncProcessor::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_flag_.store(true);
    }
    
    queue_condition_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
}

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads) {
    threads_.reserve(num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) return;
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : threads_) {
        worker.join();
    }
}

size_t ThreadPool::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

} // namespace fast_processing