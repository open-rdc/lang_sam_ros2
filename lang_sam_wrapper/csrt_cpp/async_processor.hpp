#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>

namespace fast_processing {

// 非同期タスクの基底クラス
struct AsyncTask {
    int frame_id;
    cv::Mat image;
    std::string task_type;
    
    AsyncTask(int id, const cv::Mat& img, const std::string& type)
        : frame_id(id), image(img.clone()), task_type(type) {}
    virtual ~AsyncTask() = default;
};

// GroundingDINO検出用タスク
struct GroundingDINOTask : public AsyncTask {
    std::string text_prompt;
    double box_threshold;
    double text_threshold;
    
    GroundingDINOTask(int id, const cv::Mat& img, 
                     const std::string& prompt,
                     double box_thresh, double text_thresh)
        : AsyncTask(id, img, "GDINO"), text_prompt(prompt), 
          box_threshold(box_thresh), text_threshold(text_thresh) {}
};

// SAM2処理用タスク
struct SAM2Task : public AsyncTask {
    std::vector<cv::Rect> bounding_boxes;
    std::vector<std::string> labels;
    
    SAM2Task(int id, const cv::Mat& img, 
             const std::vector<cv::Rect>& boxes,
             const std::vector<std::string>& lbls)
        : AsyncTask(id, img, "SAM2"), bounding_boxes(boxes), labels(lbls) {}
};

// 可視化処理用タスク
struct VisualizationTask : public AsyncTask {
    std::vector<cv::Rect> bounding_boxes;
    std::vector<std::string> labels;
    std::string output_topic;
    
    VisualizationTask(int id, const cv::Mat& img,
                     const std::vector<cv::Rect>& boxes,
                     const std::vector<std::string>& lbls,
                     const std::string& topic)
        : AsyncTask(id, img, "VIZ"), bounding_boxes(boxes), labels(lbls), output_topic(topic) {}
};

// 処理結果
struct ProcessingResult {
    int frame_id;
    std::string task_type;
    cv::Mat result_image;
    bool success;
    std::string error_message;
    
    ProcessingResult(int id, const std::string& type, const cv::Mat& img, bool ok = true, const std::string& err = "")
        : frame_id(id), task_type(type), result_image(img.clone()), success(ok), error_message(err) {}
};

// 非同期処理マネージャー
class AsyncProcessor {
private:
    std::vector<std::thread> worker_threads_;
    std::queue<std::unique_ptr<AsyncTask>> task_queue_;
    std::unordered_map<int, std::unique_ptr<ProcessingResult>> results_;
    
    mutable std::mutex queue_mutex_;
    mutable std::mutex results_mutex_;
    std::condition_variable queue_condition_;
    
    std::atomic<bool> shutdown_flag_{false};
    std::atomic<int> active_workers_{0};
    
    size_t max_queue_size_;
    size_t max_results_size_;
    
    // ワーカースレッドのメインループ
    void worker_loop();
    
    // 結果キャッシュのクリーンアップ
    void cleanup_old_results();
    
public:
    explicit AsyncProcessor(size_t num_workers = 3, 
                           size_t max_queue = 10, 
                           size_t max_results = 20);
    ~AsyncProcessor();
    
    // タスク投入
    bool submit_grounding_dino_task(int frame_id, const cv::Mat& image,
                                   const std::string& text_prompt,
                                   double box_threshold, double text_threshold);
    
    bool submit_sam2_task(int frame_id, const cv::Mat& image,
                         const std::vector<cv::Rect>& boxes,
                         const std::vector<std::string>& labels);
    
    bool submit_visualization_task(int frame_id, const cv::Mat& image,
                                  const std::vector<cv::Rect>& boxes,
                                  const std::vector<std::string>& labels,
                                  const std::string& output_topic);
    
    // 結果取得
    std::unique_ptr<ProcessingResult> get_result(int frame_id, const std::string& task_type);
    bool has_result(int frame_id, const std::string& task_type) const;
    
    // ステータス情報
    size_t get_queue_size() const;
    size_t get_results_size() const;
    int get_active_workers() const;
    
    // 管理機能
    void clear_all_results();
    void shutdown();
};

// スレッドプールベースの軽量処理器
class ThreadPool {
private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    size_t get_queue_size() const;
};

// テンプレート実装
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks_.emplace([task](){ (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}

} // namespace fast_processing