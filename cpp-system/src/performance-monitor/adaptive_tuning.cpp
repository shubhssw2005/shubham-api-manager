#include "performance-monitor/adaptive_tuning.hpp"
#include "common/logger.hpp"

#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>

namespace ultra_cpp {
namespace performance {

AdaptiveTuningSystem::AdaptiveTuningSystem(const TuningConfig& config) 
    : config_(config) {
    metrics_history_.reserve(config_.history_size);
    tuning_history_.reserve(config_.history_size);
    
    LOG_INFO("Initialized adaptive tuning system with {} objective", 
             static_cast<int>(config_.objective));
}

AdaptiveTuningSystem::~AdaptiveTuningSystem() {
    stop_auto_tuning();
}

void AdaptiveTuningSystem::register_parameter(const TunableParameter& param) {
    std::lock_guard<std::mutex> lock(mutex_);
    parameters_[param.name] = param;
    
    LOG_DEBUG("Registered tunable parameter: {} (range: {} - {})", 
              param.name, param.min_value, param.max_value);
}

void AdaptiveTuningSystem::unregister_parameter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    parameters_.erase(name);
    
    LOG_DEBUG("Unregistered tunable parameter: {}", name);
}

std::vector<std::string> AdaptiveTuningSystem::list_parameters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(parameters_.size());
    
    for (const auto& [name, param] : parameters_) {
        names.push_back(name);
    }
    
    return names;
}

bool AdaptiveTuningSystem::set_parameter(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        LOG_WARN("Parameter '{}' not found", name);
        return false;
    }
    
    auto& param = it->second;
    
    // Clamp value to valid range
    value = std::clamp(value, param.min_value, param.max_value);
    
    double old_value = param.current_value;
    param.current_value = value;
    
    // Apply the change
    if (param.setter) {
        param.setter(value);
    }
    
    LOG_DEBUG("Set parameter '{}' from {} to {}", name, old_value, value);
    return true;
}

double AdaptiveTuningSystem::get_parameter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        return it->second.current_value;
    }
    
    LOG_WARN("Parameter '{}' not found", name);
    return 0.0;
}

void AdaptiveTuningSystem::update_metrics(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add to history
    metrics_history_.push_back(metrics);
    
    // Maintain history size limit
    if (metrics_history_.size() > config_.history_size) {
        metrics_history_.erase(metrics_history_.begin());
    }
    
    // Notify tuning thread if waiting
    cv_.notify_one();
}

AdaptiveTuningSystem::PerformanceMetrics AdaptiveTuningSystem::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!metrics_history_.empty()) {
        return metrics_history_.back();
    }
    
    return PerformanceMetrics{};
}

void AdaptiveTuningSystem::start_auto_tuning() {
    if (auto_tuning_active_.load()) {
        LOG_WARN("Auto-tuning is already active");
        return;
    }
    
    auto_tuning_active_.store(true);
    tuning_thread_ = std::thread(&AdaptiveTuningSystem::tuning_loop, this);
    
    LOG_INFO("Started auto-tuning with {} algorithm", 
             static_cast<int>(current_algorithm_));
}

void AdaptiveTuningSystem::stop_auto_tuning() {
    if (!auto_tuning_active_.load()) {
        return;
    }
    
    auto_tuning_active_.store(false);
    cv_.notify_all();
    
    if (tuning_thread_.joinable()) {
        tuning_thread_.join();
    }
    
    LOG_INFO("Stopped auto-tuning");
}

bool AdaptiveTuningSystem::is_auto_tuning_active() const {
    return auto_tuning_active_.load();
}

void AdaptiveTuningSystem::tuning_loop() {
    LOG_DEBUG("Tuning loop started");
    
    while (auto_tuning_active_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait for tuning interval or stop signal
        if (cv_.wait_for(lock, config_.tuning_interval, 
                        [this] { return !auto_tuning_active_.load(); })) {
            break; // Stop signal received
        }
        
        // Check if we have enough metrics data
        if (metrics_history_.size() < 2) {
            continue;
        }
        
        lock.unlock();
        
        // Run the selected tuning algorithm
        try {
            switch (current_algorithm_) {
                case TuningAlgorithm::GRADIENT_DESCENT:
                    run_gradient_descent();
                    break;
                case TuningAlgorithm::SIMULATED_ANNEALING:
                    run_simulated_annealing();
                    break;
                case TuningAlgorithm::GENETIC_ALGORITHM:
                    run_genetic_algorithm();
                    break;
                case TuningAlgorithm::BAYESIAN_OPTIMIZATION:
                    run_bayesian_optimization();
                    break;
                case TuningAlgorithm::REINFORCEMENT_LEARNING:
                    run_reinforcement_learning();
                    break;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Tuning algorithm failed: {}", e.what());
        }
    }
    
    LOG_DEBUG("Tuning loop stopped");
}

void AdaptiveTuningSystem::run_gradient_descent() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Calculate current fitness
    if (metrics_history_.empty()) return;
    
    double current_fitness = calculate_fitness(metrics_history_.back());
    
    // Try adjusting each parameter
    for (auto& [name, param] : parameters_) {
        // Skip if not in tuning list
        if (!config_.parameters_to_tune.empty() &&
            std::find(config_.parameters_to_tune.begin(), 
                     config_.parameters_to_tune.end(), name) == 
            config_.parameters_to_tune.end()) {
            continue;
        }
        
        double gradient = calculate_gradient(name);
        
        // Apply gradient descent update
        double delta = config_.learning_rate * gradient * param.sensitivity;
        double new_value = param.current_value + delta;
        
        // Clamp to valid range
        new_value = std::clamp(new_value, param.min_value, param.max_value);
        
        if (std::abs(new_value - param.current_value) > param.step_size) {
            double old_value = param.current_value;
            apply_parameter_change(name, new_value);
            
            LOG_DEBUG("Gradient descent: {} {} -> {} (gradient: {})", 
                      name, old_value, new_value, gradient);
        }
    }
}

void AdaptiveTuningSystem::run_simulated_annealing() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (parameters_.empty()) return;
    
    // Select random parameter
    auto it = parameters_.begin();
    std::advance(it, gen() % parameters_.size());
    auto& [name, param] = *it;
    
    // Calculate current fitness
    double current_fitness = calculate_fitness(metrics_history_.back());
    
    // Generate random perturbation
    double perturbation = (dis(gen) - 0.5) * 2.0 * param.step_size;
    double new_value = std::clamp(param.current_value + perturbation,
                                 param.min_value, param.max_value);
    
    // Calculate temperature (decreases over time)
    static size_t iteration = 0;
    double temperature = 1.0 / (1.0 + iteration * 0.01);
    ++iteration;
    
    double old_value = param.current_value;
    apply_parameter_change(name, new_value);
    
    // Wait for metrics update (simplified)
    std::this_thread::sleep_for(config_.measurement_interval);
    
    // Calculate new fitness (would need actual new metrics)
    double new_fitness = current_fitness; // Placeholder
    
    // Accept or reject based on simulated annealing criteria
    double delta = new_fitness - current_fitness;
    if (delta > 0 || dis(gen) < std::exp(delta / temperature)) {
        // Accept the change
        LOG_DEBUG("Simulated annealing: accepted {} {} -> {}", 
                  name, old_value, new_value);
    } else {
        // Reject the change
        apply_parameter_change(name, old_value);
        LOG_DEBUG("Simulated annealing: rejected {} {} -> {}", 
                  name, old_value, new_value);
    }
}

void AdaptiveTuningSystem::run_genetic_algorithm() {
    // Simplified genetic algorithm implementation
    LOG_DEBUG("Running genetic algorithm tuning (simplified)");
    
    // This would implement a full GA with population, crossover, mutation, etc.
    // For now, just do random search with selection
    run_simulated_annealing();
}

void AdaptiveTuningSystem::run_bayesian_optimization() {
    // Simplified Bayesian optimization
    LOG_DEBUG("Running Bayesian optimization tuning (simplified)");
    
    // This would implement Gaussian Process regression and acquisition functions
    // For now, fall back to gradient descent
    run_gradient_descent();
}

void AdaptiveTuningSystem::run_reinforcement_learning() {
    // Simplified RL implementation
    LOG_DEBUG("Running reinforcement learning tuning (simplified)");
    
    // This would implement Q-learning or policy gradient methods
    // For now, use epsilon-greedy exploration
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) < config_.exploration_rate) {
        // Explore: random action
        run_simulated_annealing();
    } else {
        // Exploit: use best known strategy
        run_gradient_descent();
    }
}

double AdaptiveTuningSystem::calculate_fitness(const PerformanceMetrics& metrics) const {
    double fitness = 0.0;
    
    switch (config_.objective) {
        case OptimizationObjective::MAXIMIZE_THROUGHPUT:
            fitness = metrics.throughput.load();
            break;
            
        case OptimizationObjective::MINIMIZE_LATENCY:
            fitness = -metrics.latency_p95.load(); // Negative because we want to minimize
            break;
            
        case OptimizationObjective::MINIMIZE_CPU_USAGE:
            fitness = -metrics.cpu_utilization.load();
            break;
            
        case OptimizationObjective::MINIMIZE_MEMORY_USAGE:
            fitness = -metrics.memory_utilization.load();
            break;
            
        case OptimizationObjective::MAXIMIZE_CACHE_HIT_RATE:
            fitness = metrics.cache_hit_rate.load();
            break;
            
        case OptimizationObjective::MINIMIZE_ERROR_RATE:
            fitness = -static_cast<double>(metrics.error_rate.load());
            break;
            
        case OptimizationObjective::BALANCED_PERFORMANCE:
            // Weighted combination of multiple objectives
            fitness = 0.3 * metrics.throughput.load() +
                     0.3 * (-metrics.latency_p95.load() / 1000000.0) + // Convert ns to ms
                     0.2 * metrics.cache_hit_rate.load() +
                     0.1 * (-metrics.cpu_utilization.load()) +
                     0.1 * (-static_cast<double>(metrics.error_rate.load()));
            break;
    }
    
    return fitness;
}

double AdaptiveTuningSystem::calculate_gradient(const std::string& param_name) {
    if (metrics_history_.size() < 2) {
        return 0.0;
    }
    
    // Simple finite difference approximation
    double current_fitness = calculate_fitness(metrics_history_.back());
    double previous_fitness = calculate_fitness(metrics_history_[metrics_history_.size() - 2]);
    
    return current_fitness - previous_fitness;
}

void AdaptiveTuningSystem::apply_parameter_change(const std::string& name, double new_value) {
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        double old_value = it->second.current_value;
        it->second.current_value = new_value;
        
        if (it->second.setter) {
            it->second.setter(new_value);
        }
        
        // Record the change
        double improvement = 0.0; // Would calculate actual improvement
        record_tuning_result(name, old_value, new_value, improvement);
    }
}

void AdaptiveTuningSystem::record_tuning_result(const std::string& param_name,
                                               double old_value, double new_value,
                                               double improvement) {
    TuningResult result;
    result.parameter_name = param_name;
    result.old_value = old_value;
    result.new_value = new_value;
    result.performance_improvement = improvement;
    result.timestamp = std::chrono::steady_clock::now();
    
    tuning_history_.push_back(result);
    
    // Maintain history size limit
    if (tuning_history_.size() > config_.history_size) {
        tuning_history_.erase(tuning_history_.begin());
    }
}

// Simplified implementations for other classes
struct AdaptiveScheduler::Impl {
    std::vector<std::thread> worker_threads;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running{false};
    SchedulerMetrics metrics;
    
    void worker_loop() {
        while (running.load()) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] { return !task_queue.empty() || !running.load(); });
            
            if (!running.load()) break;
            
            if (!task_queue.empty()) {
                auto task = std::move(task_queue.front());
                task_queue.pop();
                lock.unlock();
                
                auto start = std::chrono::high_resolution_clock::now();
                task();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                metrics.average_execution_time.store(duration.count());
                metrics.tasks_completed.fetch_add(1);
            }
        }
    }
};

AdaptiveScheduler::AdaptiveScheduler() : impl_(std::make_unique<Impl>()) {
    impl_->running.store(true);
    
    // Start worker threads
    size_t num_threads = std::thread::hardware_concurrency();
    for (size_t i = 0; i < num_threads; ++i) {
        impl_->worker_threads.emplace_back(&Impl::worker_loop, impl_.get());
    }
}

AdaptiveScheduler::~AdaptiveScheduler() {
    impl_->running.store(false);
    impl_->queue_cv.notify_all();
    
    for (auto& thread : impl_->worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

AdaptiveScheduler::SchedulerMetrics AdaptiveScheduler::get_metrics() const {
    return impl_->metrics;
}

// Simplified implementations for other tuning components
struct MemoryAllocationTuner::Impl {
    AllocationStats stats;
    std::unordered_map<void*, std::chrono::steady_clock::time_point> allocation_times;
    std::mutex mutex;
};

MemoryAllocationTuner::MemoryAllocationTuner() : impl_(std::make_unique<Impl>()) {}
MemoryAllocationTuner::~MemoryAllocationTuner() = default;

void MemoryAllocationTuner::record_allocation(size_t size, void* ptr) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stats.total_allocations.fetch_add(1);
    impl_->stats.bytes_allocated.fetch_add(size);
    impl_->allocation_times[ptr] = std::chrono::steady_clock::now();
}

void MemoryAllocationTuner::record_deallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stats.total_deallocations.fetch_add(1);
    impl_->allocation_times.erase(ptr);
}

MemoryAllocationTuner::AllocationStats MemoryAllocationTuner::get_stats() const {
    return impl_->stats;
}

struct CacheTuner::Impl {
    std::unordered_map<std::string, CacheMetrics> cache_metrics;
    std::mutex mutex;
};

CacheTuner::CacheTuner() : impl_(std::make_unique<Impl>()) {}
CacheTuner::~CacheTuner() = default;

void CacheTuner::record_cache_access(const std::string& cache_name, 
                                    const std::string& key, bool hit) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metrics = impl_->cache_metrics[cache_name];
    
    if (hit) {
        metrics.hits.fetch_add(1);
    } else {
        metrics.misses.fetch_add(1);
    }
    
    uint64_t total = metrics.hits.load() + metrics.misses.load();
    if (total > 0) {
        metrics.hit_rate.store(static_cast<double>(metrics.hits.load()) / total);
    }
}

CacheTuner::CacheMetrics CacheTuner::get_cache_metrics(const std::string& cache_name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->cache_metrics.find(cache_name);
    return it != impl_->cache_metrics.end() ? it->second : CacheMetrics{};
}

struct NetworkTuner::Impl {
    NetworkConfig config;
    NetworkMetrics metrics;
    std::mutex mutex;
};

NetworkTuner::NetworkTuner() : impl_(std::make_unique<Impl>()) {}
NetworkTuner::~NetworkTuner() = default;

void NetworkTuner::set_network_config(const NetworkConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config = config;
}

NetworkTuner::NetworkConfig NetworkTuner::get_network_config() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config;
}

NetworkTuner::NetworkMetrics NetworkTuner::get_metrics() const {
    return impl_->metrics;
}

// PerformanceTuningManager implementation
PerformanceTuningManager::PerformanceTuningManager() 
    : adaptive_tuning_(std::make_unique<AdaptiveTuningSystem>())
    , adaptive_scheduler_(std::make_unique<AdaptiveScheduler>())
    , memory_tuner_(std::make_unique<MemoryAllocationTuner>())
    , cache_tuner_(std::make_unique<CacheTuner>())
    , network_tuner_(std::make_unique<NetworkTuner>()) {
    
    LOG_INFO("Initialized performance tuning manager");
}

PerformanceTuningManager::~PerformanceTuningManager() = default;

AdaptiveTuningSystem& PerformanceTuningManager::get_adaptive_tuning_system() {
    return *adaptive_tuning_;
}

AdaptiveScheduler& PerformanceTuningManager::get_adaptive_scheduler() {
    return *adaptive_scheduler_;
}

MemoryAllocationTuner& PerformanceTuningManager::get_memory_tuner() {
    return *memory_tuner_;
}

CacheTuner& PerformanceTuningManager::get_cache_tuner() {
    return *cache_tuner_;
}

NetworkTuner& PerformanceTuningManager::get_network_tuner() {
    return *network_tuner_;
}

void PerformanceTuningManager::start_all_tuning() {
    adaptive_tuning_->start_auto_tuning();
    LOG_INFO("Started all performance tuning systems");
}

void PerformanceTuningManager::stop_all_tuning() {
    adaptive_tuning_->stop_auto_tuning();
    LOG_INFO("Stopped all performance tuning systems");
}

} // namespace performance
} // namespace ultra_cpp