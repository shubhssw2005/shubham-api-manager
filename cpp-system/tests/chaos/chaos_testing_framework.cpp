#include "chaos_testing_framework.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <signal.h>
#include <json/json.h>

namespace ultra::testing::chaos {

// NetworkChaosInjector implementation
void NetworkChaosInjector::inject_network_latency(const std::string& interface, 
                                                 std::chrono::milliseconds latency) {
    std::string command = "tc qdisc add dev " + interface + 
                         " root netem delay " + std::to_string(latency.count()) + "ms";
    execute_tc_command(command);
    
    active_failures_[interface].latency = latency;
}

void NetworkChaosInjector::inject_packet_loss(const std::string& interface, double loss_rate) {
    std::string command = "tc qdisc add dev " + interface + 
                         " root netem loss " + std::to_string(loss_rate * 100) + "%";
    execute_tc_command(command);
    
    active_failures_[interface].packet_loss_rate = loss_rate;
}

void NetworkChaosInjector::inject_bandwidth_limit(const std::string& interface, double limit_mbps) {
    std::string rate = std::to_string(static_cast<int>(limit_mbps)) + "mbit";
    std::string command = "tc qdisc add dev " + interface + " root tbf rate " + rate + 
                         " burst 32kbit latency 400ms";
    execute_tc_command(command);
    
    active_failures_[interface].bandwidth_limit_mbps = limit_mbps;
}

void NetworkChaosInjector::inject_network_partition(const std::vector<std::string>& blocked_ips) {
    for (const auto& ip : blocked_ips) {
        std::string command = "iptables -A INPUT -s " + ip + " -j DROP";
        execute_iptables_command(command);
        command = "iptables -A OUTPUT -d " + ip + " -j DROP";
        execute_iptables_command(command);
    }
}

void NetworkChaosInjector::restore_network(const std::string& interface) {
    std::string command = "tc qdisc del dev " + interface + " root";
    execute_tc_command(command);
    
    // Clear iptables rules (simplified - in production would be more targeted)
    execute_iptables_command("iptables -F");
    
    active_failures_.erase(interface);
}

void NetworkChaosInjector::execute_tc_command(const std::string& command) {
    // In a real implementation, this would execute the tc command
    // For testing purposes, we'll just log it
    std::cout << "Executing TC command: " << command << std::endl;
}

void NetworkChaosInjector::execute_iptables_command(const std::string& command) {
    // In a real implementation, this would execute the iptables command
    // For testing purposes, we'll just log it
    std::cout << "Executing iptables command: " << command << std::endl;
}

// ResourceChaosInjector implementation
void ResourceChaosInjector::inject_memory_pressure(size_t pressure_mb, 
                                                  std::chrono::milliseconds duration) {
    stop_stress_ = false;
    stress_threads_.emplace_back(&ResourceChaosInjector::memory_stress_worker, 
                                this, pressure_mb);
    
    // Auto-stop after duration
    std::thread([this, duration]() {
        std::this_thread::sleep_for(duration);
        stop_stress_ = true;
    }).detach();
}

void ResourceChaosInjector::inject_cpu_stress(double cpu_percent, 
                                            std::chrono::milliseconds duration) {
    stop_stress_ = false;
    stress_threads_.emplace_back(&ResourceChaosInjector::cpu_stress_worker, 
                                this, cpu_percent);
    
    // Auto-stop after duration
    std::thread([this, duration]() {
        std::this_thread::sleep_for(duration);
        stop_stress_ = true;
    }).detach();
}

void ResourceChaosInjector::inject_disk_io_failure(const std::string& path, 
                                                  std::chrono::milliseconds duration) {
    stop_stress_ = false;
    stress_threads_.emplace_back(&ResourceChaosInjector::disk_stress_worker, this, path);
    
    // Auto-stop after duration
    std::thread([this, duration]() {
        std::this_thread::sleep_for(duration);
        stop_stress_ = true;
    }).detach();
}

void ResourceChaosInjector::inject_disk_full(const std::string& path, size_t fill_size_mb) {
    std::string temp_file = path + "/chaos_fill_" + std::to_string(rand());
    temp_files_.push_back(temp_file);
    
    // Create large file to fill disk
    std::ofstream file(temp_file, std::ios::binary);
    std::vector<char> buffer(1024 * 1024, 0); // 1MB buffer
    
    for (size_t i = 0; i < fill_size_mb; ++i) {
        file.write(buffer.data(), buffer.size());
    }
    file.close();
}

void ResourceChaosInjector::inject_fd_exhaustion(size_t max_fds) {
    // Set resource limit for file descriptors
    struct rlimit limit;
    limit.rlim_cur = max_fds;
    limit.rlim_max = max_fds;
    setrlimit(RLIMIT_NOFILE, &limit);
}

void ResourceChaosInjector::cleanup_all_injections() {
    stop_stress_ = true;
    
    // Wait for all stress threads to complete
    for (auto& thread : stress_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    stress_threads_.clear();
    
    // Remove temporary files
    for (const auto& file : temp_files_) {
        std::remove(file.c_str());
    }
    temp_files_.clear();
}

void ResourceChaosInjector::memory_stress_worker(size_t size_mb) {
    std::vector<std::unique_ptr<char[]>> allocations;
    
    while (!stop_stress_) {
        try {
            // Allocate 1MB chunks
            auto chunk = std::make_unique<char[]>(1024 * 1024);
            // Touch the memory to ensure it's actually allocated
            memset(chunk.get(), 0x42, 1024 * 1024);
            allocations.push_back(std::move(chunk));
            
            if (allocations.size() >= size_mb) {
                // Hold the memory for a while
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::bad_alloc&) {
            // Memory allocation failed, which is expected under pressure
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void ResourceChaosInjector::cpu_stress_worker(double target_percent) {
    auto work_duration = std::chrono::microseconds(
        static_cast<long>(target_percent * 1000)); // Work time per millisecond
    auto sleep_duration = std::chrono::microseconds(1000) - work_duration;
    
    while (!stop_stress_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Busy work
        volatile long dummy = 0;
        while (std::chrono::high_resolution_clock::now() - start < work_duration) {
            for (int i = 0; i < 1000; ++i) {
                dummy += i;
            }
        }
        
        // Sleep to achieve target CPU usage
        if (sleep_duration.count() > 0) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }
}

void ResourceChaosInjector::disk_stress_worker(const std::string& path) {
    std::string temp_file = path + "/chaos_io_" + std::to_string(rand());
    
    while (!stop_stress_) {
        try {
            // Write and read operations to stress disk I/O
            std::ofstream out_file(temp_file, std::ios::binary);
            std::vector<char> data(1024, 0x42);
            
            for (int i = 0; i < 100; ++i) {
                out_file.write(data.data(), data.size());
            }
            out_file.close();
            
            // Read the file back
            std::ifstream in_file(temp_file, std::ios::binary);
            char buffer[1024];
            while (in_file.read(buffer, sizeof(buffer))) {
                // Just read, don't process
            }
            in_file.close();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } catch (...) {
            // I/O error occurred, which is expected
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    // Cleanup
    std::remove(temp_file.c_str());
}

// ApplicationChaosInjector implementation
void ApplicationChaosInjector::inject_exception(const std::string& component, 
                                              const std::string& exception_type,
                                              double probability) {
    exception_probabilities_[component] = probability;
}

void ApplicationChaosInjector::inject_timeout(const std::string& component,
                                            std::chrono::milliseconds timeout_duration,
                                            double probability) {
    if (should_inject_failure(component)) {
        timeout_injections_[component] = timeout_duration;
    }
}

void ApplicationChaosInjector::inject_memory_leak(size_t leak_size_mb, 
                                                std::chrono::milliseconds interval) {
    std::thread([this, leak_size_mb, interval]() {
        while (memory_chaos_enabled_) {
            // Allocate memory and "forget" to free it
            void* leaked = malloc(leak_size_mb * 1024 * 1024);
            if (leaked) {
                leaked_memory_.push_back(leaked);
                // Touch the memory to ensure it's actually allocated
                memset(leaked, 0x42, leak_size_mb * 1024 * 1024);
            }
            
            std::this_thread::sleep_for(interval);
        }
    }).detach();
}

void ApplicationChaosInjector::inject_thread_starvation(const std::string& component,
                                                      std::chrono::milliseconds duration) {
    // Create many threads to exhaust thread pool
    std::vector<std::thread> starvation_threads;
    
    for (int i = 0; i < 1000; ++i) {
        starvation_threads.emplace_back([duration]() {
            std::this_thread::sleep_for(duration);
        });
    }
    
    // Detach threads to let them run
    for (auto& thread : starvation_threads) {
        thread.detach();
    }
}

void ApplicationChaosInjector::enable_memory_chaos() {
    memory_chaos_enabled_ = true;
}

void ApplicationChaosInjector::disable_memory_chaos() {
    memory_chaos_enabled_ = false;
    
    // Clean up leaked memory
    for (void* ptr : leaked_memory_) {
        free(ptr);
    }
    leaked_memory_.clear();
}

void ApplicationChaosInjector::enable_thread_chaos() {
    thread_chaos_enabled_ = true;
}

void ApplicationChaosInjector::disable_thread_chaos() {
    thread_chaos_enabled_ = false;
}

bool ApplicationChaosInjector::should_inject_failure(const std::string& component) {
    auto it = exception_probabilities_.find(component);
    if (it == exception_probabilities_.end()) {
        return false;
    }
    
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(rng_) < it->second;
}

// ChaosSystemMonitor implementation
void ChaosSystemMonitor::start_monitoring() {
    monitoring_active_ = true;
    monitoring_thread_ = std::thread(&ChaosSystemMonitor::monitoring_loop, this);
}

void ChaosSystemMonitor::stop_monitoring() {
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

ChaosSystemMonitor::SystemState ChaosSystemMonitor::get_current_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_history_.empty() ? SystemState{} : state_history_.back();
}

std::vector<ChaosSystemMonitor::SystemState> ChaosSystemMonitor::get_history() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_history_;
}

bool ChaosSystemMonitor::detect_anomaly(const SystemState& current, 
                                       const SystemState& baseline) const {
    // Simple anomaly detection based on thresholds
    const double CPU_THRESHOLD = 2.0;      // 2x baseline
    const double MEMORY_THRESHOLD = 1.5;   // 1.5x baseline
    const double RESPONSE_TIME_THRESHOLD = 3.0; // 3x baseline
    
    return (current.cpu_usage_percent > baseline.cpu_usage_percent * CPU_THRESHOLD) ||
           (current.memory_usage_mb > baseline.memory_usage_mb * MEMORY_THRESHOLD) ||
           (current.response_time_ms > baseline.response_time_ms * RESPONSE_TIME_THRESHOLD);
}

void ChaosSystemMonitor::set_alert_callback(AlertCallback callback) {
    alert_callback_ = callback;
}

void ChaosSystemMonitor::monitoring_loop() {
    while (monitoring_active_) {
        SystemState current_state = collect_system_metrics();
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_history_.push_back(current_state);
            
            // Keep only last 1000 samples
            if (state_history_.size() > 1000) {
                state_history_.erase(state_history_.begin());
            }
        }
        
        // Check for anomalies
        if (state_history_.size() > 10) {
            SystemState baseline = state_history_[state_history_.size() - 10];
            if (detect_anomaly(current_state, baseline) && alert_callback_) {
                alert_callback_("Anomaly detected", current_state);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

ChaosSystemMonitor::SystemState ChaosSystemMonitor::collect_system_metrics() {
    SystemState state;
    
    // Collect CPU usage
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open()) {
        std::string line;
        std::getline(stat_file, line);
        // Parse CPU usage from /proc/stat (simplified)
        state.cpu_usage_percent = 50.0; // Placeholder
    }
    
    // Collect memory usage
    std::ifstream meminfo_file("/proc/meminfo");
    if (meminfo_file.is_open()) {
        std::string line;
        while (std::getline(meminfo_file, line)) {
            if (line.find("MemAvailable:") == 0) {
                // Parse memory usage (simplified)
                state.memory_usage_mb = 1024; // Placeholder
                break;
            }
        }
    }
    
    // Collect disk usage
    struct statvfs disk_stat;
    if (statvfs("/", &disk_stat) == 0) {
        state.disk_usage_mb = (disk_stat.f_blocks - disk_stat.f_bavail) * 
                             disk_stat.f_frsize / (1024 * 1024);
    }
    
    // Collect file descriptor count
    state.open_file_descriptors = 100; // Placeholder
    
    // Collect thread count
    state.thread_count = 50; // Placeholder
    
    return state;
}

// ChaosTestingEngine implementation
ChaosTestingEngine::ChaosTestingEngine() {
    network_injector_ = std::make_unique<NetworkChaosInjector>();
    resource_injector_ = std::make_unique<ResourceChaosInjector>();
    app_injector_ = std::make_unique<ApplicationChaosInjector>();
    system_monitor_ = std::make_shared<ChaosSystemMonitor>();
}

ChaosTestingEngine::~ChaosTestingEngine() {
    stop_chaos_testing();
}

void ChaosTestingEngine::register_experiment(const ChaosExperiment& experiment) {
    experiments_[experiment.name] = experiment;
}

void ChaosTestingEngine::remove_experiment(const std::string& name) {
    experiments_.erase(name);
}

void ChaosTestingEngine::start_chaos_testing() {
    chaos_active_ = true;
    chaos_paused_ = false;
    system_monitor_->start_monitoring();
    chaos_thread_ = std::thread(&ChaosTestingEngine::chaos_execution_loop, this);
}

void ChaosTestingEngine::stop_chaos_testing() {
    chaos_active_ = false;
    if (chaos_thread_.joinable()) {
        chaos_thread_.join();
    }
    system_monitor_->stop_monitoring();
    
    // Cleanup all injections
    resource_injector_->cleanup_all_injections();
    app_injector_->disable_memory_chaos();
    app_injector_->disable_thread_chaos();
}

ChaosExperimentResult ChaosTestingEngine::run_experiment(const std::string& experiment_name,
                                                       std::chrono::seconds duration) {
    auto it = experiments_.find(experiment_name);
    if (it == experiments_.end()) {
        ChaosExperimentResult result;
        result.experiment_name = experiment_name;
        result.experiment_passed = false;
        result.failure_reason = "Experiment not found";
        return result;
    }
    
    ChaosExperimentResult result;
    result.experiment_name = experiment_name;
    result.type = it->second.type;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + duration;
    
    while (std::chrono::high_resolution_clock::now() < end_time) {
        if (check_safety_limits()) {
            execute_experiment(it->second);
            result.total_injections++;
            
            // Monitor system response
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            if (it->second.validation_function && it->second.validation_function()) {
                result.successful_injections++;
            } else {
                result.failed_injections++;
            }
        }
        
        std::this_thread::sleep_for(it->second.interval);
    }
    
    result.experiment_passed = result.failed_injections == 0;
    
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        experiment_history_.push_back(result);
    }
    
    return result;
}

void ChaosTestingEngine::chaos_execution_loop() {
    while (chaos_active_) {
        if (!chaos_paused_ && check_safety_limits()) {
            // Randomly select and execute experiments
            if (!experiments_.empty()) {
                auto it = experiments_.begin();
                std::advance(it, rand() % experiments_.size());
                execute_experiment(it->second);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void ChaosTestingEngine::execute_experiment(const ChaosExperiment& experiment) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(std::mt19937{std::random_device{}()}) > experiment.probability * global_probability_) {
        return; // Skip this execution
    }
    
    try {
        switch (experiment.type) {
            case ChaosType::NETWORK_LATENCY:
                network_injector_->inject_network_latency("eth0", experiment.duration);
                break;
                
            case ChaosType::NETWORK_PACKET_LOSS:
                network_injector_->inject_packet_loss("eth0", 0.1); // 10% loss
                break;
                
            case ChaosType::MEMORY_PRESSURE:
                resource_injector_->inject_memory_pressure(512, experiment.duration);
                break;
                
            case ChaosType::CPU_STRESS:
                resource_injector_->inject_cpu_stress(0.8, experiment.duration);
                break;
                
            case ChaosType::EXCEPTION_INJECTION:
                app_injector_->inject_exception("test_component", "std::runtime_error", 0.1);
                break;
                
            case ChaosType::MEMORY_LEAK_INJECTION:
                app_injector_->inject_memory_leak(10, std::chrono::seconds(1));
                break;
                
            default:
                if (experiment.custom_failure_action) {
                    experiment.custom_failure_action();
                }
                break;
        }
        
        // Wait for the chaos to take effect
        std::this_thread::sleep_for(experiment.duration);
        
        // Execute recovery if available
        if (experiment.recovery_action) {
            experiment.recovery_action();
        }
        
    } catch (const std::exception& e) {
        // Log the exception but continue
        std::cerr << "Chaos experiment failed: " << e.what() << std::endl;
    }
}

bool ChaosTestingEngine::check_safety_limits() const {
    if (!system_monitor_) return true;
    
    auto current_state = system_monitor_->get_current_state();
    
    for (const auto& [metric, limit] : safety_limits_) {
        if (metric == "cpu_usage" && current_state.cpu_usage_percent > limit) {
            return false;
        }
        if (metric == "memory_usage" && current_state.memory_usage_mb > limit) {
            return false;
        }
        if (metric == "error_rate" && current_state.error_rate > limit) {
            return false;
        }
    }
    
    return true;
}

// ChaosExperimentLibrary implementation
ChaosExperiment ChaosExperimentLibrary::network_latency_experiment() {
    ChaosExperiment experiment;
    experiment.name = "network_latency";
    experiment.type = ChaosType::NETWORK_LATENCY;
    experiment.probability = 0.2;
    experiment.duration = std::chrono::seconds(10);
    experiment.interval = std::chrono::seconds(30);
    experiment.parameters["latency_ms"] = "100";
    experiment.parameters["interface"] = "eth0";
    return experiment;
}

ChaosExperiment ChaosExperimentLibrary::memory_pressure_experiment() {
    ChaosExperiment experiment;
    experiment.name = "memory_pressure";
    experiment.type = ChaosType::MEMORY_PRESSURE;
    experiment.probability = 0.15;
    experiment.duration = std::chrono::seconds(15);
    experiment.interval = std::chrono::minutes(2);
    experiment.parameters["pressure_mb"] = "512";
    return experiment;
}

ChaosExperiment ChaosExperimentLibrary::exception_injection_experiment() {
    ChaosExperiment experiment;
    experiment.name = "exception_injection";
    experiment.type = ChaosType::EXCEPTION_INJECTION;
    experiment.probability = 0.1;
    experiment.duration = std::chrono::seconds(5);
    experiment.interval = std::chrono::seconds(20);
    experiment.target_components = {"cache", "api_gateway", "stream_processor"};
    return experiment;
}

} // namespace ultra::testing::chaos