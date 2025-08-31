#include "common/circuit_breaker.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <unordered_map>

namespace ultra {
namespace common {

CircuitBreaker::CircuitBreaker(const std::string& name, const Config& config)
    : name_(name), config_(config), current_timeout_ms_(config.initial_timeout_ms) {
    
    LOG_INFO("Circuit breaker '{}' initialized with failure_threshold={}, min_requests={}", 
             name_, config_.failure_threshold, config_.minimum_requests);
}

CircuitBreaker::~CircuitBreaker() {
    LOG_DEBUG("Circuit breaker '{}' destroyed", name_);
}

bool CircuitBreaker::allow_request() {
    State current_state = state_.load();
    
    switch (current_state) {
        case State::CLOSED:
            return true;
            
        case State::OPEN:
            if (should_attempt_reset()) {
                transition_to_half_open();
                return true;
            }
            return false;
            
        case State::HALF_OPEN:
            return half_open_calls_.load() < config_.half_open_max_calls;
    }
    
    return false;
}

void CircuitBreaker::record_success(std::chrono::steady_clock::time_point start_time) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    stats_.total_requests.fetch_add(1);
    stats_.successful_requests.fetch_add(1);
    
    State current_state = state_.load();
    
    {
        std::lock_guard<std::mutex> lock(records_mutex_);
        request_records_.push_back({end_time, true, static_cast<uint64_t>(duration.count())});
        cleanup_old_records();
    }
    
    if (current_state == State::HALF_OPEN) {
        uint32_t successes = half_open_successes_.fetch_add(1) + 1;
        uint32_t calls = half_open_calls_.fetch_add(1) + 1;
        
        if (calls >= config_.half_open_max_calls) {
            double success_rate = static_cast<double>(successes) / calls;
            if (success_rate >= config_.half_open_success_threshold) {
                transition_to_closed();
            } else {
                transition_to_open();
            }
        }
    }
}

void CircuitBreaker::record_failure(std::chrono::steady_clock::time_point start_time) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    stats_.total_requests.fetch_add(1);
    stats_.failed_requests.fetch_add(1);
    
    State current_state = state_.load();
    
    {
        std::lock_guard<std::mutex> lock(records_mutex_);
        request_records_.push_back({end_time, false, static_cast<uint64_t>(duration.count())});
        cleanup_old_records();
    }
    
    last_failure_time_.store(end_time);
    consecutive_failures_.fetch_add(1);
    
    if (current_state == State::CLOSED) {
        double failure_rate = calculate_failure_rate();
        uint32_t total_requests = 0;
        
        {
            std::lock_guard<std::mutex> lock(records_mutex_);
            total_requests = request_records_.size();
        }
        
        if (total_requests >= config_.minimum_requests && failure_rate >= config_.failure_threshold) {
            transition_to_open();
        }
    } else if (current_state == State::HALF_OPEN) {
        half_open_calls_.fetch_add(1);
        transition_to_open();
    }
}

void CircuitBreaker::record_rejected() {
    stats_.rejected_requests.fetch_add(1);
}

void CircuitBreaker::transition_to_open() {
    State expected = State::CLOSED;
    if (state_.compare_exchange_strong(expected, State::OPEN) ||
        (expected = State::HALF_OPEN, state_.compare_exchange_strong(expected, State::OPEN))) {
        
        stats_.state_changes.fetch_add(1);
        uint32_t new_timeout = calculate_next_timeout();
        current_timeout_ms_.store(new_timeout);
        stats_.current_timeout_ms.store(new_timeout);
        
        LOG_WARNING("Circuit breaker '{}' opened. Timeout: {}ms, Consecutive failures: {}", 
                   name_, new_timeout, consecutive_failures_.load());
    }
}

void CircuitBreaker::transition_to_half_open() {
    State expected = State::OPEN;
    if (state_.compare_exchange_strong(expected, State::HALF_OPEN)) {
        stats_.state_changes.fetch_add(1);
        half_open_calls_.store(0);
        half_open_successes_.store(0);
        
        LOG_INFO("Circuit breaker '{}' transitioned to half-open state", name_);
    }
}

void CircuitBreaker::transition_to_closed() {
    State expected = State::HALF_OPEN;
    if (state_.compare_exchange_strong(expected, State::CLOSED)) {
        stats_.state_changes.fetch_add(1);
        consecutive_failures_.store(0);
        current_timeout_ms_.store(config_.initial_timeout_ms);
        stats_.current_timeout_ms.store(config_.initial_timeout_ms);
        
        LOG_INFO("Circuit breaker '{}' closed. Service recovered.", name_);
    }
}

bool CircuitBreaker::should_attempt_reset() const {
    if (state_.load() != State::OPEN) {
        return false;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto last_failure = last_failure_time_.load();
    auto timeout = std::chrono::milliseconds(current_timeout_ms_.load());
    
    return (now - last_failure) >= timeout;
}

double CircuitBreaker::calculate_failure_rate() const {
    std::lock_guard<std::mutex> lock(records_mutex_);
    
    if (request_records_.empty()) {
        return 0.0;
    }
    
    uint32_t failures = 0;
    for (const auto& record : request_records_) {
        if (!record.success) {
            failures++;
        }
    }
    
    return static_cast<double>(failures) / request_records_.size();
}

void CircuitBreaker::cleanup_old_records() {
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::milliseconds(config_.time_window_ms);
    
    request_records_.erase(
        std::remove_if(request_records_.begin(), request_records_.end(),
                      [cutoff](const RequestRecord& record) {
                          return record.timestamp < cutoff;
                      }),
        request_records_.end()
    );
}

uint32_t CircuitBreaker::calculate_next_timeout() const {
    uint32_t current = current_timeout_ms_.load();
    uint32_t next = static_cast<uint32_t>(current * config_.backoff_multiplier);
    return std::min(next, config_.max_timeout_ms);
}

void CircuitBreaker::force_open() {
    state_.store(State::OPEN);
    stats_.state_changes.fetch_add(1);
    LOG_WARNING("Circuit breaker '{}' forced open", name_);
}

void CircuitBreaker::force_close() {
    state_.store(State::CLOSED);
    consecutive_failures_.store(0);
    current_timeout_ms_.store(config_.initial_timeout_ms);
    stats_.state_changes.fetch_add(1);
    LOG_INFO("Circuit breaker '{}' forced closed", name_);
}

void CircuitBreaker::reset() {
    std::lock_guard<std::mutex> lock(records_mutex_);
    request_records_.clear();
    
    state_.store(State::CLOSED);
    consecutive_failures_.store(0);
    current_timeout_ms_.store(config_.initial_timeout_ms);
    half_open_calls_.store(0);
    half_open_successes_.store(0);
    
    // Reset stats
    stats_.total_requests.store(0);
    stats_.successful_requests.store(0);
    stats_.failed_requests.store(0);
    stats_.rejected_requests.store(0);
    stats_.state_changes.store(0);
    stats_.current_timeout_ms.store(config_.initial_timeout_ms);
    
    LOG_INFO("Circuit breaker '{}' reset", name_);
}

// Circuit Breaker Manager Implementation
CircuitBreakerManager& CircuitBreakerManager::instance() {
    static CircuitBreakerManager instance;
    return instance;
}

std::shared_ptr<CircuitBreaker> CircuitBreakerManager::get_or_create(
    const std::string& name, const CircuitBreaker::Config& config) {
    
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    auto it = breakers_.find(name);
    if (it != breakers_.end()) {
        return it->second;
    }
    
    auto breaker = std::make_shared<CircuitBreaker>(name, config);
    breakers_[name] = breaker;
    return breaker;
}

std::shared_ptr<CircuitBreaker> CircuitBreakerManager::get(const std::string& name) {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    auto it = breakers_.find(name);
    return (it != breakers_.end()) ? it->second : nullptr;
}

void CircuitBreakerManager::remove(const std::string& name) {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    breakers_.erase(name);
}

std::vector<std::string> CircuitBreakerManager::list_circuit_breakers() const {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    std::vector<std::string> names;
    names.reserve(breakers_.size());
    
    for (const auto& pair : breakers_) {
        names.push_back(pair.first);
    }
    
    return names;
}

CircuitBreakerManager::AggregatedStats CircuitBreakerManager::get_aggregated_stats() const {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    AggregatedStats aggregated{};
    aggregated.total_circuits = breakers_.size();
    
    for (const auto& pair : breakers_) {
        const auto& breaker = pair.second;
        auto stats = breaker->get_stats();
        auto state = breaker->get_state();
        
        switch (state) {
            case CircuitBreaker::State::OPEN:
                aggregated.open_circuits++;
                break;
            case CircuitBreaker::State::HALF_OPEN:
                aggregated.half_open_circuits++;
                break;
            case CircuitBreaker::State::CLOSED:
                aggregated.closed_circuits++;
                break;
        }
        
        aggregated.total_requests += stats.total_requests.load();
        aggregated.total_failures += stats.failed_requests.load();
        aggregated.total_rejections += stats.rejected_requests.load();
    }
    
    return aggregated;
}

} // namespace common
} // namespace ultra