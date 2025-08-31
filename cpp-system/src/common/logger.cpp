#include "common/logger.hpp"
#include <queue>
#include <condition_variable>
#include <atomic>

namespace ultra {
namespace common {

// Static member definitions
Logger::Config Logger::config_;
LogLevel Logger::current_level_ = LogLevel::INFO;
std::string Logger::logger_name_ = "ultra";
std::mutex Logger::log_mutex_;
std::unique_ptr<std::ofstream> Logger::log_file_;
std::unique_ptr<std::thread> Logger::async_thread_;
std::atomic<bool> Logger::async_shutdown_{false};
std::queue<Logger::LogEntry> Logger::async_queue_;
std::mutex Logger::async_mutex_;
std::condition_variable Logger::async_cv_;

void Logger::initialize(const std::string& name, const Config& config) {
    logger_name_ = name;
    config_ = config;
    current_level_ = config.level;
    
    if (config_.enable_file_output) {
        log_file_ = std::make_unique<std::ofstream>(config_.log_file_path, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << config_.log_file_path << std::endl;
        }
    }
    
    if (config_.enable_async_logging) {
        async_shutdown_.store(false);
        async_thread_ = std::make_unique<std::thread>(async_logging_thread);
    }
}

void Logger::set_level(LogLevel level) {
    current_level_ = level;
}

LogLevel Logger::get_level() {
    return current_level_;
}

void Logger::set_format(OutputFormat format) {
    config_.format = format;
}

void Logger::shutdown() {
    if (async_thread_) {
        async_shutdown_.store(true);
        async_cv_.notify_all();
        if (async_thread_->joinable()) {
            async_thread_->join();
        }
    }
    
    if (log_file_) {
        log_file_->close();
    }
}

void Logger::log_structured(LogLevel level, const std::string& message, const LogContext& context) {
    if (level < current_level_) return;
    log_with_context(level, message, context);
}

void Logger::log_error(const std::string& message, const std::exception& e, const LogContext& context) {
    LogContext error_context = context;
    error_context.add("exception_type", typeid(e).name())
                 .add("exception_message", e.what());
    
    log_with_context(LogLevel::ERROR, message, error_context);
}

void Logger::log_performance(const std::string& operation, uint64_t duration_ns, const LogContext& context) {
    LogContext perf_context = context;
    perf_context.add("operation", operation)
                .add("duration_ns", std::to_string(duration_ns))
                .add("duration_ms", std::to_string(duration_ns / 1000000.0))
                .add("log_type", "performance");
    
    log_with_context(LogLevel::INFO, "Performance: " + operation, perf_context);
}

void Logger::log_security_event(const std::string& event_type, const std::string& description, 
                               const LogContext& context) {
    LogContext security_context = context;
    security_context.add("event_type", event_type)
                   .add("description", description)
                   .add("log_type", "security")
                   .add("severity", "high");
    
    log_with_context(LogLevel::WARNING, "Security Event: " + event_type, security_context);
}

void Logger::log_audit(const std::string& action, const std::string& user, const std::string& resource,
                      const LogContext& context) {
    LogContext audit_context = context;
    audit_context.add("action", action)
                 .add("user", user)
                 .add("resource", resource)
                 .add("log_type", "audit");
    
    log_with_context(LogLevel::INFO, "Audit: " + action, audit_context);
}

void Logger::log_with_context(LogLevel level, const std::string& message, const LogContext& context) {
    LogEntry entry{
        level,
        message,
        context,
        std::chrono::system_clock::now(),
        std::this_thread::get_id()
    };
    
    if (config_.enable_async_logging) {
        {
            std::lock_guard<std::mutex> lock(async_mutex_);
            async_queue_.push(entry);
            
            // Prevent unbounded queue growth
            while (async_queue_.size() > config_.async_buffer_size) {
                async_queue_.pop();
            }
        }
        async_cv_.notify_one();
    } else {
        write_log_entry(entry);
    }
}

void Logger::write_log_entry(const LogEntry& entry) {
    std::string formatted = format_log_entry(entry);
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (config_.enable_console_output) {
        std::cout << formatted << std::endl;
    }
    
    if (config_.enable_file_output && log_file_ && log_file_->is_open()) {
        *log_file_ << formatted << std::endl;
        log_file_->flush();
    }
}

void Logger::async_logging_thread() {
    while (!async_shutdown_.load()) {
        std::unique_lock<std::mutex> lock(async_mutex_);
        async_cv_.wait(lock, [] { return !async_queue_.empty() || async_shutdown_.load(); });
        
        while (!async_queue_.empty()) {
            LogEntry entry = async_queue_.front();
            async_queue_.pop();
            lock.unlock();
            
            write_log_entry(entry);
            
            lock.lock();
        }
    }
    
    // Process remaining entries
    std::lock_guard<std::mutex> lock(async_mutex_);
    while (!async_queue_.empty()) {
        write_log_entry(async_queue_.front());
        async_queue_.pop();
    }
}

std::string Logger::format_log_entry(const LogEntry& entry) {
    switch (config_.format) {
        case OutputFormat::JSON:
            return format_json(entry);
        case OutputFormat::LOGFMT:
            return format_logfmt(entry);
        case OutputFormat::PLAIN:
        default:
            return format_plain(entry);
    }
}

std::string Logger::format_plain(const LogEntry& entry) {
    auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    oss << " [" << level_to_string(entry.level) << "]";
    
    if (config_.include_thread_id) {
        oss << " [" << entry.thread_id << "]";
    }
    
    oss << " " << entry.message;
    
    // Add context fields
    const auto& fields = entry.context.get_fields();
    if (!fields.empty()) {
        oss << " {";
        bool first = true;
        for (const auto& pair : fields) {
            if (!first) oss << ", ";
            oss << pair.first << "=" << pair.second;
            first = false;
        }
        oss << "}";
    }
    
    return oss.str();
}

std::string Logger::format_json(const LogEntry& entry) {
    auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;
    
    std::ostringstream timestamp_oss;
    timestamp_oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    timestamp_oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    
    std::ostringstream oss;
    oss << "{";
    oss << "\"timestamp\":\"" << timestamp_oss.str() << "\",";
    oss << "\"level\":\"" << level_to_string(entry.level) << "\",";
    oss << "\"logger\":\"" << logger_name_ << "\",";
    oss << "\"message\":\"" << entry.message << "\"";
    
    if (config_.include_thread_id) {
        oss << ",\"thread_id\":\"" << entry.thread_id << "\"";
    }
    
    // Add context fields
    const auto& fields = entry.context.get_fields();
    for (const auto& pair : fields) {
        oss << ",\"" << pair.first << "\":\"" << pair.second << "\"";
    }
    
    oss << "}";
    return oss.str();
}

std::string Logger::format_logfmt(const LogEntry& entry) {
    auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;
    
    std::ostringstream timestamp_oss;
    timestamp_oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    timestamp_oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    
    std::ostringstream oss;
    oss << "timestamp=" << timestamp_oss.str();
    oss << " level=" << level_to_string(entry.level);
    oss << " logger=" << logger_name_;
    oss << " message=\"" << entry.message << "\"";
    
    if (config_.include_thread_id) {
        oss << " thread_id=" << entry.thread_id;
    }
    
    // Add context fields
    const auto& fields = entry.context.get_fields();
    for (const auto& pair : fields) {
        oss << " " << pair.first << "=" << pair.second;
    }
    
    return oss.str();
}

} // namespace common
} // namespace ultra