#pragma once

#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

namespace ultra {
namespace common {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

/**
 * Structured logging context for adding key-value pairs to log entries
 */
class LogContext {
public:
    LogContext& add(const std::string& key, const std::string& value) {
        fields_[key] = value;
        return *this;
    }
    
    template<typename T>
    LogContext& add(const std::string& key, const T& value) {
        std::ostringstream oss;
        oss << value;
        fields_[key] = oss.str();
        return *this;
    }
    
    std::string to_json() const {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& pair : fields_) {
            if (!first) oss << ",";
            oss << "\"" << pair.first << "\":\"" << pair.second << "\"";
            first = false;
        }
        oss << "}";
        return oss.str();
    }
    
    const std::unordered_map<std::string, std::string>& get_fields() const {
        return fields_;
    }

private:
    std::unordered_map<std::string, std::string> fields_;
};

/**
 * Enhanced Logger with structured logging support
 */
class Logger {
public:
    enum class OutputFormat {
        PLAIN,      // Human-readable format
        JSON,       // Structured JSON format
        LOGFMT      // Key=value format
    };
    
    struct Config {
        LogLevel level = LogLevel::INFO;
        OutputFormat format = OutputFormat::PLAIN;
        bool enable_file_output = false;
        std::string log_file_path = "ultra.log";
        bool enable_console_output = true;
        bool enable_async_logging = false;
        size_t async_buffer_size = 1000;
        bool include_thread_id = true;
        bool include_source_location = false;
    };
    
    static void initialize(const std::string& name, const Config& config);
    static void set_level(LogLevel level);
    static LogLevel get_level();
    static void set_format(OutputFormat format);
    static void shutdown();
    
    // Basic logging methods
    template<typename... Args>
    static void log(LogLevel level, const std::string& format, Args&&... args) {
        if (level < current_level_) return;
        
        std::string message = format_string(format, std::forward<Args>(args)...);
        log_with_context(level, message, LogContext{});
    }
    
    // Structured logging methods
    static void log_structured(LogLevel level, const std::string& message, const LogContext& context);
    
    // Error logging with additional context
    static void log_error(const std::string& message, const std::exception& e, const LogContext& context = {});
    
    // Performance logging
    static void log_performance(const std::string& operation, uint64_t duration_ns, const LogContext& context = {});
    
    // Security logging
    static void log_security_event(const std::string& event_type, const std::string& description, 
                                  const LogContext& context = {});
    
    // Audit logging
    static void log_audit(const std::string& action, const std::string& user, const std::string& resource,
                         const LogContext& context = {});

private:
    static Config config_;
    static LogLevel current_level_;
    static std::string logger_name_;
    static std::mutex log_mutex_;
    static std::unique_ptr<std::ofstream> log_file_;
    static std::unique_ptr<std::thread> async_thread_;
    static std::atomic<bool> async_shutdown_;
    
    struct LogEntry {
        LogLevel level;
        std::string message;
        LogContext context;
        std::chrono::system_clock::time_point timestamp;
        std::thread::id thread_id;
    };
    
    static std::queue<LogEntry> async_queue_;
    static std::mutex async_mutex_;
    static std::condition_variable async_cv_;
    
    static void log_with_context(LogLevel level, const std::string& message, const LogContext& context);
    static void write_log_entry(const LogEntry& entry);
    static void async_logging_thread();
    
    static std::string format_log_entry(const LogEntry& entry);
    static std::string format_plain(const LogEntry& entry);
    static std::string format_json(const LogEntry& entry);
    static std::string format_logfmt(const LogEntry& entry);
    
    static std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRIT";
            default: return "UNKNOWN";
        }
    }
    
    template<typename T>
    static std::string format_string(const std::string& format, T&& value) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            std::ostringstream oss;
            oss << value;
            std::string result = format;
            result.replace(pos, 2, oss.str());
            return result;
        }
        return format;
    }
    
    template<typename T, typename... Args>
    static std::string format_string(const std::string& format, T&& value, Args&&... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            std::ostringstream oss;
            oss << value;
            std::string partial = format;
            partial.replace(pos, 2, oss.str());
            return format_string(partial, std::forward<Args>(args)...);
        }
        return format;
    }
    
    static std::string format_string(const std::string& format) {
        return format;
    }
};

// Convenience macros for structured logging
#define LOG_TRACE(...) ultra::common::Logger::log(ultra::common::LogLevel::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) ultra::common::Logger::log(ultra::common::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...) ultra::common::Logger::log(ultra::common::LogLevel::INFO, __VA_ARGS__)
#define LOG_WARNING(...) ultra::common::Logger::log(ultra::common::LogLevel::WARNING, __VA_ARGS__)
#define LOG_ERROR(...) ultra::common::Logger::log(ultra::common::LogLevel::ERROR, __VA_ARGS__)
#define LOG_CRITICAL(...) ultra::common::Logger::log(ultra::common::LogLevel::CRITICAL, __VA_ARGS__)

// Structured logging macros
#define LOG_STRUCTURED(level, message, context) \
    ultra::common::Logger::log_structured(level, message, context)

#define LOG_ERROR_WITH_EXCEPTION(message, exception, context) \
    ultra::common::Logger::log_error(message, exception, context)

#define LOG_PERFORMANCE(operation, duration_ns, context) \
    ultra::common::Logger::log_performance(operation, duration_ns, context)

#define LOG_SECURITY(event_type, description, context) \
    ultra::common::Logger::log_security_event(event_type, description, context)

#define LOG_AUDIT(action, user, resource, context) \
    ultra::common::Logger::log_audit(action, user, resource, context)

// Performance timing helper
class PerformanceTimer {
public:
    explicit PerformanceTimer(const std::string& operation_name, const LogContext& context = {})
        : operation_name_(operation_name), context_(context), 
          start_time_(std::chrono::high_resolution_clock::now()) {}
    
    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
        Logger::log_performance(operation_name_, duration.count(), context_);
    }

private:
    std::string operation_name_;
    LogContext context_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

#define PERFORMANCE_TIMER(operation_name) \
    ultra::common::PerformanceTimer _perf_timer(operation_name)

#define PERFORMANCE_TIMER_WITH_CONTEXT(operation_name, context) \
    ultra::common::PerformanceTimer _perf_timer(operation_name, context)

} // namespace common
} // namespace ultra