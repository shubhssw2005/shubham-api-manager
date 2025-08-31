#pragma once

#include <stdexcept>
#include <string>
#include <system_error>

namespace ultra {
namespace common {

// Base exception class for ultra system
class UltraException : public std::runtime_error {
public:
    explicit UltraException(const std::string& message) 
        : std::runtime_error(message) {}
};

// GPU-specific exceptions
class GPUException : public UltraException {
public:
    explicit GPUException(const std::string& message)
        : UltraException("GPU Error: " + message) {}
};

class CUDAException : public GPUException {
public:
    explicit CUDAException(const std::string& message, int error_code = 0)
        : GPUException(message + " (CUDA Error: " + std::to_string(error_code) + ")"),
          error_code_(error_code) {}
    
    int get_error_code() const { return error_code_; }

private:
    int error_code_;
};

// Memory-related exceptions
class MemoryException : public UltraException {
public:
    explicit MemoryException(const std::string& message)
        : UltraException("Memory Error: " + message) {}
};

class OutOfMemoryException : public MemoryException {
public:
    explicit OutOfMemoryException(size_t requested_size)
        : MemoryException("Out of memory (requested: " + std::to_string(requested_size) + " bytes)"),
          requested_size_(requested_size) {}
    
    size_t get_requested_size() const { return requested_size_; }

private:
    size_t requested_size_;
};

// Performance-related exceptions
class PerformanceException : public UltraException {
public:
    explicit PerformanceException(const std::string& message)
        : UltraException("Performance Error: " + message) {}
};

class TimeoutException : public PerformanceException {
public:
    explicit TimeoutException(const std::string& operation, uint64_t timeout_ms)
        : PerformanceException("Operation '" + operation + "' timed out after " + 
                              std::to_string(timeout_ms) + "ms"),
          timeout_ms_(timeout_ms) {}
    
    uint64_t get_timeout() const { return timeout_ms_; }

private:
    uint64_t timeout_ms_;
};

// Configuration exceptions
class ConfigException : public UltraException {
public:
    explicit ConfigException(const std::string& message)
        : UltraException("Configuration Error: " + message) {}
};

// Error handling utilities
class ErrorHandler {
public:
    using ErrorCallback = std::function<void(const std::exception&)>;
    
    static void set_global_error_handler(ErrorCallback callback);
    static void handle_error(const std::exception& e);
    
    // RAII error scope
    class ErrorScope {
    public:
        explicit ErrorScope(ErrorCallback callback);
        ~ErrorScope();
    private:
        ErrorCallback previous_callback_;
    };

private:
    static ErrorCallback global_callback_;
};

// Utility macros for error handling
#define ULTRA_THROW_IF(condition, exception_type, message) \
    do { \
        if (condition) { \
            throw exception_type(message); \
        } \
    } while(0)

#define ULTRA_ASSERT(condition, message) \
    ULTRA_THROW_IF(!(condition), ultra::common::UltraException, message)

#define ULTRA_CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw ultra::common::CUDAException(cudaGetErrorString(error), error); \
        } \
    } while(0)

} // namespace common
} // namespace ultra