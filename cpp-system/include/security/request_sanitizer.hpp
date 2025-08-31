#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <regex>
#include <memory>
#include <immintrin.h>  // For SIMD intrinsics

namespace ultra_cpp::security {

enum class SanitizationLevel {
    BASIC,      // Basic XSS and SQL injection protection
    STRICT,     // Strict validation with whitelist approach
    PARANOID    // Maximum security with extensive filtering
};

enum class ValidationResult {
    VALID,
    INVALID_CHARACTERS,
    SUSPICIOUS_PATTERN,
    TOO_LONG,
    EMPTY_REQUIRED,
    MALFORMED_ENCODING
};

struct SanitizationConfig {
    SanitizationLevel level = SanitizationLevel::BASIC;
    size_t max_length = 8192;
    bool allow_html = false;
    bool allow_javascript = false;
    bool allow_sql_keywords = false;
    bool normalize_unicode = true;
    bool validate_utf8 = true;
    std::unordered_set<std::string> allowed_tags;
    std::unordered_set<std::string> blocked_patterns;
};

class SIMDStringProcessor {
public:
    // SIMD-accelerated string operations
    static bool contains_suspicious_chars(const char* data, size_t length);
    static bool has_sql_injection_patterns(const char* data, size_t length);
    static bool has_xss_patterns(const char* data, size_t length);
    static size_t count_special_chars(const char* data, size_t length);
    static bool is_valid_utf8_simd(const char* data, size_t length);
    
    // SIMD string search for multiple patterns
    static bool contains_any_pattern(const char* data, size_t length, 
                                   const std::vector<std::string>& patterns);
    
    // Fast character replacement using SIMD
    static std::string replace_chars_simd(const std::string& input, 
                                        char target, char replacement);
    
    // URL encoding/decoding with SIMD
    static std::string url_decode_simd(const std::string& input);
    static std::string html_encode_simd(const std::string& input);

private:
    // SIMD helper functions
    static __m256i load_chars(const char* ptr);
    static bool check_range_simd(__m256i chars, char min_char, char max_char);
    static uint32_t find_pattern_simd(const char* haystack, size_t haystack_len,
                                    const char* needle, size_t needle_len);
};

class RequestSanitizer {
public:
    explicit RequestSanitizer(const SanitizationConfig& config);
    
    // Main sanitization methods
    ValidationResult validate_string(const std::string& input);
    std::string sanitize_string(const std::string& input);
    
    // Specific validation methods
    bool is_safe_filename(const std::string& filename);
    bool is_safe_path(const std::string& path);
    bool is_safe_email(const std::string& email);
    bool is_safe_url(const std::string& url);
    bool is_safe_json(const std::string& json);
    
    // HTTP-specific sanitization
    std::string sanitize_header_value(const std::string& value);
    std::string sanitize_query_parameter(const std::string& param);
    std::string sanitize_form_data(const std::string& data);
    
    // Bulk operations for performance
    struct BulkValidationRequest {
        std::string data;
        std::string field_name;
        bool required = false;
    };
    
    struct BulkValidationResult {
        ValidationResult result;
        std::string sanitized_data;
        std::string error_message;
    };
    
    std::vector<BulkValidationResult> validate_bulk(
        const std::vector<BulkValidationRequest>& requests);
    
    // Configuration management
    void update_config(const SanitizationConfig& config);
    const SanitizationConfig& get_config() const { return config_; }
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> strings_processed{0};
        std::atomic<uint64_t> strings_sanitized{0};
        std::atomic<uint64_t> validation_failures{0};
        std::atomic<uint64_t> suspicious_patterns_found{0};
        std::atomic<uint64_t> avg_processing_time_ns{0};
    };
    
    Stats get_stats() const { return stats_; }
    void reset_stats();

private:
    SanitizationConfig config_;
    mutable Stats stats_;
    
    // Compiled regex patterns for performance
    std::vector<std::regex> sql_injection_patterns_;
    std::vector<std::regex> xss_patterns_;
    std::vector<std::regex> path_traversal_patterns_;
    
    // Precompiled pattern strings for SIMD processing
    std::vector<std::string> simd_sql_patterns_;
    std::vector<std::string> simd_xss_patterns_;
    
    // Internal validation methods
    bool contains_sql_injection(const std::string& input);
    bool contains_xss_attempt(const std::string& input);
    bool contains_path_traversal(const std::string& input);
    bool has_suspicious_encoding(const std::string& input);
    
    // String processing helpers
    std::string normalize_string(const std::string& input);
    std::string remove_null_bytes(const std::string& input);
    std::string decode_entities(const std::string& input);
    
    // Pattern compilation
    void compile_patterns();
    void compile_simd_patterns();
};

class InputValidator {
public:
    struct FieldRule {
        std::string field_name;
        bool required = false;
        size_t min_length = 0;
        size_t max_length = SIZE_MAX;
        std::regex pattern;
        std::function<bool(const std::string&)> custom_validator;
    };
    
    struct ValidationSchema {
        std::vector<FieldRule> rules;
        bool strict_mode = false;  // Reject unknown fields
    };
    
    explicit InputValidator(const ValidationSchema& schema);
    
    // Validate structured input (JSON, form data, etc.)
    struct ValidationError {
        std::string field_name;
        std::string error_message;
        ValidationResult error_type;
    };
    
    struct ValidationReport {
        bool is_valid;
        std::vector<ValidationError> errors;
        std::unordered_map<std::string, std::string> sanitized_values;
    };
    
    ValidationReport validate_json(const std::string& json_input);
    ValidationReport validate_form_data(const std::unordered_map<std::string, std::string>& data);
    ValidationReport validate_query_params(const std::unordered_map<std::string, std::string>& params);
    
    // Schema management
    void update_schema(const ValidationSchema& schema);
    void add_field_rule(const FieldRule& rule);
    void remove_field_rule(const std::string& field_name);
    
    // Performance monitoring
    struct ValidatorStats {
        std::atomic<uint64_t> validations_performed{0};
        std::atomic<uint64_t> validation_failures{0};
        std::atomic<uint64_t> fields_processed{0};
        std::atomic<uint64_t> avg_validation_time_ns{0};
    };
    
    ValidatorStats get_stats() const { return stats_; }

private:
    ValidationSchema schema_;
    RequestSanitizer sanitizer_;
    mutable ValidatorStats stats_;
    
    ValidationError validate_field(const std::string& field_name, 
                                 const std::string& value,
                                 const FieldRule& rule);
};

// High-performance batch processor for request sanitization
class BatchSanitizer {
public:
    struct BatchConfig {
        size_t batch_size = 1000;
        size_t worker_threads = std::thread::hardware_concurrency();
        bool use_simd = true;
    };
    
    explicit BatchSanitizer(const BatchConfig& config);
    ~BatchSanitizer();
    
    // Batch processing
    struct BatchItem {
        std::string data;
        std::string identifier;
        SanitizationLevel level;
    };
    
    struct BatchResult {
        std::string identifier;
        ValidationResult result;
        std::string sanitized_data;
        std::chrono::nanoseconds processing_time;
    };
    
    std::vector<BatchResult> process_batch(const std::vector<BatchItem>& items);
    
    // Async processing
    using BatchCallback = std::function<void(std::vector<BatchResult>)>;
    void process_batch_async(const std::vector<BatchItem>& items, BatchCallback callback);
    
    // Performance metrics
    struct BatchStats {
        std::atomic<uint64_t> batches_processed{0};
        std::atomic<uint64_t> items_processed{0};
        std::atomic<uint64_t> total_processing_time_ns{0};
        std::atomic<uint64_t> avg_batch_size{0};
    };
    
    BatchStats get_stats() const { return stats_; }

private:
    BatchConfig config_;
    mutable BatchStats stats_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_{false};
    
    void worker_thread_func();
    BatchResult process_single_item(const BatchItem& item);
};

} // namespace ultra_cpp::security