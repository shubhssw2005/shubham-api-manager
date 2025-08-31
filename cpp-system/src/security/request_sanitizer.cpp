#include "security/request_sanitizer.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <json/json.h>
#include <chrono>

namespace ultra_cpp::security {

// SIMD String Processor Implementation
bool SIMDStringProcessor::contains_suspicious_chars(const char* data, size_t length) {
    if (length == 0) return false;
    
    // Check for common suspicious characters using SIMD
    const __m256i suspicious_chars = _mm256_set_epi8(
        '<', '>', '&', '"', '\'', '(', ')', ';', 
        '|', '`', '$', '{', '}', '[', ']', '\\',
        '<', '>', '&', '"', '\'', '(', ')', ';', 
        '|', '`', '$', '{', '}', '[', ']', '\\'
    );
    
    size_t simd_length = (length / 32) * 32;
    
    for (size_t i = 0; i < simd_length; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        
        // Compare each byte with suspicious characters
        __m256i cmp_result = _mm256_cmpeq_epi8(chunk, suspicious_chars);
        
        if (!_mm256_testz_si256(cmp_result, cmp_result)) {
            return true;  // Found suspicious character
        }
    }
    
    // Check remaining bytes
    for (size_t i = simd_length; i < length; ++i) {
        char c = data[i];
        if (c == '<' || c == '>' || c == '&' || c == '"' || c == '\'' ||
            c == '(' || c == ')' || c == ';' || c == '|' || c == '`' ||
            c == '$' || c == '{' || c == '}' || c == '[' || c == ']' || c == '\\') {
            return true;
        }
    }
    
    return false;
}

bool SIMDStringProcessor::has_sql_injection_patterns(const char* data, size_t length) {
    // Common SQL injection patterns
    const std::vector<std::string> patterns = {
        "union", "select", "insert", "update", "delete", "drop", "create",
        "alter", "exec", "execute", "sp_", "xp_", "--", "/*", "*/"
    };
    
    return contains_any_pattern(data, length, patterns);
}

bool SIMDStringProcessor::has_xss_patterns(const char* data, size_t length) {
    // Common XSS patterns
    const std::vector<std::string> patterns = {
        "script", "javascript:", "vbscript:", "onload", "onerror", "onclick",
        "onmouseover", "onfocus", "onblur", "eval(", "alert(", "document."
    };
    
    return contains_any_pattern(data, length, patterns);
}

size_t SIMDStringProcessor::count_special_chars(const char* data, size_t length) {
    size_t count = 0;
    size_t simd_length = (length / 32) * 32;
    
    // Define special character ranges
    const __m256i min_special = _mm256_set1_epi8(32);   // Below space
    const __m256i max_special = _mm256_set1_epi8(126);  // Above tilde
    
    for (size_t i = 0; i < simd_length; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        
        // Check for characters outside printable ASCII range
        __m256i below_min = _mm256_cmpgt_epi8(min_special, chunk);
        __m256i above_max = _mm256_cmpgt_epi8(chunk, max_special);
        __m256i special_mask = _mm256_or_si256(below_min, above_max);
        
        // Count set bits
        uint32_t mask = _mm256_movemask_epi8(special_mask);
        count += __builtin_popcount(mask);
    }
    
    // Handle remaining bytes
    for (size_t i = simd_length; i < length; ++i) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c < 32 || c > 126) {
            count++;
        }
    }
    
    return count;
}

bool SIMDStringProcessor::is_valid_utf8_simd(const char* data, size_t length) {
    // Simplified UTF-8 validation using SIMD
    // This is a basic implementation - full UTF-8 validation is more complex
    
    size_t i = 0;
    while (i < length) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        
        if (c < 0x80) {
            // ASCII character
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            if (i + 1 >= length || (data[i + 1] & 0xC0) != 0x80) {
                return false;
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            if (i + 2 >= length || 
                (data[i + 1] & 0xC0) != 0x80 || 
                (data[i + 2] & 0xC0) != 0x80) {
                return false;
            }
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            if (i + 3 >= length || 
                (data[i + 1] & 0xC0) != 0x80 || 
                (data[i + 2] & 0xC0) != 0x80 || 
                (data[i + 3] & 0xC0) != 0x80) {
                return false;
            }
            i += 4;
        } else {
            return false;  // Invalid UTF-8 start byte
        }
    }
    
    return true;
}

bool SIMDStringProcessor::contains_any_pattern(const char* data, size_t length,
                                             const std::vector<std::string>& patterns) {
    std::string input(data, length);
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);
    
    for (const auto& pattern : patterns) {
        if (input.find(pattern) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string SIMDStringProcessor::replace_chars_simd(const std::string& input,
                                                   char target, char replacement) {
    std::string result = input;
    char* data = result.data();
    size_t length = result.length();
    
    const __m256i target_vec = _mm256_set1_epi8(target);
    const __m256i replacement_vec = _mm256_set1_epi8(replacement);
    
    size_t simd_length = (length / 32) * 32;
    
    for (size_t i = 0; i < simd_length; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i mask = _mm256_cmpeq_epi8(chunk, target_vec);
        __m256i replaced = _mm256_blendv_epi8(chunk, replacement_vec, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + i), replaced);
    }
    
    // Handle remaining bytes
    for (size_t i = simd_length; i < length; ++i) {
        if (data[i] == target) {
            data[i] = replacement;
        }
    }
    
    return result;
}

std::string SIMDStringProcessor::url_decode_simd(const std::string& input) {
    std::string result;
    result.reserve(input.length());
    
    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] == '%' && i + 2 < input.length()) {
            // Decode hex sequence
            char hex[3] = {input[i + 1], input[i + 2], '\0'};
            char* end;
            long value = std::strtol(hex, &end, 16);
            
            if (end == hex + 2) {  // Valid hex
                result += static_cast<char>(value);
                i += 2;
            } else {
                result += input[i];
            }
        } else if (input[i] == '+') {
            result += ' ';
        } else {
            result += input[i];
        }
    }
    
    return result;
}

std::string SIMDStringProcessor::html_encode_simd(const std::string& input) {
    std::string result;
    result.reserve(input.length() * 2);  // Estimate
    
    for (char c : input) {
        switch (c) {
            case '<':
                result += "&lt;";
                break;
            case '>':
                result += "&gt;";
                break;
            case '&':
                result += "&amp;";
                break;
            case '"':
                result += "&quot;";
                break;
            case '\'':
                result += "&#x27;";
                break;
            default:
                result += c;
                break;
        }
    }
    
    return result;
}

// RequestSanitizer Implementation
RequestSanitizer::RequestSanitizer(const SanitizationConfig& config) 
    : config_(config) {
    compile_patterns();
    compile_simd_patterns();
    
    LOG_INFO("RequestSanitizer initialized with {} level", 
             static_cast<int>(config_.level));
}

ValidationResult RequestSanitizer::validate_string(const std::string& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    stats_.strings_processed.fetch_add(1, std::memory_order_relaxed);
    
    // Check length
    if (input.length() > config_.max_length) {
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return ValidationResult::TOO_LONG;
    }
    
    if (input.empty()) {
        return ValidationResult::EMPTY_REQUIRED;
    }
    
    // Validate UTF-8 encoding
    if (config_.validate_utf8 && 
        !SIMDStringProcessor::is_valid_utf8_simd(input.c_str(), input.length())) {
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return ValidationResult::MALFORMED_ENCODING;
    }
    
    // Check for suspicious characters
    if (SIMDStringProcessor::contains_suspicious_chars(input.c_str(), input.length())) {
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return ValidationResult::INVALID_CHARACTERS;
    }
    
    // Check for injection patterns
    if (contains_sql_injection(input) || contains_xss_attempt(input) || 
        contains_path_traversal(input)) {
        stats_.suspicious_patterns_found.fetch_add(1, std::memory_order_relaxed);
        stats_.validation_failures.fetch_add(1, std::memory_order_relaxed);
        return ValidationResult::SUSPICIOUS_PATTERN;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    stats_.avg_processing_time_ns.store(duration.count(), std::memory_order_relaxed);
    
    return ValidationResult::VALID;
}

std::string RequestSanitizer::sanitize_string(const std::string& input) {
    stats_.strings_sanitized.fetch_add(1, std::memory_order_relaxed);
    
    std::string result = input;
    
    // Remove null bytes
    result = remove_null_bytes(result);
    
    // Normalize string
    if (config_.normalize_unicode) {
        result = normalize_string(result);
    }
    
    // HTML encode if not allowing HTML
    if (!config_.allow_html) {
        result = SIMDStringProcessor::html_encode_simd(result);
    }
    
    // Remove or encode dangerous patterns based on level
    switch (config_.level) {
        case SanitizationLevel::BASIC:
            // Basic XSS protection
            result = SIMDStringProcessor::replace_chars_simd(result, '<', '&');
            result = SIMDStringProcessor::replace_chars_simd(result, '>', '&');
            break;
            
        case SanitizationLevel::STRICT:
            // Strict filtering
            result = SIMDStringProcessor::html_encode_simd(result);
            break;
            
        case SanitizationLevel::PARANOID:
            // Maximum security - only allow alphanumeric and basic punctuation
            std::string safe_result;
            for (char c : result) {
                if (std::isalnum(c) || c == ' ' || c == '.' || c == ',' || 
                    c == '-' || c == '_') {
                    safe_result += c;
                }
            }
            result = safe_result;
            break;
    }
    
    // Truncate if too long
    if (result.length() > config_.max_length) {
        result = result.substr(0, config_.max_length);
    }
    
    return result;
}

bool RequestSanitizer::is_safe_filename(const std::string& filename) {
    // Check for path traversal
    if (filename.find("..") != std::string::npos ||
        filename.find("/") != std::string::npos ||
        filename.find("\\") != std::string::npos) {
        return false;
    }
    
    // Check for reserved names (Windows)
    const std::vector<std::string> reserved = {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
        "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3",
        "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    };
    
    std::string upper_filename = filename;
    std::transform(upper_filename.begin(), upper_filename.end(), 
                   upper_filename.begin(), ::toupper);
    
    for (const auto& reserved_name : reserved) {
        if (upper_filename == reserved_name) {
            return false;
        }
    }
    
    return validate_string(filename) == ValidationResult::VALID;
}

bool RequestSanitizer::is_safe_path(const std::string& path) {
    return !contains_path_traversal(path) && 
           validate_string(path) == ValidationResult::VALID;
}

bool RequestSanitizer::is_safe_email(const std::string& email) {
    // Basic email validation
    std::regex email_regex(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    return std::regex_match(email, email_regex) && 
           validate_string(email) == ValidationResult::VALID;
}

bool RequestSanitizer::is_safe_url(const std::string& url) {
    // Basic URL validation
    std::regex url_regex(R"(^https?://[a-zA-Z0-9.-]+(/.*)?$)");
    return std::regex_match(url, url_regex) && 
           validate_string(url) == ValidationResult::VALID;
}

bool RequestSanitizer::is_safe_json(const std::string& json) {
    try {
        Json::Value root;
        Json::Reader reader;
        return reader.parse(json, root) && 
               validate_string(json) == ValidationResult::VALID;
    } catch (const std::exception&) {
        return false;
    }
}

std::string RequestSanitizer::sanitize_header_value(const std::string& value) {
    std::string result = value;
    
    // Remove CRLF injection attempts
    result = SIMDStringProcessor::replace_chars_simd(result, '\r', ' ');
    result = SIMDStringProcessor::replace_chars_simd(result, '\n', ' ');
    
    return sanitize_string(result);
}

std::string RequestSanitizer::sanitize_query_parameter(const std::string& param) {
    // URL decode first
    std::string decoded = SIMDStringProcessor::url_decode_simd(param);
    return sanitize_string(decoded);
}

std::string RequestSanitizer::sanitize_form_data(const std::string& data) {
    return sanitize_string(data);
}

std::vector<RequestSanitizer::BulkValidationResult> RequestSanitizer::validate_bulk(
    const std::vector<BulkValidationRequest>& requests) {
    
    std::vector<BulkValidationResult> results;
    results.reserve(requests.size());
    
    for (const auto& request : requests) {
        BulkValidationResult result;
        result.result = validate_string(request.data);
        
        if (result.result == ValidationResult::VALID) {
            result.sanitized_data = sanitize_string(request.data);
        } else {
            result.error_message = "Validation failed for field: " + request.field_name;
        }
        
        results.push_back(result);
    }
    
    return results;
}

void RequestSanitizer::update_config(const SanitizationConfig& config) {
    config_ = config;
    compile_patterns();
    compile_simd_patterns();
    LOG_INFO("RequestSanitizer configuration updated");
}

void RequestSanitizer::reset_stats() {
    stats_.strings_processed.store(0, std::memory_order_relaxed);
    stats_.strings_sanitized.store(0, std::memory_order_relaxed);
    stats_.validation_failures.store(0, std::memory_order_relaxed);
    stats_.suspicious_patterns_found.store(0, std::memory_order_relaxed);
    stats_.avg_processing_time_ns.store(0, std::memory_order_relaxed);
}

bool RequestSanitizer::contains_sql_injection(const std::string& input) {
    return SIMDStringProcessor::has_sql_injection_patterns(input.c_str(), input.length());
}

bool RequestSanitizer::contains_xss_attempt(const std::string& input) {
    return SIMDStringProcessor::has_xss_patterns(input.c_str(), input.length());
}

bool RequestSanitizer::contains_path_traversal(const std::string& input) {
    return input.find("../") != std::string::npos ||
           input.find("..\\") != std::string::npos ||
           input.find("/.") != std::string::npos ||
           input.find("\\.") != std::string::npos;
}

bool RequestSanitizer::has_suspicious_encoding(const std::string& input) {
    // Check for various encoding attempts to bypass filters
    return input.find("%2e%2e") != std::string::npos ||  // ..
           input.find("%2f") != std::string::npos ||      // /
           input.find("%5c") != std::string::npos ||      // \
           input.find("%00") != std::string::npos;        // null byte
}

std::string RequestSanitizer::normalize_string(const std::string& input) {
    // Basic normalization - convert to lowercase and trim
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    
    // Trim whitespace
    result.erase(result.begin(), std::find_if(result.begin(), result.end(),
                                             [](unsigned char ch) { return !std::isspace(ch); }));
    result.erase(std::find_if(result.rbegin(), result.rend(),
                             [](unsigned char ch) { return !std::isspace(ch); }).base(),
                result.end());
    
    return result;
}

std::string RequestSanitizer::remove_null_bytes(const std::string& input) {
    std::string result;
    result.reserve(input.length());
    
    for (char c : input) {
        if (c != '\0') {
            result += c;
        }
    }
    
    return result;
}

std::string RequestSanitizer::decode_entities(const std::string& input) {
    std::string result = input;
    
    // Basic HTML entity decoding
    const std::vector<std::pair<std::string, std::string>> entities = {
        {"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}, 
        {"&quot;", "\""}, {"&#x27;", "'"}
    };
    
    for (const auto& [entity, replacement] : entities) {
        size_t pos = 0;
        while ((pos = result.find(entity, pos)) != std::string::npos) {
            result.replace(pos, entity.length(), replacement);
            pos += replacement.length();
        }
    }
    
    return result;
}

void RequestSanitizer::compile_patterns() {
    // Compile regex patterns for performance
    sql_injection_patterns_.clear();
    xss_patterns_.clear();
    path_traversal_patterns_.clear();
    
    // SQL injection patterns
    const std::vector<std::string> sql_patterns = {
        R"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        R"(--|\*\/|\/\*)",
        R"(\b(sp_|xp_)\w+)",
        R"(;\s*(drop|delete|update|insert))"
    };
    
    for (const auto& pattern : sql_patterns) {
        sql_injection_patterns_.emplace_back(pattern, std::regex::icase);
    }
    
    // XSS patterns
    const std::vector<std::string> xss_patterns = {
        R"(<\s*script\b)",
        R"(\bon\w+\s*=)",
        R"(javascript\s*:)",
        R"(vbscript\s*:)",
        R"(\beval\s*\()",
        R"(\balert\s*\()"
    };
    
    for (const auto& pattern : xss_patterns) {
        xss_patterns_.emplace_back(pattern, std::regex::icase);
    }
    
    // Path traversal patterns
    const std::vector<std::string> path_patterns = {
        R"(\.\./)",
        R"(\.\.\\)",
        R"(%2e%2e%2f)",
        R"(%2e%2e%5c)"
    };
    
    for (const auto& pattern : path_patterns) {
        path_traversal_patterns_.emplace_back(pattern, std::regex::icase);
    }
}

void RequestSanitizer::compile_simd_patterns() {
    // Prepare patterns for SIMD processing
    simd_sql_patterns_ = {
        "union", "select", "insert", "update", "delete", "drop", "create",
        "alter", "exec", "execute", "sp_", "xp_", "--", "/*", "*/"
    };
    
    simd_xss_patterns_ = {
        "script", "javascript:", "vbscript:", "onload", "onerror", "onclick",
        "onmouseover", "onfocus", "onblur", "eval(", "alert(", "document."
    };
}

} // namespace ultra_cpp::security