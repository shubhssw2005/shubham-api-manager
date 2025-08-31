#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include "security/jwt_validator.hpp"
#include "security/rate_limiter.hpp"
#include "security/request_sanitizer.hpp"
#include "security/audit_logger.hpp"

using namespace ultra_cpp::security;

void demonstrate_jwt_validation() {
    std::cout << "\n=== JWT Validation Demo ===" << std::endl;
    
    JWTValidator::Config config{
        .issuer = "demo-issuer",
        .audience = "demo-audience",
        .clock_skew_tolerance = std::chrono::seconds(300)
    };
    
    JWTValidator validator(config);
    
    // Add a test RSA public key (in production, load from secure storage)
    std::string test_public_key = R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4f5wg5l2hKsTeNem/V41
fGnJm6gOdrj8ym3rFkEjWT2btf+FxKlaAWYt9/WJdOQoCIaOHNmkliXvHI7gOKVF
96+rXqzsHq0A5X2Zb11jwrB4c7SLxoiQy/UV5s0zTPy+hdrMcVBK8eOvPiTKXgM0
VfxuwTUtlrBdXywNtgHBuDQjb3X07pYrZ+5B8fBF/O6rVQlQinWEU6+HOB2oNOtm
iiscc4Ldh9SsQftXvbHWRH7r5KjEXN7LGAjIllGKQHrfbKJ6dcJALz/DQQiQiSr8
9QrBcKWXYgqDLLLdmf05B4+3adWNUMHJ1VLKlQIDAQAB
-----END PUBLIC KEY-----
)";
    
    validator.add_rsa_public_key("demo-key-1", test_public_key);
    
    // Test with invalid tokens
    std::vector<std::string> test_tokens = {
        "invalid.token.format",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
        ""
    };
    
    for (const auto& token : test_tokens) {
        bool is_valid = validator.is_token_valid(token);
        std::cout << "Token validation result: " << (is_valid ? "VALID" : "INVALID") << std::endl;
    }
    
    auto stats = validator.get_stats();
    std::cout << "JWT Validator Stats:" << std::endl;
    std::cout << "  Tokens validated: " << stats.tokens_validated.load() << std::endl;
    std::cout << "  Validation failures: " << stats.validation_failures.load() << std::endl;
    std::cout << "  Avg validation time: " << stats.avg_validation_time_ns.load() << " ns" << std::endl;
}

void demonstrate_rate_limiting() {
    std::cout << "\n=== Rate Limiting Demo ===" << std::endl;
    
    RateLimiter::GlobalConfig config{
        .default_requests_per_second = 10,
        .default_burst_capacity = 20,
        .max_tenants = 1000
    };
    
    RateLimiter rate_limiter(config);
    
    // Add specific tenant configurations
    RateLimiter::TenantConfig tenant1{
        .tenant_id = "premium-tenant",
        .requests_per_second = 100,
        .burst_capacity = 200,
        .enabled = true
    };
    
    RateLimiter::TenantConfig tenant2{
        .tenant_id = "basic-tenant",
        .requests_per_second = 10,
        .burst_capacity = 20,
        .enabled = true
    };
    
    rate_limiter.add_tenant(tenant1);
    rate_limiter.add_tenant(tenant2);
    
    // Simulate request bursts
    std::cout << "Testing premium tenant (100 RPS, 200 burst):" << std::endl;
    int premium_allowed = 0;
    for (int i = 0; i < 250; ++i) {
        if (rate_limiter.is_allowed("premium-tenant")) {
            premium_allowed++;
        }
    }
    std::cout << "  Allowed: " << premium_allowed << "/250 requests" << std::endl;
    
    std::cout << "Testing basic tenant (10 RPS, 20 burst):" << std::endl;
    int basic_allowed = 0;
    for (int i = 0; i < 50; ++i) {
        if (rate_limiter.is_allowed("basic-tenant")) {
            basic_allowed++;
        }
    }
    std::cout << "  Allowed: " << basic_allowed << "/50 requests" << std::endl;
    
    // Test bulk operations
    std::vector<RateLimiter::BulkRequest> bulk_requests;
    for (int i = 0; i < 10; ++i) {
        bulk_requests.push_back({"premium-tenant", "user-" + std::to_string(i), 1});
    }
    
    auto bulk_results = rate_limiter.check_bulk(bulk_requests);
    int bulk_allowed = 0;
    for (const auto& result : bulk_results) {
        if (result.allowed) bulk_allowed++;
    }
    std::cout << "Bulk operation: " << bulk_allowed << "/10 requests allowed" << std::endl;
    
    auto stats = rate_limiter.get_global_stats();
    std::cout << "Rate Limiter Stats:" << std::endl;
    std::cout << "  Total tenants: " << stats.total_tenants.load() << std::endl;
    std::cout << "  Total requests: " << stats.total_requests.load() << std::endl;
    std::cout << "  Total allowed: " << stats.total_allowed.load() << std::endl;
    std::cout << "  Total denied: " << stats.total_denied.load() << std::endl;
}

void demonstrate_lock_free_rate_limiting() {
    std::cout << "\n=== Lock-Free Rate Limiting Performance Demo ===" << std::endl;
    
    LockFreeRateLimiter::Config config{
        .max_tenants = 1000,
        .default_rate = 10000,
        .default_burst = 20000
    };
    
    LockFreeRateLimiter rate_limiter(config);
    
    // Performance test
    const int num_requests = 1000000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int allowed_count = 0;
    for (int i = 0; i < num_requests; ++i) {
        uint64_t tenant_hash = std::hash<int>{}(i % 100);  // 100 different tenants
        if (rate_limiter.is_allowed(tenant_hash)) {
            allowed_count++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Lock-free rate limiter performance:" << std::endl;
    std::cout << "  Processed: " << num_requests << " requests" << std::endl;
    std::cout << "  Allowed: " << allowed_count << " requests" << std::endl;
    std::cout << "  Duration: " << duration.count() << " μs" << std::endl;
    std::cout << "  Throughput: " << (num_requests * 1000000.0 / duration.count()) << " req/sec" << std::endl;
    
    auto stats = rate_limiter.get_stats();
    std::cout << "  Avg processing time: " << stats.avg_processing_time_ns.load() << " ns" << std::endl;
}

void demonstrate_request_sanitization() {
    std::cout << "\n=== Request Sanitization Demo ===" << std::endl;
    
    SanitizationConfig config{
        .level = SanitizationLevel::STRICT,
        .max_length = 1000,
        .allow_html = false,
        .allow_javascript = false,
        .validate_utf8 = true
    };
    
    RequestSanitizer sanitizer(config);
    
    // Test various input types
    std::vector<std::pair<std::string, std::string>> test_inputs = {
        {"Normal text", "This is normal text input"},
        {"XSS attempt", "<script>alert('XSS')</script>"},
        {"SQL injection", "'; DROP TABLE users; --"},
        {"Path traversal", "../../../etc/passwd"},
        {"Valid email", "user@example.com"},
        {"Valid URL", "https://example.com/path"},
        {"Unicode text", "Hello, 世界! 🌍"},
        {"HTML content", "<p>This is <b>bold</b> text</p>"}
    };
    
    std::cout << "Input validation results:" << std::endl;
    for (const auto& [description, input] : test_inputs) {
        auto result = sanitizer.validate_string(input);
        std::string result_str;
        
        switch (result) {
            case ValidationResult::VALID:
                result_str = "VALID";
                break;
            case ValidationResult::INVALID_CHARACTERS:
                result_str = "INVALID_CHARACTERS";
                break;
            case ValidationResult::SUSPICIOUS_PATTERN:
                result_str = "SUSPICIOUS_PATTERN";
                break;
            case ValidationResult::TOO_LONG:
                result_str = "TOO_LONG";
                break;
            case ValidationResult::EMPTY_REQUIRED:
                result_str = "EMPTY_REQUIRED";
                break;
            case ValidationResult::MALFORMED_ENCODING:
                result_str = "MALFORMED_ENCODING";
                break;
        }
        
        std::cout << "  " << description << ": " << result_str << std::endl;
        
        if (result != ValidationResult::VALID) {
            std::string sanitized = sanitizer.sanitize_string(input);
            std::cout << "    Sanitized: " << sanitized << std::endl;
        }
    }
    
    // Test specific validators
    std::cout << "\nSpecific validation tests:" << std::endl;
    std::cout << "  Email 'user@example.com': " << 
                 (sanitizer.is_safe_email("user@example.com") ? "SAFE" : "UNSAFE") << std::endl;
    std::cout << "  Filename 'document.pdf': " << 
                 (sanitizer.is_safe_filename("document.pdf") ? "SAFE" : "UNSAFE") << std::endl;
    std::cout << "  Path '../etc/passwd': " << 
                 (sanitizer.is_safe_path("../etc/passwd") ? "SAFE" : "UNSAFE") << std::endl;
    
    auto stats = sanitizer.get_stats();
    std::cout << "Sanitizer Stats:" << std::endl;
    std::cout << "  Strings processed: " << stats.strings_processed.load() << std::endl;
    std::cout << "  Strings sanitized: " << stats.strings_sanitized.load() << std::endl;
    std::cout << "  Validation failures: " << stats.validation_failures.load() << std::endl;
    std::cout << "  Suspicious patterns found: " << stats.suspicious_patterns_found.load() << std::endl;
}

void demonstrate_simd_performance() {
    std::cout << "\n=== SIMD String Processing Performance Demo ===" << std::endl;
    
    const std::string test_data = R"(
        This is a test string with various content including:
        - Potential XSS: <script>alert('test')</script>
        - SQL injection: '; DROP TABLE users; --
        - Path traversal: ../../../etc/passwd
        - Unicode content: Hello, 世界! 🌍
        - Special characters: !@#$%^&*()_+-={}[]|;':\",./<>?
    )";
    
    const int iterations = 100000;
    
    // Test SIMD functions
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        SIMDStringProcessor::contains_suspicious_chars(test_data.c_str(), test_data.length());
        SIMDStringProcessor::has_sql_injection_patterns(test_data.c_str(), test_data.length());
        SIMDStringProcessor::has_xss_patterns(test_data.c_str(), test_data.length());
        SIMDStringProcessor::count_special_chars(test_data.c_str(), test_data.length());
        SIMDStringProcessor::is_valid_utf8_simd(test_data.c_str(), test_data.length());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "SIMD processing performance:" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Duration: " << duration.count() << " μs" << std::endl;
    std::cout << "  Avg per iteration: " << (duration.count() / static_cast<double>(iterations)) << " μs" << std::endl;
    
    // Test results
    std::cout << "\nSIMD detection results:" << std::endl;
    std::cout << "  Suspicious chars: " << 
                 (SIMDStringProcessor::contains_suspicious_chars(test_data.c_str(), test_data.length()) ? "YES" : "NO") << std::endl;
    std::cout << "  SQL injection patterns: " << 
                 (SIMDStringProcessor::has_sql_injection_patterns(test_data.c_str(), test_data.length()) ? "YES" : "NO") << std::endl;
    std::cout << "  XSS patterns: " << 
                 (SIMDStringProcessor::has_xss_patterns(test_data.c_str(), test_data.length()) ? "YES" : "NO") << std::endl;
    std::cout << "  Special char count: " << 
                 SIMDStringProcessor::count_special_chars(test_data.c_str(), test_data.length()) << std::endl;
    std::cout << "  Valid UTF-8: " << 
                 (SIMDStringProcessor::is_valid_utf8_simd(test_data.c_str(), test_data.length()) ? "YES" : "NO") << std::endl;
}

void demonstrate_audit_logging() {
    std::cout << "\n=== Audit Logging Demo ===" << std::endl;
    
    AuditLogger::Config config;
    config.storage_config.storage_path = "/tmp/demo_audit";
    config.storage_config.max_file_size = 1024 * 1024;  // 1MB
    config.async_logging = false;  // Synchronous for demo
    config.min_log_level = SecurityLevel::LOW;
    
    // Create storage directory
    std::filesystem::create_directories(config.storage_config.storage_path);
    
    AuditLogger audit_logger(config);
    
    // Log various security events
    std::cout << "Logging security events..." << std::endl;
    
    audit_logger.log_authentication_success("demo_user", "192.168.1.100");
    audit_logger.log_authentication_failure("invalid_user", "192.168.1.101", "Invalid credentials");
    audit_logger.log_authorization_failure("demo_user", "/admin/users", "GET");
    audit_logger.log_rate_limit_exceeded("demo_tenant", "192.168.1.102");
    audit_logger.log_suspicious_request("Potential attack detected", "192.168.1.103");
    audit_logger.log_injection_attempt(SecurityEventType::SQL_INJECTION_ATTEMPT, 
                                     "'; DROP TABLE users; --", "192.168.1.104");
    audit_logger.log_injection_attempt(SecurityEventType::XSS_ATTEMPT, 
                                     "<script>alert('xss')</script>", "192.168.1.105");
    audit_logger.log_data_access("demo_user", "sensitive_data", "READ");
    audit_logger.log_configuration_change("admin_user", "security_settings", "Updated rate limits");
    audit_logger.log_system_error("authentication_service", "Database connection timeout");
    
    // Custom event with metadata
    std::unordered_map<std::string, std::string> metadata = {
        {"component", "demo_app"},
        {"version", "1.0.0"},
        {"environment", "development"}
    };
    audit_logger.log_custom_event("demo_event", metadata);
    
    // Flush logs
    audit_logger.flush();
    
    // Verify integrity
    bool integrity_valid = audit_logger.verify_log_integrity();
    std::cout << "Log integrity: " << (integrity_valid ? "VALID" : "INVALID") << std::endl;
    
    // Generate integrity report
    std::string report = audit_logger.generate_integrity_report();
    std::cout << "Integrity report generated (" << report.length() << " bytes)" << std::endl;
    
    auto stats = audit_logger.get_stats();
    std::cout << "Audit Logger Stats:" << std::endl;
    std::cout << "  Events logged: " << stats.events_logged.load() << std::endl;
    std::cout << "  Events dropped: " << stats.events_dropped.load() << std::endl;
    std::cout << "  Flush operations: " << stats.flush_operations.load() << std::endl;
    std::cout << "  Integrity checks: " << stats.integrity_checks.load() << std::endl;
    std::cout << "  Avg log time: " << stats.avg_log_time_ns.load() << " ns" << std::endl;
    
    // Cleanup
    std::filesystem::remove_all("/tmp/demo_audit");
}

void demonstrate_fast_audit_logging() {
    std::cout << "\n=== Fast Audit Logging Performance Demo ===" << std::endl;
    
    FastAuditLogger::Config config{
        .ring_buffer_size = 100000,
        .storage_path = "/tmp/demo_fast_audit",
        .memory_mapped_storage = true,
        .batch_size = 1000
    };
    
    FastAuditLogger fast_logger(config);
    
    // Performance test
    const int num_events = 500000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int successful_logs = 0;
    for (int i = 0; i < num_events; ++i) {
        uint64_t user_hash = std::hash<int>{}(i % 1000);
        uint64_t resource_hash = std::hash<std::string>{}("resource_" + std::to_string(i % 100));
        uint32_t details_hash = std::hash<std::string>{}("operation_" + std::to_string(i));
        
        if (fast_logger.log_event_fast(SecurityEventType::DATA_ACCESS, 
                                      user_hash, resource_hash, details_hash)) {
            successful_logs++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Fast audit logging performance:" << std::endl;
    std::cout << "  Events: " << num_events << std::endl;
    std::cout << "  Successful: " << successful_logs << std::endl;
    std::cout << "  Duration: " << duration.count() << " μs" << std::endl;
    std::cout << "  Throughput: " << (num_events * 1000000.0 / duration.count()) << " events/sec" << std::endl;
    
    // Flush to storage
    fast_logger.flush_to_storage();
    
    auto stats = fast_logger.get_stats();
    std::cout << "Fast Logger Stats:" << std::endl;
    std::cout << "  Events logged: " << stats.events_logged.load() << std::endl;
    std::cout << "  Events dropped: " << stats.events_dropped.load() << std::endl;
    std::cout << "  Avg log time: " << stats.avg_log_time_ns.load() << " ns" << std::endl;
    std::cout << "  Ring buffer wraps: " << stats.ring_buffer_wraps.load() << std::endl;
    
    // Cleanup
    std::filesystem::remove_all("/tmp/demo_fast_audit");
}

int main() {
    std::cout << "Ultra Low-Latency C++ Security System Demo" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        demonstrate_jwt_validation();
        demonstrate_rate_limiting();
        demonstrate_lock_free_rate_limiting();
        demonstrate_request_sanitization();
        demonstrate_simd_performance();
        demonstrate_audit_logging();
        demonstrate_fast_audit_logging();
        
        std::cout << "\n=== Demo Complete ===" << std::endl;
        std::cout << "All security components demonstrated successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}