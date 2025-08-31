#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "security/jwt_validator.hpp"
#include "security/rate_limiter.hpp"
#include "security/request_sanitizer.hpp"
#include "security/audit_logger.hpp"
#include <chrono>
#include <thread>

using namespace ultra_cpp::security;

class SecurityIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test configurations
        jwt_config_.issuer = "test-issuer";
        jwt_config_.audience = "test-audience";
        jwt_config_.clock_skew_tolerance = std::chrono::seconds(300);
        
        rate_limiter_config_.default_requests_per_second = 100;
        rate_limiter_config_.default_burst_capacity = 200;
        rate_limiter_config_.max_tenants = 1000;
        
        sanitizer_config_.level = SanitizationLevel::STRICT;
        sanitizer_config_.max_length = 4096;
        sanitizer_config_.validate_utf8 = true;
        
        audit_config_.storage_config.storage_path = "/tmp/test_audit";
        audit_config_.storage_config.max_file_size = 1024 * 1024;  // 1MB
        audit_config_.async_logging = false;  // Synchronous for testing
    }
    
    void TearDown() override {
        // Cleanup test files
        std::filesystem::remove_all("/tmp/test_audit");
    }
    
    JWTValidator::Config jwt_config_;
    RateLimiter::GlobalConfig rate_limiter_config_;
    SanitizationConfig sanitizer_config_;
    AuditLogger::Config audit_config_;
};

TEST_F(SecurityIntegrationTest, JWTValidatorBasicFunctionality) {
    JWTValidator validator(jwt_config_);
    
    // Test with invalid token format
    EXPECT_FALSE(validator.is_token_valid("invalid.token"));
    EXPECT_FALSE(validator.is_token_valid("invalid"));
    EXPECT_FALSE(validator.is_token_valid(""));
    
    // Test stats
    auto stats = validator.get_stats();
    EXPECT_GT(stats.tokens_validated.load(), 0);
    EXPECT_GT(stats.validation_failures.load(), 0);
}

TEST_F(SecurityIntegrationTest, RateLimiterFunctionality) {
    RateLimiter rate_limiter(rate_limiter_config_);
    
    // Add a test tenant
    RateLimiter::TenantConfig tenant_config{
        .tenant_id = "test-tenant",
        .requests_per_second = 10,
        .burst_capacity = 20,
        .enabled = true
    };
    
    EXPECT_TRUE(rate_limiter.add_tenant(tenant_config));
    EXPECT_TRUE(rate_limiter.is_tenant_enabled("test-tenant"));
    
    // Test rate limiting
    int allowed_count = 0;
    for (int i = 0; i < 25; ++i) {
        if (rate_limiter.is_allowed("test-tenant")) {
            allowed_count++;
        }
    }
    
    // Should allow up to burst capacity
    EXPECT_LE(allowed_count, 20);
    EXPECT_GT(allowed_count, 0);
    
    // Test bulk operations
    std::vector<RateLimiter::BulkRequest> bulk_requests;
    for (int i = 0; i < 5; ++i) {
        bulk_requests.push_back({"test-tenant", "", 1});
    }
    
    auto bulk_results = rate_limiter.check_bulk(bulk_requests);
    EXPECT_EQ(bulk_results.size(), 5);
    
    // Test stats
    auto stats = rate_limiter.get_global_stats();
    EXPECT_GT(stats.total_requests.load(), 0);
}

TEST_F(SecurityIntegrationTest, LockFreeRateLimiterPerformance) {
    LockFreeRateLimiter::Config config{
        .max_tenants = 1000,
        .default_rate = 1000,
        .default_burst = 2000
    };
    
    LockFreeRateLimiter rate_limiter(config);
    
    // Performance test
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int allowed_count = 0;
    const int test_requests = 10000;
    
    for (int i = 0; i < test_requests; ++i) {
        uint64_t tenant_hash = std::hash<int>{}(i % 100);  // 100 different tenants
        if (rate_limiter.is_allowed(tenant_hash)) {
            allowed_count++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Should process requests very quickly
    EXPECT_LT(duration.count(), 10000);  // Less than 10ms for 10k requests
    EXPECT_GT(allowed_count, 0);
    
    auto stats = rate_limiter.get_stats();
    EXPECT_EQ(stats.requests_processed.load(), test_requests);
}

TEST_F(SecurityIntegrationTest, RequestSanitizerValidation) {
    RequestSanitizer sanitizer(sanitizer_config_);
    
    // Test valid inputs
    EXPECT_EQ(sanitizer.validate_string("hello world"), ValidationResult::VALID);
    EXPECT_EQ(sanitizer.validate_string("user@example.com"), ValidationResult::VALID);
    EXPECT_EQ(sanitizer.validate_string("123-456-789"), ValidationResult::VALID);
    
    // Test invalid inputs
    EXPECT_EQ(sanitizer.validate_string("<script>alert('xss')</script>"), 
              ValidationResult::INVALID_CHARACTERS);
    EXPECT_EQ(sanitizer.validate_string("'; DROP TABLE users; --"), 
              ValidationResult::SUSPICIOUS_PATTERN);
    
    // Test length limits
    std::string long_string(5000, 'a');
    EXPECT_EQ(sanitizer.validate_string(long_string), ValidationResult::TOO_LONG);
    
    // Test sanitization
    std::string malicious = "<script>alert('xss')</script>";
    std::string sanitized = sanitizer.sanitize_string(malicious);
    EXPECT_NE(sanitized, malicious);
    EXPECT_EQ(sanitizer.validate_string(sanitized), ValidationResult::VALID);
    
    // Test specific validators
    EXPECT_TRUE(sanitizer.is_safe_email("user@example.com"));
    EXPECT_FALSE(sanitizer.is_safe_email("invalid-email"));
    EXPECT_TRUE(sanitizer.is_safe_filename("document.pdf"));
    EXPECT_FALSE(sanitizer.is_safe_filename("../../../etc/passwd"));
    
    // Test bulk validation
    std::vector<RequestSanitizer::BulkValidationRequest> bulk_requests = {
        {"hello", "field1", false},
        {"<script>", "field2", false},
        {"valid@email.com", "field3", false}
    };
    
    auto bulk_results = sanitizer.validate_bulk(bulk_requests);
    EXPECT_EQ(bulk_results.size(), 3);
    EXPECT_EQ(bulk_results[0].result, ValidationResult::VALID);
    EXPECT_EQ(bulk_results[1].result, ValidationResult::INVALID_CHARACTERS);
    EXPECT_EQ(bulk_results[2].result, ValidationResult::VALID);
    
    // Test stats
    auto stats = sanitizer.get_stats();
    EXPECT_GT(stats.strings_processed.load(), 0);
}

TEST_F(SecurityIntegrationTest, SIMDStringProcessorPerformance) {
    const std::string test_data = "This is a test string with some <script> tags and SQL injection attempts like UNION SELECT";
    const size_t iterations = 100000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        SIMDStringProcessor::contains_suspicious_chars(test_data.c_str(), test_data.length());
        SIMDStringProcessor::has_sql_injection_patterns(test_data.c_str(), test_data.length());
        SIMDStringProcessor::has_xss_patterns(test_data.c_str(), test_data.length());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Should be very fast with SIMD acceleration
    EXPECT_LT(duration.count(), 100000);  // Less than 100ms for 100k iterations
    
    // Test correctness
    EXPECT_TRUE(SIMDStringProcessor::contains_suspicious_chars(test_data.c_str(), test_data.length()));
    EXPECT_TRUE(SIMDStringProcessor::has_sql_injection_patterns(test_data.c_str(), test_data.length()));
    EXPECT_TRUE(SIMDStringProcessor::has_xss_patterns(test_data.c_str(), test_data.length()));
    
    // Test UTF-8 validation
    std::string valid_utf8 = "Hello, 世界! 🌍";
    EXPECT_TRUE(SIMDStringProcessor::is_valid_utf8_simd(valid_utf8.c_str(), valid_utf8.length()));
    
    std::string invalid_utf8 = "Hello\xFF\xFE";
    EXPECT_FALSE(SIMDStringProcessor::is_valid_utf8_simd(invalid_utf8.c_str(), invalid_utf8.length()));
}

TEST_F(SecurityIntegrationTest, AuditLoggerFunctionality) {
    // Create audit logger
    AuditLogger audit_logger(audit_config_);
    
    // Test various event types
    audit_logger.log_authentication_success("user123", "192.168.1.100");
    audit_logger.log_authentication_failure("user456", "192.168.1.101", "Invalid password");
    audit_logger.log_authorization_failure("user123", "/admin/users", "GET");
    audit_logger.log_rate_limit_exceeded("tenant1", "192.168.1.102");
    audit_logger.log_suspicious_request("Potential SQL injection", "192.168.1.103");
    audit_logger.log_injection_attempt(SecurityEventType::SQL_INJECTION_ATTEMPT, 
                                     "'; DROP TABLE users; --", "192.168.1.104");
    
    // Custom event with metadata
    std::unordered_map<std::string, std::string> metadata = {
        {"component", "api-gateway"},
        {"version", "1.0.0"},
        {"severity", "high"}
    };
    audit_logger.log_custom_event("custom_security_event", metadata);
    
    // Flush to ensure all events are written
    audit_logger.flush();
    
    // Test stats
    auto stats = audit_logger.get_stats();
    EXPECT_GE(stats.events_logged.load(), 7);  // At least 7 events logged
    
    // Test integrity verification
    EXPECT_TRUE(audit_logger.verify_log_integrity());
    
    // Test integrity report generation
    std::string report = audit_logger.generate_integrity_report();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("integrity_valid"), std::string::npos);
}

TEST_F(SecurityIntegrationTest, TamperEvidenceEngine) {
    TamperEvidenceEngine::Config config{
        .hmac_key = "test-secret-key-for-hmac-operations",
        .hash_algorithm = "SHA256",
        .enable_chain_validation = true
    };
    
    TamperEvidenceEngine engine(config);
    
    // Create test events
    SecurityEvent event1;
    event1.event_id = 1;
    event1.event_type = SecurityEventType::AUTHENTICATION_SUCCESS;
    event1.severity = SecurityLevel::LOW;
    event1.timestamp = std::chrono::system_clock::now();
    event1.user_id = "user123";
    event1.sequence_number = 1;
    
    SecurityEvent event2;
    event2.event_id = 2;
    event2.event_type = SecurityEventType::DATA_ACCESS;
    event2.severity = SecurityLevel::MEDIUM;
    event2.timestamp = std::chrono::system_clock::now();
    event2.user_id = "user456";
    event2.sequence_number = 2;
    
    // Compute hashes
    event1.hash = engine.compute_event_hash(event1);
    event2.previous_hash = event1.hash;
    event2.hash = engine.compute_event_hash(event2);
    
    // Validate individual events
    EXPECT_TRUE(engine.validate_event_integrity(event1));
    EXPECT_TRUE(engine.validate_event_integrity(event2));
    
    // Validate chain
    std::vector<SecurityEvent> chain = {event1, event2};
    EXPECT_TRUE(engine.validate_chain_integrity(chain));
    
    // Test HMAC operations
    std::string test_data = "sensitive audit data";
    std::string hmac = engine.compute_hmac(test_data);
    EXPECT_TRUE(engine.verify_hmac(test_data, hmac));
    EXPECT_FALSE(engine.verify_hmac("tampered data", hmac));
    
    // Test key rotation
    std::string old_fingerprint = engine.get_key_fingerprint();
    engine.rotate_hmac_key("new-secret-key");
    std::string new_fingerprint = engine.get_key_fingerprint();
    EXPECT_NE(old_fingerprint, new_fingerprint);
}

TEST_F(SecurityIntegrationTest, FastAuditLoggerPerformance) {
    FastAuditLogger::Config config{
        .ring_buffer_size = 10000,
        .storage_path = "/tmp/test_fast_audit",
        .memory_mapped_storage = true,
        .batch_size = 1000
    };
    
    FastAuditLogger fast_logger(config);
    
    // Performance test
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int num_events = 50000;
    int successful_logs = 0;
    
    for (int i = 0; i < num_events; ++i) {
        uint64_t user_hash = std::hash<int>{}(i % 1000);
        uint64_t resource_hash = std::hash<std::string>{}("resource_" + std::to_string(i % 100));
        uint32_t details_hash = std::hash<std::string>{}("details_" + std::to_string(i));
        
        if (fast_logger.log_event_fast(SecurityEventType::DATA_ACCESS, 
                                      user_hash, resource_hash, details_hash)) {
            successful_logs++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Should be extremely fast
    EXPECT_LT(duration.count(), 50000);  // Less than 50ms for 50k events
    EXPECT_GT(successful_logs, num_events * 0.9);  // At least 90% success rate
    
    // Flush to storage
    fast_logger.flush_to_storage();
    
    auto stats = fast_logger.get_stats();
    EXPECT_GE(stats.events_logged.load(), successful_logs);
    EXPECT_LT(stats.avg_log_time_ns.load(), 1000);  // Less than 1 microsecond per event
    
    // Cleanup
    std::filesystem::remove_all("/tmp/test_fast_audit");
}

TEST_F(SecurityIntegrationTest, IntegratedSecurityWorkflow) {
    // Create all security components
    JWTValidator jwt_validator(jwt_config_);
    RateLimiter rate_limiter(rate_limiter_config_);
    RequestSanitizer sanitizer(sanitizer_config_);
    AuditLogger audit_logger(audit_config_);
    
    // Add test tenant
    RateLimiter::TenantConfig tenant_config{
        .tenant_id = "integration-test",
        .requests_per_second = 50,
        .burst_capacity = 100,
        .enabled = true
    };
    rate_limiter.add_tenant(tenant_config);
    
    // Simulate request processing workflow
    std::vector<std::string> test_requests = {
        "normal request data",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "valid@email.com"
    };
    
    int processed_requests = 0;
    int blocked_requests = 0;
    
    for (const auto& request_data : test_requests) {
        // Step 1: Rate limiting check
        if (!rate_limiter.is_allowed("integration-test")) {
            audit_logger.log_rate_limit_exceeded("integration-test", "192.168.1.100");
            blocked_requests++;
            continue;
        }
        
        // Step 2: Request sanitization and validation
        auto validation_result = sanitizer.validate_string(request_data);
        if (validation_result != ValidationResult::VALID) {
            // Log security event based on validation result
            switch (validation_result) {
                case ValidationResult::SUSPICIOUS_PATTERN:
                    audit_logger.log_injection_attempt(SecurityEventType::SQL_INJECTION_ATTEMPT,
                                                     request_data, "192.168.1.100");
                    break;
                case ValidationResult::INVALID_CHARACTERS:
                    audit_logger.log_injection_attempt(SecurityEventType::XSS_ATTEMPT,
                                                     request_data, "192.168.1.100");
                    break;
                default:
                    audit_logger.log_suspicious_request("Invalid request: " + request_data,
                                                      "192.168.1.100");
                    break;
            }
            blocked_requests++;
            continue;
        }
        
        // Step 3: Process valid request
        std::string sanitized_data = sanitizer.sanitize_string(request_data);
        audit_logger.log_data_access("user123", "test-resource", "process_request");
        processed_requests++;
    }
    
    // Flush audit logs
    audit_logger.flush();
    
    // Verify results
    EXPECT_GT(processed_requests, 0);
    EXPECT_GT(blocked_requests, 0);
    EXPECT_EQ(processed_requests + blocked_requests, test_requests.size());
    
    // Verify audit log integrity
    EXPECT_TRUE(audit_logger.verify_log_integrity());
    
    // Check component statistics
    auto rate_limiter_stats = rate_limiter.get_global_stats();
    auto sanitizer_stats = sanitizer.get_stats();
    auto audit_stats = audit_logger.get_stats();
    
    EXPECT_GT(rate_limiter_stats.total_requests.load(), 0);
    EXPECT_GT(sanitizer_stats.strings_processed.load(), 0);
    EXPECT_GT(audit_stats.events_logged.load(), 0);
}

// Benchmark test for overall security performance
TEST_F(SecurityIntegrationTest, SecurityPerformanceBenchmark) {
    // Setup components
    RateLimiter rate_limiter(rate_limiter_config_);
    RequestSanitizer sanitizer(sanitizer_config_);
    
    RateLimiter::TenantConfig tenant_config{
        .tenant_id = "benchmark-test",
        .requests_per_second = 10000,
        .burst_capacity = 20000,
        .enabled = true
    };
    rate_limiter.add_tenant(tenant_config);
    
    const int num_requests = 100000;
    const std::string test_data = "benchmark test data with some special chars: <>&\"'";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int processed = 0;
    for (int i = 0; i < num_requests; ++i) {
        // Rate limiting check
        if (rate_limiter.is_allowed("benchmark-test")) {
            // Request validation
            if (sanitizer.validate_string(test_data) == ValidationResult::VALID) {
                processed++;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Performance expectations
    EXPECT_LT(duration.count(), 1000);  // Less than 1 second for 100k requests
    EXPECT_GT(processed, num_requests * 0.8);  // At least 80% processed
    
    // Calculate throughput
    double throughput = static_cast<double>(num_requests) / (duration.count() / 1000.0);
    EXPECT_GT(throughput, 100000);  // At least 100k requests per second
    
    std::cout << "Security benchmark results:" << std::endl;
    std::cout << "  Requests processed: " << processed << "/" << num_requests << std::endl;
    std::cout << "  Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(throughput) << " req/sec" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}