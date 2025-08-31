#include "security/audit_logger.hpp"
#include "common/logger.hpp"
#include <json/json.h>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace ultra_cpp::security {

// TamperEvidenceEngine Implementation
TamperEvidenceEngine::TamperEvidenceEngine(const Config& config) 
    : config_(config), current_hmac_key_(config.hmac_key) {
    
    if (current_hmac_key_.empty()) {
        // Generate a random key if none provided
        unsigned char key_bytes[32];
        if (RAND_bytes(key_bytes, sizeof(key_bytes)) == 1) {
            current_hmac_key_ = std::string(reinterpret_cast<char*>(key_bytes), sizeof(key_bytes));
        }
    }
    
    LOG_INFO("TamperEvidenceEngine initialized with {} algorithm", config_.hash_algorithm);
}

std::string TamperEvidenceEngine::compute_event_hash(const SecurityEvent& event) {
    std::string serialized = serialize_event_for_hash(event);
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, serialized.c_str(), serialized.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return ss.str();
}

std::string TamperEvidenceEngine::compute_chain_hash(const std::string& current_hash,
                                                    const std::string& previous_hash) {
    std::string combined = previous_hash + current_hash;
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, combined.c_str(), combined.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return ss.str();
}

bool TamperEvidenceEngine::validate_event_integrity(const SecurityEvent& event) {
    std::string computed_hash = compute_event_hash(event);
    return computed_hash == event.hash;
}

bool TamperEvidenceEngine::validate_chain_integrity(const std::vector<SecurityEvent>& events) {
    if (events.empty()) return true;
    
    for (size_t i = 1; i < events.size(); ++i) {
        const auto& current = events[i];
        const auto& previous = events[i - 1];
        
        // Verify current event hash
        if (!validate_event_integrity(current)) {
            LOG_ERROR("Event integrity validation failed for event {}", current.event_id);
            return false;
        }
        
        // Verify chain linkage
        if (current.previous_hash != previous.hash) {
            LOG_ERROR("Chain integrity validation failed between events {} and {}", 
                     previous.event_id, current.event_id);
            return false;
        }
        
        // Verify sequence numbers
        if (current.sequence_number != previous.sequence_number + 1) {
            LOG_ERROR("Sequence number validation failed for event {}", current.event_id);
            return false;
        }
    }
    
    return true;
}

std::string TamperEvidenceEngine::compute_hmac(const std::string& data) {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int result_len;
    
    HMAC(EVP_sha256(), 
         current_hmac_key_.c_str(), current_hmac_key_.length(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.length(),
         result, &result_len);
    
    std::stringstream ss;
    for (unsigned int i = 0; i < result_len; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(result[i]);
    }
    
    return ss.str();
}

bool TamperEvidenceEngine::verify_hmac(const std::string& data, const std::string& hmac) {
    std::string computed_hmac = compute_hmac(data);
    return computed_hmac == hmac;
}

void TamperEvidenceEngine::rotate_hmac_key(const std::string& new_key) {
    current_hmac_key_ = new_key;
    LOG_INFO("HMAC key rotated, new fingerprint: {}", get_key_fingerprint());
}

std::string TamperEvidenceEngine::get_key_fingerprint() const {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, current_hmac_key_.c_str(), current_hmac_key_.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < 8; ++i) {  // First 8 bytes for fingerprint
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return ss.str();
}

std::string TamperEvidenceEngine::serialize_event_for_hash(const SecurityEvent& event) {
    Json::Value json;
    json["event_id"] = static_cast<Json::UInt64>(event.event_id);
    json["event_type"] = static_cast<int>(event.event_type);
    json["severity"] = static_cast<int>(event.severity);
    json["timestamp"] = static_cast<Json::UInt64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            event.timestamp.time_since_epoch()).count());
    json["user_id"] = event.user_id;
    json["tenant_id"] = event.tenant_id;
    json["source_ip"] = event.source_ip;
    json["user_agent"] = event.user_agent;
    json["resource"] = event.resource;
    json["action"] = event.action;
    json["details"] = event.details;
    json["sequence_number"] = static_cast<Json::UInt64>(event.sequence_number);
    
    // Add metadata
    for (const auto& [key, value] : event.metadata) {
        json["metadata"][key] = value;
    }
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";  // Compact format
    return Json::writeString(builder, json);
}

// AuditStorage Implementation
AuditStorage::AuditStorage(const StorageConfig& config) : config_(config) {
    // Create storage directories
    std::filesystem::create_directories(config_.storage_path);
    std::filesystem::create_directories(config_.backup_path);
    
    // Open initial log file
    if (!open_new_file()) {
        throw std::runtime_error("Failed to open initial audit log file");
    }
    
    LOG_INFO("AuditStorage initialized at {}", config_.storage_path);
}

AuditStorage::~AuditStorage() {
    close_current_file();
}

bool AuditStorage::store_event(const SecurityEvent& event) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    try {
        std::string serialized = serialize_event(event);
        
        if (config_.encrypt_storage) {
            serialized = encrypt_data(serialized);
        }
        
        current_file_ << serialized << std::endl;
        current_file_.flush();
        
        current_file_size_.fetch_add(serialized.length() + 1, std::memory_order_relaxed);
        stats_.events_stored.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_written.fetch_add(serialized.length() + 1, std::memory_order_relaxed);
        
        // Check if file rotation is needed
        if (current_file_size_.load() >= config_.max_file_size) {
            rotate_log_files();
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to store audit event: {}", e.what());
        stats_.storage_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

bool AuditStorage::store_events_batch(const std::vector<SecurityEvent>& events) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    try {
        size_t total_size = 0;
        
        for (const auto& event : events) {
            std::string serialized = serialize_event(event);
            
            if (config_.encrypt_storage) {
                serialized = encrypt_data(serialized);
            }
            
            current_file_ << serialized << std::endl;
            total_size += serialized.length() + 1;
        }
        
        current_file_.flush();
        
        current_file_size_.fetch_add(total_size, std::memory_order_relaxed);
        stats_.events_stored.fetch_add(events.size(), std::memory_order_relaxed);
        stats_.bytes_written.fetch_add(total_size, std::memory_order_relaxed);
        
        // Check if file rotation is needed
        if (current_file_size_.load() >= config_.max_file_size) {
            rotate_log_files();
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to store audit events batch: {}", e.what());
        stats_.storage_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

std::vector<SecurityEvent> AuditStorage::get_events_by_time_range(
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end) {
    
    std::vector<SecurityEvent> events;
    
    // This is a simplified implementation
    // In production, you'd want to index by timestamp for efficient queries
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                std::ifstream file(entry.path());
                std::string line;
                
                while (std::getline(file, line)) {
                    if (config_.encrypt_storage) {
                        line = decrypt_data(line);
                    }
                    
                    SecurityEvent event = deserialize_event(line);
                    
                    if (event.timestamp >= start && event.timestamp <= end) {
                        events.push_back(event);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to query events by time range: {}", e.what());
    }
    
    return events;
}

std::vector<SecurityEvent> AuditStorage::get_events_by_user(const std::string& user_id) {
    std::vector<SecurityEvent> events;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                std::ifstream file(entry.path());
                std::string line;
                
                while (std::getline(file, line)) {
                    if (config_.encrypt_storage) {
                        line = decrypt_data(line);
                    }
                    
                    SecurityEvent event = deserialize_event(line);
                    
                    if (event.user_id == user_id) {
                        events.push_back(event);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to query events by user: {}", e.what());
    }
    
    return events;
}

std::vector<SecurityEvent> AuditStorage::get_events_by_type(SecurityEventType event_type) {
    std::vector<SecurityEvent> events;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                std::ifstream file(entry.path());
                std::string line;
                
                while (std::getline(file, line)) {
                    if (config_.encrypt_storage) {
                        line = decrypt_data(line);
                    }
                    
                    SecurityEvent event = deserialize_event(line);
                    
                    if (event.event_type == event_type) {
                        events.push_back(event);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to query events by type: {}", e.what());
    }
    
    return events;
}

std::vector<SecurityEvent> AuditStorage::get_events_by_severity(SecurityLevel severity) {
    std::vector<SecurityEvent> events;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                std::ifstream file(entry.path());
                std::string line;
                
                while (std::getline(file, line)) {
                    if (config_.encrypt_storage) {
                        line = decrypt_data(line);
                    }
                    
                    SecurityEvent event = deserialize_event(line);
                    
                    if (event.severity == severity) {
                        events.push_back(event);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to query events by severity: {}", e.what());
    }
    
    return events;
}

void AuditStorage::rotate_log_files() {
    close_current_file();
    
    if (!open_new_file()) {
        LOG_ERROR("Failed to open new log file during rotation");
        return;
    }
    
    stats_.files_rotated.fetch_add(1, std::memory_order_relaxed);
    LOG_INFO("Rotated audit log file to {}", current_filename_);
}

void AuditStorage::cleanup_old_files() {
    auto cutoff_time = std::chrono::system_clock::now() - config_.retention_period;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                auto file_time = std::filesystem::last_write_time(entry);
                auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    file_time - std::filesystem::file_time_type::clock::now() + 
                    std::chrono::system_clock::now());
                
                if (system_time < cutoff_time) {
                    std::filesystem::remove(entry);
                    LOG_INFO("Removed old audit log file: {}", entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to cleanup old files: {}", e.what());
    }
}

void AuditStorage::backup_current_logs() {
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                auto backup_path = std::filesystem::path(config_.backup_path) / entry.path().filename();
                std::filesystem::copy_file(entry, backup_path, 
                                         std::filesystem::copy_options::overwrite_existing);
            }
        }
        LOG_INFO("Backed up audit logs to {}", config_.backup_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to backup logs: {}", e.what());
    }
}

bool AuditStorage::verify_storage_integrity() {
    stats_.integrity_checks.fetch_add(1, std::memory_order_relaxed);
    
    try {
        // Verify file checksums, detect corruption, etc.
        // This is a simplified implementation
        
        for (const auto& entry : std::filesystem::directory_iterator(config_.storage_path)) {
            if (entry.is_regular_file()) {
                std::ifstream file(entry.path(), std::ios::binary);
                if (!file.good()) {
                    stats_.integrity_failures.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Storage integrity verification failed: {}", e.what());
        stats_.integrity_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

std::string AuditStorage::generate_filename() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "audit_" << std::put_time(std::gmtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
    
    return (std::filesystem::path(config_.storage_path) / ss.str()).string();
}

bool AuditStorage::open_new_file() {
    current_filename_ = generate_filename();
    current_file_.open(current_filename_, std::ios::app);
    
    if (current_file_.is_open()) {
        current_file_size_.store(0, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

void AuditStorage::close_current_file() {
    if (current_file_.is_open()) {
        current_file_.close();
    }
}

std::string AuditStorage::encrypt_data(const std::string& data) {
    // Simplified AES encryption - in production use proper key management
    // This is just a placeholder implementation
    return data;  // TODO: Implement proper encryption
}

std::string AuditStorage::decrypt_data(const std::string& encrypted_data) {
    // Simplified AES decryption - in production use proper key management
    // This is just a placeholder implementation
    return encrypted_data;  // TODO: Implement proper decryption
}

std::string AuditStorage::serialize_event(const SecurityEvent& event) {
    Json::Value json;
    json["event_id"] = static_cast<Json::UInt64>(event.event_id);
    json["event_type"] = static_cast<int>(event.event_type);
    json["severity"] = static_cast<int>(event.severity);
    json["timestamp"] = static_cast<Json::UInt64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            event.timestamp.time_since_epoch()).count());
    json["user_id"] = event.user_id;
    json["tenant_id"] = event.tenant_id;
    json["source_ip"] = event.source_ip;
    json["user_agent"] = event.user_agent;
    json["resource"] = event.resource;
    json["action"] = event.action;
    json["details"] = event.details;
    json["hash"] = event.hash;
    json["previous_hash"] = event.previous_hash;
    json["sequence_number"] = static_cast<Json::UInt64>(event.sequence_number);
    
    // Add metadata
    for (const auto& [key, value] : event.metadata) {
        json["metadata"][key] = value;
    }
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";  // Compact format
    return Json::writeString(builder, json);
}

SecurityEvent AuditStorage::deserialize_event(const std::string& data) {
    Json::Value json;
    Json::Reader reader;
    
    if (!reader.parse(data, json)) {
        throw std::runtime_error("Failed to parse JSON event data");
    }
    
    SecurityEvent event;
    event.event_id = json["event_id"].asUInt64();
    event.event_type = static_cast<SecurityEventType>(json["event_type"].asInt());
    event.severity = static_cast<SecurityLevel>(json["severity"].asInt());
    event.timestamp = std::chrono::system_clock::from_time_t(
        json["timestamp"].asUInt64() / 1000);
    event.user_id = json["user_id"].asString();
    event.tenant_id = json["tenant_id"].asString();
    event.source_ip = json["source_ip"].asString();
    event.user_agent = json["user_agent"].asString();
    event.resource = json["resource"].asString();
    event.action = json["action"].asString();
    event.details = json["details"].asString();
    event.hash = json["hash"].asString();
    event.previous_hash = json["previous_hash"].asString();
    event.sequence_number = json["sequence_number"].asUInt64();
    
    // Load metadata
    if (json.isMember("metadata")) {
        for (const auto& key : json["metadata"].getMemberNames()) {
            event.metadata[key] = json["metadata"][key].asString();
        }
    }
    
    return event;
}

// AuditLogger Implementation
AuditLogger::AuditLogger(const Config& config) : config_(config) {
    tamper_engine_ = std::make_unique<TamperEvidenceEngine>(config_.tamper_config);
    storage_ = std::make_unique<AuditStorage>(config_.storage_config);
    
    if (config_.async_logging) {
        // Start worker threads
        for (size_t i = 0; i < config_.worker_threads; ++i) {
            worker_threads_.emplace_back(&AuditLogger::worker_thread_func, this);
        }
    }
    
    LOG_INFO("AuditLogger initialized with {} worker threads", config_.worker_threads);
}

AuditLogger::~AuditLogger() {
    stop();
}

void AuditLogger::log_event(const SecurityEvent& event) {
    if (!should_log_event(event)) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SecurityEvent processed_event = event;
    process_event(processed_event);
    
    if (config_.async_logging) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        
        if (event_buffer_.size() >= config_.buffer_size) {
            stats_.buffer_overflows.fetch_add(1, std::memory_order_relaxed);
            stats_.events_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        
        event_buffer_.push(processed_event);
        buffer_condition_.notify_one();
    } else {
        storage_->store_event(processed_event);
    }
    
    stats_.events_logged.fetch_add(1, std::memory_order_relaxed);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    stats_.avg_log_time_ns.store(duration.count(), std::memory_order_relaxed);
}

void AuditLogger::log_authentication_success(const std::string& user_id,
                                           const std::string& source_ip) {
    SecurityEvent event = create_base_event(SecurityEventType::AUTHENTICATION_SUCCESS, 
                                          SecurityLevel::LOW);
    event.user_id = user_id;
    event.source_ip = source_ip;
    event.action = "login";
    event.details = "User authentication successful";
    
    log_event(event);
}

void AuditLogger::log_authentication_failure(const std::string& user_id,
                                           const std::string& source_ip,
                                           const std::string& reason) {
    SecurityEvent event = create_base_event(SecurityEventType::AUTHENTICATION_FAILURE, 
                                          SecurityLevel::MEDIUM);
    event.user_id = user_id;
    event.source_ip = source_ip;
    event.action = "login_failed";
    event.details = "Authentication failed: " + reason;
    
    log_event(event);
}

void AuditLogger::log_authorization_failure(const std::string& user_id,
                                          const std::string& resource,
                                          const std::string& action) {
    SecurityEvent event = create_base_event(SecurityEventType::AUTHORIZATION_FAILURE, 
                                          SecurityLevel::HIGH);
    event.user_id = user_id;
    event.resource = resource;
    event.action = action;
    event.details = "Access denied to resource: " + resource;
    
    log_event(event);
}

void AuditLogger::log_rate_limit_exceeded(const std::string& tenant_id,
                                        const std::string& source_ip) {
    SecurityEvent event = create_base_event(SecurityEventType::RATE_LIMIT_EXCEEDED, 
                                          SecurityLevel::MEDIUM);
    event.tenant_id = tenant_id;
    event.source_ip = source_ip;
    event.action = "rate_limit";
    event.details = "Rate limit exceeded for tenant: " + tenant_id;
    
    log_event(event);
}

void AuditLogger::log_suspicious_request(const std::string& details,
                                       const std::string& source_ip) {
    SecurityEvent event = create_base_event(SecurityEventType::SUSPICIOUS_REQUEST, 
                                          SecurityLevel::HIGH);
    event.source_ip = source_ip;
    event.action = "suspicious_activity";
    event.details = details;
    
    log_event(event);
}

void AuditLogger::log_injection_attempt(SecurityEventType type,
                                      const std::string& payload,
                                      const std::string& source_ip) {
    SecurityEvent event = create_base_event(type, SecurityLevel::CRITICAL);
    event.source_ip = source_ip;
    event.action = "injection_attempt";
    event.details = "Injection attempt detected";
    event.metadata["payload"] = payload.substr(0, 1000);  // Truncate for storage
    
    log_event(event);
}

void AuditLogger::log_token_event(SecurityEventType type,
                                const std::string& user_id,
                                const std::string& token_id) {
    SecurityLevel severity = (type == SecurityEventType::INVALID_TOKEN || 
                             type == SecurityEventType::TOKEN_EXPIRED) ? 
                             SecurityLevel::MEDIUM : SecurityLevel::LOW;
    
    SecurityEvent event = create_base_event(type, severity);
    event.user_id = user_id;
    event.action = "token_validation";
    event.metadata["token_id"] = token_id;
    
    log_event(event);
}

void AuditLogger::log_data_access(const std::string& user_id,
                                const std::string& resource,
                                const std::string& action) {
    SecurityEvent event = create_base_event(SecurityEventType::DATA_ACCESS, 
                                          SecurityLevel::LOW);
    event.user_id = user_id;
    event.resource = resource;
    event.action = action;
    event.details = "Data access: " + action + " on " + resource;
    
    log_event(event);
}

void AuditLogger::log_configuration_change(const std::string& user_id,
                                         const std::string& component,
                                         const std::string& changes) {
    SecurityEvent event = create_base_event(SecurityEventType::CONFIGURATION_CHANGE, 
                                          SecurityLevel::HIGH);
    event.user_id = user_id;
    event.resource = component;
    event.action = "config_change";
    event.details = "Configuration changed: " + changes;
    
    log_event(event);
}

void AuditLogger::log_system_error(const std::string& component,
                                 const std::string& error_details) {
    SecurityEvent event = create_base_event(SecurityEventType::SYSTEM_ERROR, 
                                          SecurityLevel::MEDIUM);
    event.resource = component;
    event.action = "system_error";
    event.details = error_details;
    
    log_event(event);
}

void AuditLogger::log_custom_event(const std::string& event_name,
                                 const std::unordered_map<std::string, std::string>& metadata) {
    SecurityEvent event = create_base_event(SecurityEventType::CUSTOM_EVENT, 
                                          SecurityLevel::LOW);
    event.action = event_name;
    event.metadata = metadata;
    
    log_event(event);
}

void AuditLogger::log_events_batch(const std::vector<SecurityEvent>& events) {
    std::vector<SecurityEvent> processed_events;
    processed_events.reserve(events.size());
    
    for (const auto& event : events) {
        if (should_log_event(event)) {
            SecurityEvent processed_event = event;
            process_event(processed_event);
            processed_events.push_back(processed_event);
        }
    }
    
    if (!processed_events.empty()) {
        storage_->store_events_batch(processed_events);
        stats_.events_logged.fetch_add(processed_events.size(), std::memory_order_relaxed);
    }
}

void AuditLogger::flush() {
    if (config_.async_logging) {
        std::vector<SecurityEvent> events_to_flush;
        
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            while (!event_buffer_.empty()) {
                events_to_flush.push_back(event_buffer_.front());
                event_buffer_.pop();
            }
        }
        
        if (!events_to_flush.empty()) {
            storage_->store_events_batch(events_to_flush);
        }
    }
    
    stats_.flush_operations.fetch_add(1, std::memory_order_relaxed);
}

void AuditLogger::start() {
    // Already started in constructor if async logging is enabled
}

void AuditLogger::stop() {
    shutdown_.store(true, std::memory_order_release);
    
    if (config_.async_logging) {
        buffer_condition_.notify_all();
        
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Flush remaining events
        flush();
    }
}

std::vector<SecurityEvent> AuditLogger::query_events(
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end,
    SecurityEventType event_type) {
    
    if (event_type == SecurityEventType::CUSTOM_EVENT) {
        return storage_->get_events_by_time_range(start, end);
    } else {
        auto events = storage_->get_events_by_time_range(start, end);
        
        // Filter by event type
        events.erase(std::remove_if(events.begin(), events.end(),
                                   [event_type](const SecurityEvent& event) {
                                       return event.event_type != event_type;
                                   }), events.end());
        
        return events;
    }
}

bool AuditLogger::verify_log_integrity() {
    stats_.integrity_checks.fetch_add(1, std::memory_order_relaxed);
    
    // Get all events and verify chain integrity
    auto now = std::chrono::system_clock::now();
    auto start = now - std::chrono::hours(24);  // Last 24 hours
    
    auto events = storage_->get_events_by_time_range(start, now);
    
    // Sort by sequence number
    std::sort(events.begin(), events.end(),
              [](const SecurityEvent& a, const SecurityEvent& b) {
                  return a.sequence_number < b.sequence_number;
              });
    
    return tamper_engine_->validate_chain_integrity(events);
}

std::string AuditLogger::generate_integrity_report() {
    Json::Value report;
    report["timestamp"] = static_cast<Json::UInt64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    
    bool integrity_valid = verify_log_integrity();
    report["integrity_valid"] = integrity_valid;
    
    auto stats = get_stats();
    report["stats"]["events_logged"] = static_cast<Json::UInt64>(stats.events_logged.load());
    report["stats"]["events_dropped"] = static_cast<Json::UInt64>(stats.events_dropped.load());
    report["stats"]["integrity_checks"] = static_cast<Json::UInt64>(stats.integrity_checks.load());
    
    auto storage_stats = storage_->get_stats();
    report["storage_stats"]["events_stored"] = static_cast<Json::UInt64>(storage_stats.events_stored.load());
    report["storage_stats"]["storage_errors"] = static_cast<Json::UInt64>(storage_stats.storage_errors.load());
    report["storage_stats"]["integrity_failures"] = static_cast<Json::UInt64>(storage_stats.integrity_failures.load());
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, report);
}

void AuditLogger::update_config(const Config& config) {
    config_ = config;
    LOG_INFO("AuditLogger configuration updated");
}

void AuditLogger::set_min_log_level(SecurityLevel level) {
    config_.min_log_level = level;
}

void AuditLogger::worker_thread_func() {
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::vector<SecurityEvent> events_to_process;
        
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_condition_.wait(lock, [this] {
                return !event_buffer_.empty() || shutdown_.load(std::memory_order_acquire);
            });
            
            // Process events in batches
            size_t batch_size = std::min(event_buffer_.size(), static_cast<size_t>(1000));
            
            for (size_t i = 0; i < batch_size; ++i) {
                events_to_process.push_back(event_buffer_.front());
                event_buffer_.pop();
            }
        }
        
        if (!events_to_process.empty()) {
            storage_->store_events_batch(events_to_process);
        }
    }
}

SecurityEvent AuditLogger::create_base_event(SecurityEventType type, SecurityLevel severity) {
    SecurityEvent event;
    event.event_id = event_id_counter_.fetch_add(1, std::memory_order_relaxed);
    event.event_type = type;
    event.severity = severity;
    event.timestamp = std::chrono::system_clock::now();
    event.user_id = get_current_user_id();
    event.tenant_id = get_current_tenant_id();
    event.source_ip = get_source_ip();
    event.user_agent = get_user_agent();
    
    return event;
}

void AuditLogger::process_event(SecurityEvent& event) {
    // Set sequence number
    event.sequence_number = tamper_engine_->sequence_counter_.fetch_add(1, std::memory_order_relaxed);
    
    // Compute hash
    event.hash = tamper_engine_->compute_event_hash(event);
    
    // Set previous hash (simplified - in production you'd maintain the chain properly)
    if (event.sequence_number > 0) {
        event.previous_hash = "previous_hash_placeholder";  // TODO: Implement proper chain
    }
}

bool AuditLogger::should_log_event(const SecurityEvent& event) {
    return static_cast<int>(event.severity) >= static_cast<int>(config_.min_log_level);
}

std::string AuditLogger::get_current_user_id() {
    // TODO: Extract from current context/thread-local storage
    return "system";
}

std::string AuditLogger::get_current_tenant_id() {
    // TODO: Extract from current context/thread-local storage
    return "default";
}

std::string AuditLogger::get_source_ip() {
    // TODO: Extract from current request context
    return "127.0.0.1";
}

std::string AuditLogger::get_user_agent() {
    // TODO: Extract from current request context
    return "ultra-cpp-system/1.0";
}

} // namespace ultra_cpp::security