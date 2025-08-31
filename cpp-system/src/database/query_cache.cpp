#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include <functional>
#include <sstream>
#include <algorithm>
#include <thread>
#include <regex>

namespace ultra_cpp {
namespace database {

QueryCache::QueryCache(const Config& config) 
    : config_(config), shutdown_requested_(false) {
    
    // Start cleanup worker thread
    cleanup_thread_ = std::thread(&QueryCache::cleanup_worker, this);
    
    LOG_INFO("Query cache initialized with max {} entries, default TTL {}s", 
             config_.max_entries, config_.default_ttl_seconds);
}

QueryCache::~QueryCache() {
    shutdown_requested_.store(true);
    
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    LOG_INFO("Query cache destroyed");
}

std::optional<DatabaseConnector::QueryResult> QueryCache::get(const std::string& query, 
                                                             const std::vector<std::string>& params) {
    std::string cache_key = generate_cache_key(query, params);
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto it = cache_.find(cache_key);
    if (it == cache_.end()) {
        stats_.misses.fetch_add(1);
        return std::nullopt;
    }
    
    auto& entry = it->second;
    auto now = std::chrono::steady_clock::now();
    
    // Check if entry has expired
    if (now > entry.expires_at) {
        lock.unlock();
        
        // Remove expired entry
        std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
        auto expired_it = cache_.find(cache_key);
        if (expired_it != cache_.end() && now > expired_it->second.expires_at) {
            stats_.total_size_bytes.fetch_sub(expired_it->second.size_bytes);
            cache_.erase(expired_it);
            stats_.current_entries.fetch_sub(1);
        }
        
        stats_.misses.fetch_add(1);
        return std::nullopt;
    }
    
    // Update access statistics
    entry.access_count++;
    
    stats_.hits.fetch_add(1);
    
    LOG_DEBUG("Cache hit for query key: {}", cache_key.substr(0, 16) + "...");
    return entry.result;
}

void QueryCache::put(const std::string& query, const std::vector<std::string>& params,
                    const DatabaseConnector::QueryResult& result, uint32_t ttl_seconds) {
    
    if (!result.success) {
        // Don't cache failed queries
        return;
    }
    
    std::string cache_key = generate_cache_key(query, params);
    
    // Calculate entry size
    size_t entry_size = sizeof(CacheEntry) + cache_key.size() + result.error_message.size();
    for (const auto& row : result.rows) {
        for (const auto& cell : row) {
            entry_size += cell.size();
        }
    }
    
    // Check if result is too large to cache
    if (entry_size > config_.max_result_size_bytes) {
        LOG_DEBUG("Query result too large to cache: {} bytes", entry_size);
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    uint32_t effective_ttl = (ttl_seconds > 0) ? ttl_seconds : config_.default_ttl_seconds;
    
    CacheEntry entry;
    entry.query_hash = cache_key;
    entry.result = result;
    entry.created_at = now;
    entry.expires_at = now + std::chrono::seconds(effective_ttl);
    entry.access_count = 1;
    entry.size_bytes = entry_size;
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    // Check if we need to evict entries to make room
    while (cache_.size() >= config_.max_entries) {
        evict_lru_entries(1);
    }
    
    // Check if we need to evict based on size
    size_t total_size = stats_.total_size_bytes.load() + entry_size;
    while (total_size > config_.max_entries * config_.max_result_size_bytes / 4) { // Use 25% of theoretical max
        evict_lru_entries(1);
        total_size = stats_.total_size_bytes.load() + entry_size;
    }
    
    // Insert or update entry
    auto insert_result = cache_.emplace(cache_key, std::move(entry));
    if (insert_result.second) {
        // New entry
        stats_.current_entries.fetch_add(1);
        stats_.total_size_bytes.fetch_add(entry_size);
        LOG_DEBUG("Cached query result with key: {}, TTL: {}s", 
                 cache_key.substr(0, 16) + "...", effective_ttl);
    } else {
        // Update existing entry
        auto& existing_entry = insert_result.first->second;
        stats_.total_size_bytes.fetch_sub(existing_entry.size_bytes);
        stats_.total_size_bytes.fetch_add(entry_size);
        existing_entry = entry;
        LOG_DEBUG("Updated cached query result with key: {}", cache_key.substr(0, 16) + "...");
    }
}

void QueryCache::invalidate(const std::string& pattern) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    std::regex regex_pattern(pattern);
    size_t removed_count = 0;
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (std::regex_search(it->first, regex_pattern)) {
            stats_.total_size_bytes.fetch_sub(it->second.size_bytes);
            it = cache_.erase(it);
            removed_count++;
        } else {
            ++it;
        }
    }
    
    stats_.current_entries.fetch_sub(removed_count);
    stats_.invalidations.fetch_add(removed_count);
    
    LOG_INFO("Invalidated {} cache entries matching pattern: {}", removed_count, pattern);
}

void QueryCache::invalidate_all() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    size_t removed_count = cache_.size();
    cache_.clear();
    
    stats_.current_entries.store(0);
    stats_.total_size_bytes.store(0);
    stats_.invalidations.fetch_add(removed_count);
    
    LOG_INFO("Invalidated all {} cache entries", removed_count);
}

void QueryCache::cleanup_expired() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    size_t removed_count = 0;
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (now > it->second.expires_at) {
            stats_.total_size_bytes.fetch_sub(it->second.size_bytes);
            it = cache_.erase(it);
            removed_count++;
        } else {
            ++it;
        }
    }
    
    stats_.current_entries.fetch_sub(removed_count);
    
    if (removed_count > 0) {
        LOG_DEBUG("Cleaned up {} expired cache entries", removed_count);
    }
}

void QueryCache::set_ttl(const std::string& query, uint32_t ttl_seconds) {
    // This would require storing query->key mapping for efficient lookup
    // For now, this is a placeholder implementation
    LOG_DEBUG("TTL update requested for query (not implemented in this version)");
}

QueryCache::CacheStats QueryCache::get_stats() const noexcept {
    return stats_;
}

void QueryCache::reset_stats() noexcept {
    stats_.hits.store(0);
    stats_.misses.store(0);
    stats_.evictions.store(0);
    stats_.invalidations.store(0);
    // Don't reset current_entries and total_size_bytes as they reflect current state
}

std::string QueryCache::generate_cache_key(const std::string& query, const std::vector<std::string>& params) {
    std::hash<std::string> hasher;
    
    // Create a normalized query string
    std::string normalized_query = query;
    
    // Remove extra whitespace and normalize case for better cache hits
    std::regex whitespace_regex("\\s+");
    normalized_query = std::regex_replace(normalized_query, whitespace_regex, " ");
    
    // Convert to lowercase for case-insensitive matching
    std::transform(normalized_query.begin(), normalized_query.end(), normalized_query.begin(), ::tolower);
    
    // Combine query and parameters
    std::ostringstream key_stream;
    key_stream << normalized_query;
    
    for (const auto& param : params) {
        key_stream << "|" << param;
    }
    
    std::string combined = key_stream.str();
    
    // Generate hash
    size_t hash_value = hasher(combined);
    
    // Convert to hex string
    std::ostringstream hex_stream;
    hex_stream << std::hex << hash_value;
    
    return hex_stream.str();
}

void QueryCache::cleanup_worker() {
    LOG_INFO("Cache cleanup worker started");
    
    while (!shutdown_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(60)); // Cleanup every minute
        
        if (shutdown_requested_.load()) {
            break;
        }
        
        cleanup_expired();
    }
    
    LOG_INFO("Cache cleanup worker stopped");
}

void QueryCache::evict_lru_entries(size_t count) {
    if (cache_.empty()) {
        return;
    }
    
    // Find entries with lowest access count (simple LRU approximation)
    std::vector<std::pair<uint32_t, std::string>> access_counts;
    access_counts.reserve(cache_.size());
    
    for (const auto& [key, entry] : cache_) {
        access_counts.emplace_back(entry.access_count, key);
    }
    
    // Sort by access count (ascending)
    std::sort(access_counts.begin(), access_counts.end());
    
    // Remove the least accessed entries
    size_t removed = 0;
    for (const auto& [access_count, key] : access_counts) {
        if (removed >= count) {
            break;
        }
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            stats_.total_size_bytes.fetch_sub(it->second.size_bytes);
            cache_.erase(it);
            removed++;
        }
    }
    
    stats_.current_entries.fetch_sub(removed);
    stats_.evictions.fetch_add(removed);
    
    if (removed > 0) {
        LOG_DEBUG("Evicted {} LRU cache entries", removed);
    }
}

size_t QueryCache::calculate_entry_size(const CacheEntry& entry) {
    size_t size = sizeof(CacheEntry) + entry.query_hash.size() + entry.result.error_message.size();
    
    for (const auto& row : entry.result.rows) {
        for (const auto& cell : row) {
            size += cell.size();
        }
    }
    
    return size;
}

} // namespace database
} // namespace ultra_cpp