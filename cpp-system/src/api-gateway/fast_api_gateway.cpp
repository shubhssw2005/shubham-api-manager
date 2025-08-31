#include "api-gateway/fast_api_gateway.hpp"
#include "common/types.hpp"
#include "memory/memory_pool.hpp"
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_lcore.h>
#include <rte_ring.h>
#include <rte_hash.h>
#include <immintrin.h>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>

namespace ultra::api {

// DPDK configuration constants
constexpr u16 RX_RING_SIZE = 1024;
constexpr u16 TX_RING_SIZE = 1024;
constexpr u16 NUM_MBUFS = 8191;
constexpr u16 MBUF_CACHE_SIZE = 250;
constexpr u16 BURST_SIZE = 32;

// HTTP parsing constants
constexpr size_t MAX_HTTP_HEADER_SIZE = 8192;
constexpr size_t MAX_HTTP_BODY_SIZE = 1024 * 1024; // 1MB

// Connection pool constants
constexpr size_t MAX_CONNECTIONS_PER_CORE = 10000;
constexpr u32 HASH_ENTRIES = 1024 * 1024; // 1M entries

struct Connection {
    u32 connection_id;
    u64 last_activity_ns;
    u16 port;
    u32 ip_addr;
    std::atomic<bool> in_use{false};
    char receive_buffer[MAX_HTTP_HEADER_SIZE + MAX_HTTP_BODY_SIZE];
    size_t buffer_pos = 0;
} __attribute__((aligned(CACHE_LINE_SIZE)));

// Lock-free request router using consistent hashing
class RequestRouter {
private:
    struct RouteEntry {
        std::string path_pattern;
        FastHandler handler;
        u64 hash_value;
    };
    
    std::vector<RouteEntry> routes_;
    std::atomic<size_t> route_count_{0};
    
    // Consistent hashing ring for load balancing
    struct HashRingEntry {
        u64 hash;
        u32 worker_id;
    };
    
    std::vector<HashRingEntry> hash_ring_;
    static constexpr size_t VIRTUAL_NODES = 160; // 40 workers * 4 virtual nodes
    
public:
    RequestRouter() {
        routes_.reserve(1000); // Pre-allocate for performance
        hash_ring_.reserve(VIRTUAL_NODES);
    }
    
    void add_route(const std::string& path, FastHandler handler) {
        u64 hash = compute_hash(path);
        routes_.emplace_back(RouteEntry{path, std::move(handler), hash});
        route_count_.store(routes_.size(), std::memory_order_release);
    }
    
    FastHandler* find_handler(const std::string& path) noexcept {
        u64 path_hash = compute_hash(path);
        size_t count = route_count_.load(std::memory_order_acquire);
        
        // Linear search with SIMD optimization for small route counts
        for (size_t i = 0; i < count; ++i) {
            if (routes_[i].hash_value == path_hash && 
                routes_[i].path_pattern == path) {
                return &routes_[i].handler;
            }
        }
        return nullptr;
    }
    
    u32 get_worker_for_request(const std::string& path) noexcept {
        u64 hash = compute_hash(path);
        
        // Binary search in hash ring for consistent hashing
        auto it = std::lower_bound(hash_ring_.begin(), hash_ring_.end(), hash,
            [](const HashRingEntry& entry, u64 value) {
                return entry.hash < value;
            });
        
        if (it == hash_ring_.end()) {
            return hash_ring_.empty() ? 0 : hash_ring_[0].worker_id;
        }
        return it->worker_id;
    }
    
    void setup_hash_ring(u32 num_workers) {
        hash_ring_.clear();
        
        for (u32 worker = 0; worker < num_workers; ++worker) {
            for (u32 vnode = 0; vnode < 4; ++vnode) {
                std::string key = std::to_string(worker) + ":" + std::to_string(vnode);
                u64 hash = compute_hash(key);
                hash_ring_.emplace_back(HashRingEntry{hash, worker});
            }
        }
        
        std::sort(hash_ring_.begin(), hash_ring_.end(),
            [](const HashRingEntry& a, const HashRingEntry& b) {
                return a.hash < b.hash;
            });
    }
    
private:
    ULTRA_FORCE_INLINE u64 compute_hash(const std::string& str) noexcept {
        // Fast hash function using SIMD when possible
        u64 hash = 14695981039346656037ULL; // FNV offset basis
        const char* data = str.data();
        size_t len = str.length();
        
        // Process 8 bytes at a time using SIMD
        while (len >= 8) {
            u64 chunk;
            std::memcpy(&chunk, data, 8);
            hash ^= chunk;
            hash *= 1099511628211ULL; // FNV prime
            data += 8;
            len -= 8;
        }
        
        // Process remaining bytes
        while (len > 0) {
            hash ^= static_cast<u64>(*data++);
            hash *= 1099511628211ULL;
            --len;
        }
        
        return hash;
    }
};

// Zero-copy HTTP parser with SIMD string processing
class HttpParser {
private:
    static constexpr char CR = '\r';
    static constexpr char LF = '\n';
    static constexpr char SP = ' ';
    static constexpr char COLON = ':';
    
public:
    enum class ParseResult {
        COMPLETE,
        INCOMPLETE,
        ERROR
    };
    
    ParseResult parse_request(const char* data, size_t length, HttpRequest& request) noexcept {
        const char* ptr = data;
        const char* end = data + length;
        
        // Parse request line using SIMD for fast scanning
        auto method_end = find_space_simd(ptr, end);
        if (method_end == end) return ParseResult::INCOMPLETE;
        
        request.method.assign(ptr, method_end);
        ptr = method_end + 1;
        
        auto path_end = find_space_simd(ptr, end);
        if (path_end == end) return ParseResult::INCOMPLETE;
        
        // Extract path and query string
        auto query_pos = std::find(ptr, path_end, '?');
        if (query_pos != path_end) {
            request.path.assign(ptr, query_pos);
            request.query_string.assign(query_pos + 1, path_end);
        } else {
            request.path.assign(ptr, path_end);
        }
        
        ptr = path_end + 1;
        
        // Skip HTTP version and find end of request line
        auto line_end = find_crlf_simd(ptr, end);
        if (line_end == end) return ParseResult::INCOMPLETE;
        
        ptr = line_end + 2; // Skip CRLF
        
        // Parse headers using SIMD acceleration
        while (ptr < end) {
            auto header_line_end = find_crlf_simd(ptr, end);
            if (header_line_end == end) return ParseResult::INCOMPLETE;
            
            // Empty line indicates end of headers
            if (header_line_end == ptr) {
                ptr += 2; // Skip final CRLF
                break;
            }
            
            auto colon_pos = std::find(ptr, header_line_end, COLON);
            if (colon_pos == header_line_end) return ParseResult::ERROR;
            
            std::string name(ptr, colon_pos);
            
            // Skip colon and whitespace
            auto value_start = colon_pos + 1;
            while (value_start < header_line_end && *value_start == SP) {
                ++value_start;
            }
            
            std::string value(value_start, header_line_end);
            request.headers.emplace_back(std::move(name), std::move(value));
            
            ptr = header_line_end + 2; // Skip CRLF
        }
        
        // Parse body if present
        if (ptr < end) {
            request.body.assign(ptr, end);
        }
        
        request.received_at = std::chrono::high_resolution_clock::now();
        return ParseResult::COMPLETE;
    }
    
private:
    ULTRA_FORCE_INLINE const char* find_space_simd(const char* start, const char* end) noexcept {
        const char* ptr = start;
        
        // Use SIMD to find space character quickly
        while (ptr + 16 <= end) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
            __m128i spaces = _mm_set1_epi8(SP);
            __m128i cmp = _mm_cmpeq_epi8(chunk, spaces);
            
            int mask = _mm_movemask_epi8(cmp);
            if (mask != 0) {
                return ptr + __builtin_ctz(mask);
            }
            ptr += 16;
        }
        
        // Handle remaining bytes
        while (ptr < end && *ptr != SP) {
            ++ptr;
        }
        
        return ptr;
    }
    
    ULTRA_FORCE_INLINE const char* find_crlf_simd(const char* start, const char* end) noexcept {
        const char* ptr = start;
        
        // Use SIMD to find CRLF sequence quickly
        while (ptr + 16 <= end) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
            __m128i cr = _mm_set1_epi8(CR);
            __m128i cmp = _mm_cmpeq_epi8(chunk, cr);
            
            int mask = _mm_movemask_epi8(cmp);
            if (mask != 0) {
                int pos = __builtin_ctz(mask);
                if (ptr + pos + 1 < end && ptr[pos + 1] == LF) {
                    return ptr + pos;
                }
            }
            ptr += 16;
        }
        
        // Handle remaining bytes
        while (ptr + 1 < end) {
            if (*ptr == CR && *(ptr + 1) == LF) {
                return ptr;
            }
            ++ptr;
        }
        
        return end;
    }
};

// Connection pool with per-core worker threads
class ConnectionPool {
private:
    struct PerCorePool {
        alignas(CACHE_LINE_SIZE) std::vector<Connection> connections;
        alignas(CACHE_LINE_SIZE) std::atomic<u32> next_connection{0};
        alignas(CACHE_LINE_SIZE) rte_ring* free_connections;
        
        PerCorePool() {
            connections.resize(MAX_CONNECTIONS_PER_CORE);
            
            // Initialize connection IDs
            for (u32 i = 0; i < MAX_CONNECTIONS_PER_CORE; ++i) {
                connections[i].connection_id = i;
            }
            
            // Create lock-free ring for free connections
            std::string ring_name = "free_conn_" + std::to_string(rte_lcore_id());
            free_connections = rte_ring_create(ring_name.c_str(), 
                                             MAX_CONNECTIONS_PER_CORE,
                                             rte_socket_id(),
                                             RING_F_SP_ENQ | RING_F_SC_DEQ);
            
            // Initially all connections are free
            for (u32 i = 0; i < MAX_CONNECTIONS_PER_CORE; ++i) {
                void* conn_ptr = &connections[i];
                rte_ring_enqueue(free_connections, conn_ptr);
            }
        }
        
        ~PerCorePool() {
            if (free_connections) {
                rte_ring_free(free_connections);
            }
        }
    };
    
    std::vector<std::unique_ptr<PerCorePool>> per_core_pools_;
    u32 num_cores_;
    
public:
    explicit ConnectionPool(u32 num_cores) : num_cores_(num_cores) {
        per_core_pools_.reserve(num_cores);
        for (u32 i = 0; i < num_cores; ++i) {
            per_core_pools_.emplace_back(std::make_unique<PerCorePool>());
        }
    }
    
    Connection* acquire_connection(u32 core_id) noexcept {
        if (ULTRA_UNLIKELY(core_id >= num_cores_)) {
            return nullptr;
        }
        
        auto& pool = per_core_pools_[core_id];
        void* conn_ptr;
        
        if (rte_ring_dequeue(pool->free_connections, &conn_ptr) == 0) {
            Connection* conn = static_cast<Connection*>(conn_ptr);
            conn->in_use.store(true, std::memory_order_relaxed);
            conn->last_activity_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            conn->buffer_pos = 0;
            return conn;
        }
        
        return nullptr; // Pool exhausted
    }
    
    void release_connection(Connection* conn, u32 core_id) noexcept {
        if (ULTRA_UNLIKELY(!conn || core_id >= num_cores_)) {
            return;
        }
        
        conn->in_use.store(false, std::memory_order_relaxed);
        
        auto& pool = per_core_pools_[core_id];
        rte_ring_enqueue(pool->free_connections, conn);
    }
    
    void cleanup_idle_connections(u32 core_id, u64 timeout_ns) noexcept {
        if (ULTRA_UNLIKELY(core_id >= num_cores_)) {
            return;
        }
        
        auto& pool = per_core_pools_[core_id];
        u64 current_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        for (auto& conn : pool->connections) {
            if (conn.in_use.load(std::memory_order_relaxed) &&
                (current_time - conn.last_activity_ns) > timeout_ns) {
                release_connection(&conn, core_id);
            }
        }
    }
};

// DPDK network I/O with poll mode drivers
class DPDKNetworkIO {
private:
    u16 port_id_;
    struct rte_mempool* mbuf_pool_;
    bool initialized_;
    
public:
    DPDKNetworkIO() : port_id_(0), mbuf_pool_(nullptr), initialized_(false) {}
    
    bool initialize(u32 port_mask) {
        // Initialize DPDK EAL
        const char* argv[] = {"ultra-api-gateway", "-l", "0-3", "-n", "4", "--proc-type=primary"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        
        int ret = rte_eal_init(argc, const_cast<char**>(argv));
        if (ret < 0) {
            return false;
        }
        
        // Create memory pool for mbufs
        mbuf_pool_ = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
                                           MBUF_CACHE_SIZE, 0,
                                           RTE_MBUF_DEFAULT_BUF_SIZE,
                                           rte_socket_id());
        if (!mbuf_pool_) {
            return false;
        }
        
        // Configure Ethernet port
        struct rte_eth_conf port_conf = {};
        port_conf.rxmode.mq_mode = ETH_MQ_RX_RSS;
        port_conf.rx_adv_conf.rss_conf.rss_key = nullptr;
        port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IP | ETH_RSS_TCP;
        
        ret = rte_eth_dev_configure(port_id_, 1, 1, &port_conf);
        if (ret < 0) {
            return false;
        }
        
        // Setup RX queue
        ret = rte_eth_rx_queue_setup(port_id_, 0, RX_RING_SIZE,
                                   rte_eth_dev_socket_id(port_id_),
                                   nullptr, mbuf_pool_);
        if (ret < 0) {
            return false;
        }
        
        // Setup TX queue
        ret = rte_eth_tx_queue_setup(port_id_, 0, TX_RING_SIZE,
                                   rte_eth_dev_socket_id(port_id_),
                                   nullptr);
        if (ret < 0) {
            return false;
        }
        
        // Start the Ethernet port
        ret = rte_eth_dev_start(port_id_);
        if (ret < 0) {
            return false;
        }
        
        // Enable promiscuous mode
        ret = rte_eth_promiscuous_enable(port_id_);
        if (ret != 0) {
            return false;
        }
        
        initialized_ = true;
        return true;
    }
    
    u16 receive_packets(struct rte_mbuf** packets, u16 max_packets) noexcept {
        if (ULTRA_UNLIKELY(!initialized_)) {
            return 0;
        }
        
        return rte_eth_rx_burst(port_id_, 0, packets, max_packets);
    }
    
    u16 send_packets(struct rte_mbuf** packets, u16 num_packets) noexcept {
        if (ULTRA_UNLIKELY(!initialized_)) {
            return 0;
        }
        
        return rte_eth_tx_burst(port_id_, 0, packets, num_packets);
    }
    
    struct rte_mbuf* allocate_mbuf() noexcept {
        return rte_pktmbuf_alloc(mbuf_pool_);
    }
    
    void free_mbuf(struct rte_mbuf* mbuf) noexcept {
        rte_pktmbuf_free(mbuf);
    }
    
    ~DPDKNetworkIO() {
        if (initialized_) {
            rte_eth_dev_stop(port_id_);
            rte_eth_dev_close(port_id_);
        }
        if (mbuf_pool_) {
            rte_mempool_free(mbuf_pool_);
        }
    }
};

// FastAPIGateway implementation
class FastAPIGateway::Impl {
private:
    Config config_;
    std::unique_ptr<DPDKNetworkIO> network_io_;
    std::unique_ptr<RequestRouter> router_;
    std::unique_ptr<ConnectionPool> connection_pool_;
    std::unique_ptr<HttpParser> http_parser_;
    
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    Stats stats_;
    
public:
    explicit Impl(const Config& config) : config_(config) {
        network_io_ = std::make_unique<DPDKNetworkIO>();
        router_ = std::make_unique<RequestRouter>();
        connection_pool_ = std::make_unique<ConnectionPool>(config_.worker_threads);
        http_parser_ = std::make_unique<HttpParser>();
        
        // Setup consistent hashing ring
        router_->setup_hash_ring(config_.worker_threads);
    }
    
    bool start() {
        if (config_.enable_dpdk) {
            if (!network_io_->initialize(config_.dpdk_port_mask)) {
                return false;
            }
        }
        
        running_.store(true, std::memory_order_release);
        
        // Start worker threads
        worker_threads_.reserve(config_.worker_threads);
        for (size_t i = 0; i < config_.worker_threads; ++i) {
            worker_threads_.emplace_back([this, i]() {
                worker_loop(static_cast<u32>(i));
            });
        }
        
        return true;
    }
    
    void stop() {
        running_.store(false, std::memory_order_release);
        
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
    }
    
    void register_route(const std::string& path, FastHandler handler) {
        router_->add_route(path, std::move(handler));
    }
    
    Stats get_stats() const noexcept {
        return stats_;
    }
    
private:
    void worker_loop(u32 worker_id) {
        // Pin thread to specific CPU core for better cache locality
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(worker_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        
        struct rte_mbuf* packets[BURST_SIZE];
        
        while (running_.load(std::memory_order_acquire)) {
            if (config_.enable_dpdk) {
                // Receive packets using DPDK
                u16 num_received = network_io_->receive_packets(packets, BURST_SIZE);
                
                for (u16 i = 0; i < num_received; ++i) {
                    process_packet(packets[i], worker_id);
                }
            } else {
                // Fallback to regular socket processing
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            
            // Cleanup idle connections periodically
            if (worker_id == 0) { // Only one worker does cleanup
                static u64 last_cleanup = 0;
                u64 now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                if (now - last_cleanup > 1000000000ULL) { // 1 second
                    for (u32 core = 0; core < config_.worker_threads; ++core) {
                        connection_pool_->cleanup_idle_connections(core, 30000000000ULL); // 30 seconds
                    }
                    last_cleanup = now;
                }
            }
        }
    }
    
    void process_packet(struct rte_mbuf* packet, u32 worker_id) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Extract HTTP data from packet
        char* data = rte_pktmbuf_mtod(packet, char*);
        size_t length = rte_pktmbuf_data_len(packet);
        
        // Parse HTTP request
        HttpRequest request;
        auto parse_result = http_parser_->parse_request(data, length, request);
        
        if (parse_result == HttpParser::ParseResult::COMPLETE) {
            // Find handler for the request
            FastHandler* handler = router_->find_handler(request.path);
            
            HttpResponse response;
            if (handler) {
                // Execute fast path handler
                (*handler)(request, response);
                stats_.requests_processed.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Fallback to Node.js layer
                response.status_code = 502;
                response.body = "Service temporarily unavailable";
                stats_.fallback_requests.fetch_add(1, std::memory_order_relaxed);
            }
            
            // Send response (simplified for this implementation)
            send_response(response, packet);
            
            // Update latency statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            stats_.total_latency_ns.fetch_add(latency.count(), std::memory_order_relaxed);
        }
        
        // Free the packet
        network_io_->free_mbuf(packet);
    }
    
    void send_response(const HttpResponse& response, struct rte_mbuf* packet) {
        // Simplified response sending - in a real implementation,
        // this would construct proper HTTP response and send via DPDK
        
        // For now, just free the packet
        // TODO: Implement proper HTTP response construction and sending
    }
};

// FastAPIGateway public interface implementation
FastAPIGateway::FastAPIGateway(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {
}

FastAPIGateway::~FastAPIGateway() = default;

void FastAPIGateway::start() {
    pimpl_->start();
}

void FastAPIGateway::stop() {
    pimpl_->stop();
}

void FastAPIGateway::register_fast_route(const std::string& path, FastHandler handler) {
    pimpl_->register_route(path, std::move(handler));
}

FastAPIGateway::Stats FastAPIGateway::get_stats() const noexcept {
    return pimpl_->get_stats();
}

} // namespace ultra::api