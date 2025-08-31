#pragma once

#include "common/types.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <memory>
#include <atomic>
#include <vector>

// Platform-specific includes
#ifdef __linux__
    #include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
    #include <sys/event.h>
    #include <sys/time.h>
#endif

namespace ultra::network {

// TCP connection with SO_REUSEPORT optimization
class TcpConnection {
public:
    enum class State {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        CLOSING,
        CLOSED,
        ERROR_STATE
    };
    
    struct Config {
        bool enable_reuseport = true;
        bool enable_nodelay = true;
        bool enable_keepalive = true;
        u32 keepalive_idle = 7200;     // seconds
        u32 keepalive_interval = 75;   // seconds
        u32 keepalive_probes = 9;      // count
        u32 send_buffer_size = 65536;  // bytes
        u32 recv_buffer_size = 65536;  // bytes
        u32 connect_timeout_ms = 5000; // milliseconds
        bool enable_fast_open = true;
        bool enable_defer_accept = true;
    };
    
    TcpConnection();
    explicit TcpConnection(int socket_fd);
    ~TcpConnection();
    
    // Non-copyable, movable
    TcpConnection(const TcpConnection&) = delete;
    TcpConnection& operator=(const TcpConnection&) = delete;
    TcpConnection(TcpConnection&& other) noexcept;
    TcpConnection& operator=(TcpConnection&& other) noexcept;
    
    // Connection management
    bool bind_and_listen(const std::string& address, u16 port, 
                        const Config& config) noexcept;
    
    bool bind_and_listen(const std::string& address, u16 port) noexcept {
        return bind_and_listen(address, port, Config{});
    }
    
    bool connect(const std::string& address, u16 port,
                const Config& config) noexcept;
    
    bool connect(const std::string& address, u16 port) noexcept {
        return connect(address, port, Config{});
    }
    
    std::unique_ptr<TcpConnection> accept() noexcept;
    
    void close() noexcept;
    
    // I/O operations
    ssize_t send(const void* data, size_t length, int flags = 0) noexcept;
    ssize_t receive(void* buffer, size_t length, int flags = 0) noexcept;
    
    // Zero-copy I/O using sendfile and splice
    ssize_t sendfile(int file_fd, off_t offset, size_t count) noexcept;
    ssize_t splice_from_pipe(int pipe_fd, size_t count) noexcept;
    ssize_t splice_to_pipe(int pipe_fd, size_t count) noexcept;
    
    // Vectored I/O for scatter-gather operations
    ssize_t sendv(const struct iovec* iov, int iovcnt) noexcept;
    ssize_t receivev(const struct iovec* iov, int iovcnt) noexcept;
    
    // Non-blocking I/O status
    bool would_block() const noexcept;
    bool is_connected() const noexcept;
    bool has_error() const noexcept;
    
    // Socket information
    int get_socket_fd() const noexcept { return socket_fd_; }
    State get_state() const noexcept { return state_; }
    
    struct sockaddr_in get_local_address() const noexcept;
    struct sockaddr_in get_remote_address() const noexcept;
    
    // Socket options and tuning
    bool set_socket_option(int level, int optname, const void* optval, socklen_t optlen) noexcept;
    bool get_socket_option(int level, int optname, void* optval, socklen_t* optlen) noexcept;
    
    // Performance optimizations
    bool enable_tcp_fastopen() noexcept;
    bool enable_tcp_nodelay() noexcept;
    bool enable_tcp_keepalive(u32 idle, u32 interval, u32 probes) noexcept;
    bool set_send_buffer_size(u32 size) noexcept;
    bool set_recv_buffer_size(u32 size) noexcept;
    
    // Connection statistics
    struct Stats {
        u64 bytes_sent = 0;
        u64 bytes_received = 0;
        u64 packets_sent = 0;
        u64 packets_received = 0;
        u32 retransmissions = 0;
        u32 rtt_microseconds = 0;
        timestamp_t connected_at{};
        timestamp_t last_activity{};
    };
    
    Stats get_stats() const noexcept;
    
private:
    int socket_fd_;
    State state_;
    Config config_;
    mutable Stats stats_;
    
    bool configure_socket(const Config& config) noexcept;
    bool set_non_blocking() noexcept;
    void update_stats() const noexcept;
};

// High-performance TCP connection pool with SO_REUSEPORT
class TcpConnectionPool {
public:
    struct PoolConfig {
        u32 max_connections_per_worker = 10000;
        u32 worker_threads = std::thread::hardware_concurrency();
        u32 connection_timeout_ms = 30000;
        u32 keepalive_interval_ms = 5000;
        bool enable_connection_reuse = true;
        bool enable_pipelining = false;
        TcpConnection::Config connection_config;
    };
    
    explicit TcpConnectionPool(const PoolConfig& config);
    ~TcpConnectionPool();
    
    // Pool management
    bool start() noexcept;
    void stop() noexcept;
    
    // Connection acquisition and release
    std::shared_ptr<TcpConnection> acquire_connection(const std::string& host, u16 port) noexcept;
    void release_connection(std::shared_ptr<TcpConnection> conn) noexcept;
    
    // Server-side connection handling
    bool bind_and_listen(const std::string& address, u16 port) noexcept;
    void set_connection_handler(std::function<void(std::shared_ptr<TcpConnection>)> handler);
    
    // Pool statistics
    struct PoolStats {
        aligned_atomic<u64> active_connections{0};
        aligned_atomic<u64> total_connections_created{0};
        aligned_atomic<u64> connections_reused{0};
        aligned_atomic<u64> connection_timeouts{0};
        aligned_atomic<u64> connection_errors{0};
    };
    
    PoolStats get_stats() const noexcept;
    
private:
    struct WorkerThread {
        std::thread thread;
        int epoll_fd = -1;
        std::vector<std::shared_ptr<TcpConnection>> connections;
        std::atomic<bool> running{false};
        
        // Per-worker connection cache for reuse
        std::unordered_map<std::string, std::vector<std::shared_ptr<TcpConnection>>> connection_cache;
        std::mutex cache_mutex;
    };
    
    PoolConfig config_;
    std::vector<std::unique_ptr<WorkerThread>> workers_;
    std::atomic<u32> next_worker_{0};
    std::function<void(std::shared_ptr<TcpConnection>)> connection_handler_;
    
    // Server socket with SO_REUSEPORT for load balancing
    std::vector<int> server_sockets_;
    
    PoolStats stats_;
    
    void worker_loop(WorkerThread& worker, u32 worker_id) noexcept;
    void handle_epoll_events(WorkerThread& worker, struct epoll_event* events, int num_events) noexcept;
    void cleanup_idle_connections(WorkerThread& worker) noexcept;
    
    u32 get_next_worker() noexcept;
    std::string make_connection_key(const std::string& host, u16 port) const noexcept;
};

// TCP connection manager with advanced features
class TcpConnectionManager {
public:
    struct ManagerConfig {
        u32 max_global_connections = 100000;
        u32 connection_limit_per_ip = 1000;
        u32 rate_limit_per_ip = 10000; // connections per second
        u32 blacklist_threshold = 100;  // failed connections before blacklist
        u32 blacklist_duration_ms = 300000; // 5 minutes
        bool enable_connection_tracking = true;
        bool enable_ddos_protection = true;
    };
    
    explicit TcpConnectionManager(const ManagerConfig& config);
    ~TcpConnectionManager();
    
    // Connection validation and filtering
    bool should_accept_connection(const struct sockaddr_in& client_addr) noexcept;
    void register_connection(const struct sockaddr_in& client_addr, 
                           std::shared_ptr<TcpConnection> conn) noexcept;
    void unregister_connection(const struct sockaddr_in& client_addr,
                             std::shared_ptr<TcpConnection> conn) noexcept;
    
    // Rate limiting and DDoS protection
    bool check_rate_limit(const struct sockaddr_in& client_addr) noexcept;
    void report_failed_connection(const struct sockaddr_in& client_addr) noexcept;
    void blacklist_ip(u32 ip_address, u32 duration_ms) noexcept;
    void whitelist_ip(u32 ip_address) noexcept;
    
    // Connection monitoring
    struct ConnectionInfo {
        u32 ip_address;
        u16 port;
        timestamp_t connected_at;
        u64 bytes_transferred;
        u32 connection_count;
    };
    
    std::vector<ConnectionInfo> get_active_connections() const noexcept;
    
    // Statistics and monitoring
    struct ManagerStats {
        aligned_atomic<u64> total_connections{0};
        aligned_atomic<u64> active_connections{0};
        aligned_atomic<u64> rejected_connections{0};
        aligned_atomic<u64> rate_limited_connections{0};
        aligned_atomic<u64> blacklisted_ips{0};
        aligned_atomic<u64> ddos_attacks_detected{0};
    };
    
    ManagerStats get_stats() const noexcept;
    
private:
    struct IpConnectionInfo {
        aligned_atomic<u32> active_connections{0};
        aligned_atomic<u32> total_connections{0};
        aligned_atomic<u32> failed_connections{0};
        aligned_atomic<timestamp_t> last_connection_time{};
        aligned_atomic<timestamp_t> blacklisted_until{};
        
        // Rate limiting using token bucket
        aligned_atomic<u32> tokens{0};
        aligned_atomic<timestamp_t> last_token_refill{};
    };
    
    ManagerConfig config_;
    ManagerStats stats_;
    
    // IP-based connection tracking
    std::unordered_map<u32, std::unique_ptr<IpConnectionInfo>> ip_connections_;
    mutable std::shared_mutex ip_connections_mutex_;
    
    // Active connection tracking
    std::unordered_map<u32, std::vector<std::weak_ptr<TcpConnection>>> active_connections_;
    mutable std::shared_mutex active_connections_mutex_;
    
    // Blacklist management
    std::unordered_set<u32> blacklisted_ips_;
    std::unordered_set<u32> whitelisted_ips_;
    mutable std::shared_mutex blacklist_mutex_;
    
    // Cleanup thread for expired entries
    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_running_{false};
    
    void cleanup_loop() noexcept;
    void refill_tokens(IpConnectionInfo& info, timestamp_t current_time) noexcept;
    bool is_blacklisted(u32 ip_address, timestamp_t current_time) noexcept;
    IpConnectionInfo* get_or_create_ip_info(u32 ip_address) noexcept;
};

} // namespace ultra::network