#pragma once

#include "common/types.hpp"
#include <memory>
#include <atomic>
#include <vector>
#include <functional>

// DPDK availability check
#ifdef ULTRA_DPDK_ENABLED
    #include <rte_eal.h>
    #include <rte_ethdev.h>
    #include <rte_mbuf.h>
    #include <rte_lcore.h>
    #include <rte_ring.h>
    #include <rte_hash.h>
    #include <rte_mempool.h>
    #include <rte_cycles.h>
    #include <rte_launch.h>
    #include <rte_per_lcore.h>
#else
    // Mock DPDK types when not available
    struct rte_mbuf { char data[2048]; };
    struct rte_mempool { int dummy; };
    #define SOCKET_ID_ANY 0
    #define RTE_MBUF_DEFAULT_DATAROOM 2048
    #define ETH_RSS_IP 0x1
    #define ETH_RSS_TCP 0x2
    #define ETH_RSS_UDP 0x4
    #define ETH_LINK_UP 1
    #define ETH_LINK_FULL_DUPLEX 1
    #define ETH_LINK_AUTONEG 1
    #define RTE_PROC_INVALID -1
#endif

namespace ultra::network {

// DPDK configuration constants
constexpr u16 DPDK_RX_RING_SIZE = 2048;
constexpr u16 DPDK_TX_RING_SIZE = 2048;
constexpr u16 DPDK_NUM_MBUFS = 16383;
constexpr u16 DPDK_MBUF_CACHE_SIZE = 512;
constexpr u16 DPDK_BURST_SIZE = 64;
constexpr u16 DPDK_MAX_PORTS = 16;

// Packet processing statistics
struct PacketStats {
    aligned_atomic<u64> packets_received{0};
    aligned_atomic<u64> packets_transmitted{0};
    aligned_atomic<u64> packets_dropped{0};
    aligned_atomic<u64> bytes_received{0};
    aligned_atomic<u64> bytes_transmitted{0};
    aligned_atomic<u64> processing_cycles{0};
    aligned_atomic<u64> queue_full_drops{0};
    aligned_atomic<u64> invalid_packets{0};
};

// DPDK packet processor interface
class PacketProcessor {
public:
    virtual ~PacketProcessor() = default;
    
    // Process a batch of packets
    virtual u16 process_packets(struct rte_mbuf** packets, u16 num_packets, u16 port_id) = 0;
    
    // Initialize processor for specific lcore
    virtual bool initialize(u32 lcore_id) { return true; }
    
    // Cleanup processor resources
    virtual void cleanup() {}
    
    // Get processor statistics
    virtual PacketStats get_stats() const = 0;
};

// High-performance packet buffer management
class PacketBufferPool {
public:
    struct Config {
        u32 num_mbufs = DPDK_NUM_MBUFS;
        u32 mbuf_cache_size = DPDK_MBUF_CACHE_SIZE;
        u32 mbuf_data_room_size = RTE_MBUF_DEFAULT_DATAROOM;
        u32 socket_id = SOCKET_ID_ANY;
        std::string pool_name = "packet_pool";
    };
    
    explicit PacketBufferPool(const Config& config);
    ~PacketBufferPool();
    
    // Buffer allocation and deallocation
    struct rte_mbuf* allocate_packet() noexcept;
    void free_packet(struct rte_mbuf* packet) noexcept;
    
    // Bulk operations for better performance
    u16 allocate_packets(struct rte_mbuf** packets, u16 count) noexcept;
    void free_packets(struct rte_mbuf** packets, u16 count) noexcept;
    
    // Pool information
    u32 get_available_count() const noexcept;
    u32 get_total_count() const noexcept;
    bool is_valid() const noexcept { return mempool_ != nullptr; }
    
    struct rte_mempool* get_mempool() const noexcept { return mempool_; }
    
private:
    struct rte_mempool* mempool_;
    Config config_;
};

// DPDK port configuration and management
class DpdkPort {
public:
    struct PortConfig {
        u16 port_id;
        u16 rx_queues = 1;
        u16 tx_queues = 1;
        u16 rx_ring_size = DPDK_RX_RING_SIZE;
        u16 tx_ring_size = DPDK_TX_RING_SIZE;
        bool enable_rss = true;
        bool enable_checksum_offload = true;
        bool enable_tso = true;
        u32 mtu = 1500;
        
        // RSS configuration
        u64 rss_hash_functions = ETH_RSS_IP | ETH_RSS_TCP | ETH_RSS_UDP;
        u8 rss_key[40]; // RSS key for hash distribution
    };
    
    explicit DpdkPort(const PortConfig& config, PacketBufferPool& buffer_pool);
    ~DpdkPort();
    
    // Port lifecycle management
    bool initialize() noexcept;
    bool start() noexcept;
    void stop() noexcept;
    bool is_running() const noexcept { return running_; }
    
    // Packet I/O operations
    u16 receive_packets(struct rte_mbuf** packets, u16 max_packets, u16 queue_id = 0) noexcept;
    u16 transmit_packets(struct rte_mbuf** packets, u16 num_packets, u16 queue_id = 0) noexcept;
    
    // Port statistics and information
    struct PortStats {
        u64 rx_packets;
        u64 tx_packets;
        u64 rx_bytes;
        u64 tx_bytes;
        u64 rx_errors;
        u64 tx_errors;
        u64 rx_dropped;
        u64 tx_dropped;
    };
    
    PortStats get_port_stats() const noexcept;
    void reset_port_stats() noexcept;
    
    // Port configuration
    bool set_mtu(u32 mtu) noexcept;
    bool enable_promiscuous_mode() noexcept;
    bool disable_promiscuous_mode() noexcept;
    
    // Link status
    struct LinkStatus {
        bool link_up;
        u32 link_speed; // Mbps
        bool full_duplex;
        bool autoneg;
    };
    
    LinkStatus get_link_status() const noexcept;
    
    u16 get_port_id() const noexcept { return config_.port_id; }
    
private:
    PortConfig config_;
    PacketBufferPool& buffer_pool_;
    bool initialized_;
    bool running_;
    
    bool configure_port() noexcept;
    bool setup_rx_queues() noexcept;
    bool setup_tx_queues() noexcept;
    void generate_rss_key() noexcept;
};

// Multi-queue packet processing with worker threads
class DpdkWorker {
public:
    struct WorkerConfig {
        u32 lcore_id;
        u16 port_id;
        u16 rx_queue_id;
        u16 tx_queue_id;
        std::shared_ptr<PacketProcessor> processor;
        u32 burst_size = DPDK_BURST_SIZE;
        bool enable_prefetch = true;
    };
    
    explicit DpdkWorker(const WorkerConfig& config, DpdkPort& port);
    ~DpdkWorker();
    
    // Worker lifecycle
    bool start() noexcept;
    void stop() noexcept;
    bool is_running() const noexcept { return running_; }
    
    // Worker statistics
    PacketStats get_stats() const noexcept { return stats_; }
    void reset_stats() noexcept;
    
    u32 get_lcore_id() const noexcept { return config_.lcore_id; }
    
private:
    WorkerConfig config_;
    DpdkPort& port_;
    std::atomic<bool> running_{false};
    PacketStats stats_;
    
    // Worker main loop (runs on dedicated lcore)
    static int worker_main_loop(void* arg);
    void process_packets_loop() noexcept;
    
    // Performance optimizations
    void prefetch_packets(struct rte_mbuf** packets, u16 count) noexcept;
    void update_stats(u16 rx_count, u16 tx_count, u64 cycles) noexcept;
};

// Main DPDK network engine
class DpdkNetworkEngine {
public:
    struct EngineConfig {
        std::vector<u16> port_ids;
        u32 master_lcore = 0;
        std::vector<u32> worker_lcores;
        PacketBufferPool::Config buffer_pool_config;
        DpdkPort::PortConfig default_port_config;
        bool enable_interrupt_mode = false;
        u32 stats_update_interval_ms = 1000;
    };
    
    explicit DpdkNetworkEngine(const EngineConfig& config);
    ~DpdkNetworkEngine();
    
    // Engine lifecycle
    bool initialize(int argc, char** argv) noexcept;
    bool start() noexcept;
    void stop() noexcept;
    bool is_running() const noexcept { return running_; }
    
    // Port and processor management
    bool add_port(u16 port_id, const DpdkPort::PortConfig& config = {}) noexcept;
    void remove_port(u16 port_id) noexcept;
    
    void set_packet_processor(std::shared_ptr<PacketProcessor> processor) noexcept;
    
    // Worker management
    bool assign_worker_to_port(u32 lcore_id, u16 port_id, u16 queue_id = 0) noexcept;
    void remove_worker(u32 lcore_id) noexcept;
    
    // Statistics and monitoring
    struct EngineStats {
        u64 total_packets_processed;
        u64 total_bytes_processed;
        u64 total_processing_cycles;
        double packets_per_second;
        double bytes_per_second;
        double cpu_utilization;
        u32 active_workers;
        u32 active_ports;
    };
    
    EngineStats get_engine_stats() const noexcept;
    std::vector<PacketStats> get_worker_stats() const noexcept;
    std::vector<DpdkPort::PortStats> get_port_stats() const noexcept;
    
    // Configuration and tuning
    bool set_interrupt_mode(bool enable) noexcept;
    void set_stats_update_interval(u32 interval_ms) noexcept;
    
    // Utility functions
    static bool is_dpdk_available() noexcept;
    static std::vector<u16> get_available_ports() noexcept;
    static u32 get_socket_id(u32 lcore_id) noexcept;
    
private:
    EngineConfig config_;
    bool initialized_;
    bool running_;
    
    // DPDK resources
    std::unique_ptr<PacketBufferPool> buffer_pool_;
    std::unordered_map<u16, std::unique_ptr<DpdkPort>> ports_;
    std::unordered_map<u32, std::unique_ptr<DpdkWorker>> workers_;
    
    // Packet processor
    std::shared_ptr<PacketProcessor> packet_processor_;
    
    // Statistics collection
    std::thread stats_thread_;
    std::atomic<bool> stats_running_{false};
    mutable EngineStats cached_stats_;
    mutable std::mutex stats_mutex_;
    
    // Helper functions
    bool initialize_eal(int argc, char** argv) noexcept;
    bool check_port_capabilities(u16 port_id) noexcept;
    void stats_collection_loop() noexcept;
    void update_engine_stats() noexcept;
    
    // Cleanup functions
    void cleanup_workers() noexcept;
    void cleanup_ports() noexcept;
    void cleanup_eal() noexcept;
};

// Specialized packet processors

// HTTP packet processor for web traffic
class HttpPacketProcessor : public PacketProcessor {
public:
    struct HttpConfig {
        bool enable_http2 = true;
        bool enable_compression = true;
        u32 max_header_size = 8192;
        u32 max_body_size = 1024 * 1024; // 1MB
        std::function<void(const std::string&, const std::string&)> request_handler;
    };
    
    explicit HttpPacketProcessor(const HttpConfig& config);
    
    u16 process_packets(struct rte_mbuf** packets, u16 num_packets, u16 port_id) override;
    bool initialize(u32 lcore_id) override;
    PacketStats get_stats() const override { return stats_; }
    
private:
    HttpConfig config_;
    PacketStats stats_;
    
    bool process_http_packet(struct rte_mbuf* packet);
    bool parse_http_request(const char* data, size_t length);
};

// Load balancing packet processor
class LoadBalancingProcessor : public PacketProcessor {
public:
    struct Backend {
        u32 ip_address;
        u16 port;
        u32 weight;
        bool active;
    };
    
    explicit LoadBalancingProcessor(const std::vector<Backend>& backends);
    
    u16 process_packets(struct rte_mbuf** packets, u16 num_packets, u16 port_id) override;
    bool initialize(u32 lcore_id) override;
    PacketStats get_stats() const override { return stats_; }
    
    void add_backend(const Backend& backend);
    void remove_backend(u32 ip_address, u16 port);
    void update_backend_weight(u32 ip_address, u16 port, u32 weight);
    
private:
    std::vector<Backend> backends_;
    std::atomic<size_t> next_backend_{0};
    PacketStats stats_;
    
    Backend* select_backend(const char* packet_data);
    bool rewrite_packet_destination(struct rte_mbuf* packet, const Backend& backend);
};

// Firewall packet processor
class FirewallProcessor : public PacketProcessor {
public:
    enum class Action {
        ALLOW,
        DROP,
        REJECT
    };
    
    struct Rule {
        u32 src_ip;
        u32 src_mask;
        u32 dst_ip;
        u32 dst_mask;
        u16 src_port_min;
        u16 src_port_max;
        u16 dst_port_min;
        u16 dst_port_max;
        u8 protocol; // TCP=6, UDP=17, etc.
        Action action;
        u32 priority;
    };
    
    explicit FirewallProcessor(const std::vector<Rule>& rules);
    
    u16 process_packets(struct rte_mbuf** packets, u16 num_packets, u16 port_id) override;
    bool initialize(u32 lcore_id) override;
    PacketStats get_stats() const override { return stats_; }
    
    void add_rule(const Rule& rule);
    void remove_rule(u32 rule_id);
    void clear_rules();
    
private:
    std::vector<Rule> rules_;
    PacketStats stats_;
    
    Action evaluate_packet(const char* packet_data, size_t length);
    bool match_rule(const Rule& rule, u32 src_ip, u32 dst_ip, 
                   u16 src_port, u16 dst_port, u8 protocol);
};

} // namespace ultra::network