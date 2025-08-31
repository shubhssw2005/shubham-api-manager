#include "network/dpdk_network.hpp"
#ifdef ULTRA_LOGGER_AVAILABLE
    #include "common/logger.hpp"
#else
    #include "common/simple_logger.hpp"
#endif
#include <algorithm>
#include <cstring>

#ifdef ULTRA_DPDK_ENABLED
    #include <rte_ip.h>
    #include <rte_tcp.h>
    #include <rte_udp.h>
    #include <rte_ether.h>
#endif

namespace ultra::network {

// PacketBufferPool Implementation
PacketBufferPool::PacketBufferPool(const Config& config) : config_(config), mempool_(nullptr) {
#ifdef ULTRA_DPDK_ENABLED
    // Create DPDK memory pool for packet buffers
    mempool_ = rte_pktmbuf_pool_create(
        config_.pool_name.c_str(),
        config_.num_mbufs,
        config_.mbuf_cache_size,
        0, // private data size
        config_.mbuf_data_room_size,
        config_.socket_id
    );
    
    if (!mempool_) {
        ULTRA_LOG_ERROR("Failed to create packet buffer pool: {}", rte_strerror(rte_errno));
    } else {
        ULTRA_LOG_INFO("Created packet buffer pool '{}' with {} mbufs", 
                      config_.pool_name, config_.num_mbufs);
    }
#else
    ULTRA_LOG_WARN("DPDK not available - packet buffer pool disabled");
#endif
}

PacketBufferPool::~PacketBufferPool() {
    if (mempool_) {
        rte_mempool_free(mempool_);
    }
}

struct rte_mbuf* PacketBufferPool::allocate_packet() noexcept {
    if (!mempool_) {
        return nullptr;
    }
    
    return rte_pktmbuf_alloc(mempool_);
}

void PacketBufferPool::free_packet(struct rte_mbuf* packet) noexcept {
    if (packet) {
        rte_pktmbuf_free(packet);
    }
}

u16 PacketBufferPool::allocate_packets(struct rte_mbuf** packets, u16 count) noexcept {
    if (!mempool_ || !packets) {
        return 0;
    }
    
    return rte_pktmbuf_alloc_bulk(mempool_, packets, count) == 0 ? count : 0;
}

void PacketBufferPool::free_packets(struct rte_mbuf** packets, u16 count) noexcept {
    if (!packets) {
        return;
    }
    
    for (u16 i = 0; i < count; ++i) {
        if (packets[i]) {
            rte_pktmbuf_free(packets[i]);
        }
    }
}

u32 PacketBufferPool::get_available_count() const noexcept {
    if (!mempool_) {
        return 0;
    }
    
    return rte_mempool_avail_count(mempool_);
}

u32 PacketBufferPool::get_total_count() const noexcept {
    if (!mempool_) {
        return 0;
    }
    
    return rte_mempool_count(mempool_);
}

// DpdkPort Implementation
DpdkPort::DpdkPort(const PortConfig& config, PacketBufferPool& buffer_pool)
    : config_(config), buffer_pool_(buffer_pool), initialized_(false), running_(false) {
    
    // Generate RSS key if not provided
    generate_rss_key();
}

DpdkPort::~DpdkPort() {
    stop();
}

bool DpdkPort::initialize() noexcept {
    if (initialized_) {
        return true;
    }
    
    // Check if port is available
    if (!rte_eth_dev_is_valid_port(config_.port_id)) {
        ULTRA_LOG_ERROR("Port {} is not available", config_.port_id);
        return false;
    }
    
    // Configure the port
    if (!configure_port()) {
        return false;
    }
    
    // Setup RX queues
    if (!setup_rx_queues()) {
        return false;
    }
    
    // Setup TX queues
    if (!setup_tx_queues()) {
        return false;
    }
    
    initialized_ = true;
    ULTRA_LOG_INFO("Initialized DPDK port {}", config_.port_id);
    return true;
}

bool DpdkPort::start() noexcept {
    if (!initialized_) {
        ULTRA_LOG_ERROR("Port {} not initialized", config_.port_id);
        return false;
    }
    
    if (running_) {
        return true;
    }
    
    // Start the port
    int ret = rte_eth_dev_start(config_.port_id);
    if (ret < 0) {
        ULTRA_LOG_ERROR("Failed to start port {}: {}", config_.port_id, rte_strerror(-ret));
        return false;
    }
    
    // Enable promiscuous mode if needed
    ret = rte_eth_promiscuous_enable(config_.port_id);
    if (ret != 0) {
        ULTRA_LOG_WARN("Failed to enable promiscuous mode on port {}: {}", 
                      config_.port_id, rte_strerror(-ret));
    }
    
    running_ = true;
    ULTRA_LOG_INFO("Started DPDK port {}", config_.port_id);
    return true;
}

void DpdkPort::stop() noexcept {
    if (!running_) {
        return;
    }
    
    rte_eth_dev_stop(config_.port_id);
    running_ = false;
    
    ULTRA_LOG_INFO("Stopped DPDK port {}", config_.port_id);
}

u16 DpdkPort::receive_packets(struct rte_mbuf** packets, u16 max_packets, u16 queue_id) noexcept {
    if (!running_ || !packets) {
        return 0;
    }
    
    return rte_eth_rx_burst(config_.port_id, queue_id, packets, max_packets);
}

u16 DpdkPort::transmit_packets(struct rte_mbuf** packets, u16 num_packets, u16 queue_id) noexcept {
    if (!running_ || !packets) {
        return 0;
    }
    
    return rte_eth_tx_burst(config_.port_id, queue_id, packets, num_packets);
}

bool DpdkPort::configure_port() noexcept {
    struct rte_eth_conf port_conf = {};
    
    // Configure RX mode
    port_conf.rxmode.mq_mode = config_.enable_rss ? ETH_MQ_RX_RSS : ETH_MQ_RX_NONE;
    port_conf.rxmode.mtu = config_.mtu;
    
    if (config_.enable_checksum_offload) {
        port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_CHECKSUM;
    }
    
    // Configure TX mode
    if (config_.enable_checksum_offload) {
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_IPV4_CKSUM | DEV_TX_OFFLOAD_TCP_CKSUM | DEV_TX_OFFLOAD_UDP_CKSUM;
    }
    
    if (config_.enable_tso) {
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_TCP_TSO;
    }
    
    // Configure RSS
    if (config_.enable_rss) {
        port_conf.rx_adv_conf.rss_conf.rss_key = config_.rss_key;
        port_conf.rx_adv_conf.rss_conf.rss_key_len = sizeof(config_.rss_key);
        port_conf.rx_adv_conf.rss_conf.rss_hf = config_.rss_hash_functions;
    }
    
    // Configure the port
    int ret = rte_eth_dev_configure(config_.port_id, config_.rx_queues, config_.tx_queues, &port_conf);
    if (ret < 0) {
        ULTRA_LOG_ERROR("Failed to configure port {}: {}", config_.port_id, rte_strerror(-ret));
        return false;
    }
    
    return true;
}

bool DpdkPort::setup_rx_queues() noexcept {
    for (u16 queue_id = 0; queue_id < config_.rx_queues; ++queue_id) {
        int ret = rte_eth_rx_queue_setup(
            config_.port_id,
            queue_id,
            config_.rx_ring_size,
            rte_eth_dev_socket_id(config_.port_id),
            nullptr, // Use default RX configuration
            buffer_pool_.get_mempool()
        );
        
        if (ret < 0) {
            ULTRA_LOG_ERROR("Failed to setup RX queue {} on port {}: {}", 
                          queue_id, config_.port_id, rte_strerror(-ret));
            return false;
        }
    }
    
    return true;
}

bool DpdkPort::setup_tx_queues() noexcept {
    for (u16 queue_id = 0; queue_id < config_.tx_queues; ++queue_id) {
        int ret = rte_eth_tx_queue_setup(
            config_.port_id,
            queue_id,
            config_.tx_ring_size,
            rte_eth_dev_socket_id(config_.port_id),
            nullptr // Use default TX configuration
        );
        
        if (ret < 0) {
            ULTRA_LOG_ERROR("Failed to setup TX queue {} on port {}: {}", 
                          queue_id, config_.port_id, rte_strerror(-ret));
            return false;
        }
    }
    
    return true;
}

void DpdkPort::generate_rss_key() noexcept {
    // Generate a simple RSS key - in production, use a proper random key
    for (size_t i = 0; i < sizeof(config_.rss_key); ++i) {
        config_.rss_key[i] = static_cast<u8>(i * 7 + 13); // Simple pattern
    }
}

DpdkPort::PortStats DpdkPort::get_port_stats() const noexcept {
    struct rte_eth_stats eth_stats;
    
    if (rte_eth_stats_get(config_.port_id, &eth_stats) != 0) {
        return {}; // Return empty stats on error
    }
    
    PortStats stats;
    stats.rx_packets = eth_stats.ipackets;
    stats.tx_packets = eth_stats.opackets;
    stats.rx_bytes = eth_stats.ibytes;
    stats.tx_bytes = eth_stats.obytes;
    stats.rx_errors = eth_stats.ierrors;
    stats.tx_errors = eth_stats.oerrors;
    stats.rx_dropped = eth_stats.imissed;
    stats.tx_dropped = 0; // Not directly available in rte_eth_stats
    
    return stats;
}

void DpdkPort::reset_port_stats() noexcept {
    rte_eth_stats_reset(config_.port_id);
}

DpdkPort::LinkStatus DpdkPort::get_link_status() const noexcept {
    struct rte_eth_link link;
    rte_eth_link_get_nowait(config_.port_id, &link);
    
    LinkStatus status;
    status.link_up = link.link_status == ETH_LINK_UP;
    status.link_speed = link.link_speed;
    status.full_duplex = link.link_duplex == ETH_LINK_FULL_DUPLEX;
    status.autoneg = link.link_autoneg == ETH_LINK_AUTONEG;
    
    return status;
}

// DpdkWorker Implementation
DpdkWorker::DpdkWorker(const WorkerConfig& config, DpdkPort& port)
    : config_(config), port_(port) {
}

DpdkWorker::~DpdkWorker() {
    stop();
}

bool DpdkWorker::start() noexcept {
    if (running_.exchange(true)) {
        return true; // Already running
    }
    
    // Launch worker on specified lcore
    int ret = rte_eal_remote_launch(worker_main_loop, this, config_.lcore_id);
    if (ret != 0) {
        ULTRA_LOG_ERROR("Failed to launch worker on lcore {}: {}", config_.lcore_id, rte_strerror(-ret));
        running_ = false;
        return false;
    }
    
    ULTRA_LOG_INFO("Started DPDK worker on lcore {}", config_.lcore_id);
    return true;
}

void DpdkWorker::stop() noexcept {
    if (!running_.exchange(false)) {
        return; // Not running
    }
    
    // Wait for worker to finish
    rte_eal_wait_lcore(config_.lcore_id);
    
    ULTRA_LOG_INFO("Stopped DPDK worker on lcore {}", config_.lcore_id);
}

int DpdkWorker::worker_main_loop(void* arg) {
    auto* worker = static_cast<DpdkWorker*>(arg);
    worker->process_packets_loop();
    return 0;
}

void DpdkWorker::process_packets_loop() noexcept {
    struct rte_mbuf* rx_packets[DPDK_BURST_SIZE];
    struct rte_mbuf* tx_packets[DPDK_BURST_SIZE];
    
    // Initialize processor if available
    if (config_.processor && !config_.processor->initialize(config_.lcore_id)) {
        ULTRA_LOG_ERROR("Failed to initialize packet processor on lcore {}", config_.lcore_id);
        return;
    }
    
    ULTRA_LOG_INFO("Worker {} starting packet processing loop", config_.lcore_id);
    
    while (running_.load(std::memory_order_acquire)) {
        u64 start_cycles = rte_rdtsc();
        
        // Receive packets
        u16 num_rx = port_.receive_packets(rx_packets, config_.burst_size, config_.rx_queue_id);
        
        if (num_rx > 0) {
            // Prefetch packets for better cache performance
            if (config_.enable_prefetch) {
                prefetch_packets(rx_packets, num_rx);
            }
            
            // Process packets
            u16 num_tx = 0;
            if (config_.processor) {
                num_tx = config_.processor->process_packets(rx_packets, num_rx, port_.get_port_id());
                
                // Copy processed packets to TX array
                for (u16 i = 0; i < num_tx && i < num_rx; ++i) {
                    tx_packets[i] = rx_packets[i];
                }
            } else {
                // No processor - just forward packets
                num_tx = num_rx;
                std::memcpy(tx_packets, rx_packets, num_rx * sizeof(struct rte_mbuf*));
            }
            
            // Transmit packets
            if (num_tx > 0) {
                u16 sent = port_.transmit_packets(tx_packets, num_tx, config_.tx_queue_id);
                
                // Free unsent packets
                for (u16 i = sent; i < num_tx; ++i) {
                    rte_pktmbuf_free(tx_packets[i]);
                }
            }
            
            // Free remaining RX packets that weren't processed
            for (u16 i = num_tx; i < num_rx; ++i) {
                rte_pktmbuf_free(rx_packets[i]);
            }
            
            u64 end_cycles = rte_rdtsc();
            update_stats(num_rx, num_tx, end_cycles - start_cycles);
        }
    }
    
    // Cleanup processor
    if (config_.processor) {
        config_.processor->cleanup();
    }
    
    ULTRA_LOG_INFO("Worker {} finished packet processing loop", config_.lcore_id);
}

void DpdkWorker::prefetch_packets(struct rte_mbuf** packets, u16 count) noexcept {
    for (u16 i = 0; i < count; ++i) {
        rte_prefetch0(rte_pktmbuf_mtod(packets[i], void*));
    }
}

void DpdkWorker::update_stats(u16 rx_count, u16 tx_count, u64 cycles) noexcept {
    stats_.packets_received.fetch_add(rx_count, std::memory_order_relaxed);
    stats_.packets_transmitted.fetch_add(tx_count, std::memory_order_relaxed);
    stats_.processing_cycles.fetch_add(cycles, std::memory_order_relaxed);
    
    // Update byte counts (simplified - assumes average packet size)
    constexpr u32 AVG_PACKET_SIZE = 1024;
    stats_.bytes_received.fetch_add(rx_count * AVG_PACKET_SIZE, std::memory_order_relaxed);
    stats_.bytes_transmitted.fetch_add(tx_count * AVG_PACKET_SIZE, std::memory_order_relaxed);
    
    if (rx_count > tx_count) {
        stats_.packets_dropped.fetch_add(rx_count - tx_count, std::memory_order_relaxed);
    }
}

void DpdkWorker::reset_stats() noexcept {
    stats_.packets_received.store(0);
    stats_.packets_transmitted.store(0);
    stats_.packets_dropped.store(0);
    stats_.bytes_received.store(0);
    stats_.bytes_transmitted.store(0);
    stats_.processing_cycles.store(0);
    stats_.queue_full_drops.store(0);
    stats_.invalid_packets.store(0);
}

// DpdkNetworkEngine Implementation
DpdkNetworkEngine::DpdkNetworkEngine(const EngineConfig& config)
    : config_(config), initialized_(false), running_(false) {
}

DpdkNetworkEngine::~DpdkNetworkEngine() {
    stop();
    cleanup_eal();
}

bool DpdkNetworkEngine::initialize(int argc, char** argv) noexcept {
    if (initialized_) {
        return true;
    }
    
    // Initialize DPDK EAL
    if (!initialize_eal(argc, argv)) {
        return false;
    }
    
    // Create packet buffer pool
    buffer_pool_ = std::make_unique<PacketBufferPool>(config_.buffer_pool_config);
    if (!buffer_pool_->is_valid()) {
        ULTRA_LOG_ERROR("Failed to create packet buffer pool");
        return false;
    }
    
    // Initialize ports
    for (u16 port_id : config_.port_ids) {
        if (!add_port(port_id, config_.default_port_config)) {
            ULTRA_LOG_ERROR("Failed to initialize port {}", port_id);
            return false;
        }
    }
    
    initialized_ = true;
    ULTRA_LOG_INFO("DPDK network engine initialized successfully");
    return true;
}

bool DpdkNetworkEngine::start() noexcept {
    if (!initialized_) {
        ULTRA_LOG_ERROR("Engine not initialized");
        return false;
    }
    
    if (running_) {
        return true;
    }
    
    // Start all ports
    for (auto& [port_id, port] : ports_) {
        if (!port->start()) {
            ULTRA_LOG_ERROR("Failed to start port {}", port_id);
            return false;
        }
    }
    
    // Start statistics collection
    stats_running_ = true;
    stats_thread_ = std::thread(&DpdkNetworkEngine::stats_collection_loop, this);
    
    running_ = true;
    ULTRA_LOG_INFO("DPDK network engine started");
    return true;
}

void DpdkNetworkEngine::stop() noexcept {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Stop statistics collection
    if (stats_running_) {
        stats_running_ = false;
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }
    }
    
    // Stop workers
    cleanup_workers();
    
    // Stop ports
    for (auto& [port_id, port] : ports_) {
        port->stop();
    }
    
    ULTRA_LOG_INFO("DPDK network engine stopped");
}

bool DpdkNetworkEngine::add_port(u16 port_id, const DpdkPort::PortConfig& config) noexcept {
    if (!buffer_pool_) {
        return false;
    }
    
    DpdkPort::PortConfig port_config = config;
    port_config.port_id = port_id;
    
    auto port = std::make_unique<DpdkPort>(port_config, *buffer_pool_);
    if (!port->initialize()) {
        return false;
    }
    
    ports_[port_id] = std::move(port);
    return true;
}

bool DpdkNetworkEngine::initialize_eal(int argc, char** argv) noexcept {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        ULTRA_LOG_ERROR("Failed to initialize DPDK EAL: {}", rte_strerror(rte_errno));
        return false;
    }
    
    ULTRA_LOG_INFO("DPDK EAL initialized with {} arguments processed", ret);
    return true;
}

void DpdkNetworkEngine::stats_collection_loop() noexcept {
    while (stats_running_) {
        update_engine_stats();
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.stats_update_interval_ms));
    }
}

void DpdkNetworkEngine::update_engine_stats() noexcept {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    cached_stats_ = {};
    cached_stats_.active_workers = workers_.size();
    cached_stats_.active_ports = ports_.size();
    
    // Aggregate worker statistics
    for (const auto& [lcore_id, worker] : workers_) {
        auto worker_stats = worker->get_stats();
        cached_stats_.total_packets_processed += worker_stats.packets_received.load();
        cached_stats_.total_bytes_processed += worker_stats.bytes_received.load();
        cached_stats_.total_processing_cycles += worker_stats.processing_cycles.load();
    }
    
    // Calculate rates (simplified)
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration<double>(now - last_update).count();
    
    if (elapsed_seconds > 0) {
        static u64 last_packets = 0;
        static u64 last_bytes = 0;
        
        cached_stats_.packets_per_second = 
            (cached_stats_.total_packets_processed - last_packets) / elapsed_seconds;
        cached_stats_.bytes_per_second = 
            (cached_stats_.total_bytes_processed - last_bytes) / elapsed_seconds;
        
        last_packets = cached_stats_.total_packets_processed;
        last_bytes = cached_stats_.total_bytes_processed;
        last_update = now;
    }
}

DpdkNetworkEngine::EngineStats DpdkNetworkEngine::get_engine_stats() const noexcept {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return cached_stats_;
}

void DpdkNetworkEngine::cleanup_workers() noexcept {
    for (auto& [lcore_id, worker] : workers_) {
        worker->stop();
    }
    workers_.clear();
}

void DpdkNetworkEngine::cleanup_ports() noexcept {
    ports_.clear();
}

void DpdkNetworkEngine::cleanup_eal() noexcept {
    rte_eal_cleanup();
}

// Static utility functions
bool DpdkNetworkEngine::is_dpdk_available() noexcept {
    return rte_eal_process_type() != RTE_PROC_INVALID;
}

std::vector<u16> DpdkNetworkEngine::get_available_ports() noexcept {
    std::vector<u16> ports;
    
    u16 port_id;
    RTE_ETH_FOREACH_DEV(port_id) {
        ports.push_back(port_id);
    }
    
    return ports;
}

u32 DpdkNetworkEngine::get_socket_id(u32 lcore_id) noexcept {
    return rte_lcore_to_socket_id(lcore_id);
}

} // namespace ultra::network