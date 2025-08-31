#include "network/http_protocol.hpp"
#include "network/tcp_connection.hpp"
#include "network/load_balancer.hpp"
#include "network/dpdk_network.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace ultra::network;

// Demo HTTP packet processor
class DemoHttpProcessor : public PacketProcessor {
public:
    DemoHttpProcessor() = default;
    
    u16 process_packets(struct rte_mbuf** packets, u16 num_packets, u16 port_id) override {
        for (u16 i = 0; i < num_packets; ++i) {
            process_http_packet(packets[i]);
        }
        
        stats_.packets_received.fetch_add(num_packets);
        return num_packets; // Forward all packets
    }
    
    PacketStats get_stats() const override {
        return stats_;
    }
    
private:
    PacketStats stats_;
    Http11Parser parser_;
    
    bool process_http_packet(struct rte_mbuf* packet) {
        char* data = rte_pktmbuf_mtod(packet, char*);
        size_t length = rte_pktmbuf_data_len(packet);
        
        // Try to parse as HTTP request
        HttpRequest request;
        Http11Parser::ParseState state;
        
        auto result = parser_.parse_request(data, length, request, state);
        if (result == Http11Parser::ParseResult::COMPLETE) {
            std::cout << "Parsed HTTP request: " << static_cast<int>(request.method) 
                     << " " << request.path << std::endl;
            return true;
        }
        
        return false;
    }
};

void demo_http_parsing() {
    std::cout << "\n=== HTTP Protocol Parsing Demo ===" << std::endl;
    
    Http11Parser parser;
    ProtocolDetector detector;
    
    // Sample HTTP/1.1 request
    const char* http_request = 
        "GET /api/v1/users?limit=10 HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "User-Agent: UltraClient/1.0\r\n"
        "Accept: application/json\r\n"
        "Content-Length: 0\r\n"
        "\r\n";
    
    size_t request_length = std::strlen(http_request);
    
    // Detect protocol
    auto protocol = detector.detect_protocol(http_request, request_length);
    std::cout << "Detected protocol: " << static_cast<int>(protocol) << std::endl;
    
    // Parse HTTP request
    HttpRequest request;
    Http11Parser::ParseState state;
    
    auto result = parser.parse_request(http_request, request_length, request, state);
    if (result == Http11Parser::ParseResult::COMPLETE) {
        std::cout << "Successfully parsed HTTP request:" << std::endl;
        std::cout << "  Method: " << static_cast<int>(request.method) << std::endl;
        std::cout << "  Path: " << request.path << std::endl;
        std::cout << "  Query: " << request.query_string << std::endl;
        std::cout << "  Headers: " << request.headers.size() << std::endl;
        
        for (const auto& header : request.headers) {
            std::cout << "    " << header.name << ": " << header.value << std::endl;
        }
    } else {
        std::cout << "Failed to parse HTTP request" << std::endl;
    }
}

void demo_tcp_connection() {
    std::cout << "\n=== TCP Connection Demo ===" << std::endl;
    
    TcpConnection::Config config;
    config.enable_reuseport = true;
    config.enable_nodelay = true;
    config.enable_keepalive = true;
    
    // Create server connection
    TcpConnection server;
    if (server.bind_and_listen("127.0.0.1", 8080, config)) {
        std::cout << "Server listening on 127.0.0.1:8080" << std::endl;
        
        // Accept connections in a separate thread
        std::thread server_thread([&server]() {
            for (int i = 0; i < 3; ++i) {
                auto client = server.accept();
                if (client) {
                    std::cout << "Accepted client connection" << std::endl;
                    
                    // Echo server
                    char buffer[1024];
                    ssize_t bytes_read = client->receive(buffer, sizeof(buffer));
                    if (bytes_read > 0) {
                        client->send(buffer, bytes_read);
                        std::cout << "Echoed " << bytes_read << " bytes" << std::endl;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        
        // Create client connections
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        for (int i = 0; i < 3; ++i) {
            TcpConnection client;
            if (client.connect("127.0.0.1", 8080, config)) {
                std::cout << "Client connected to server" << std::endl;
                
                std::string message = "Hello from client " + std::to_string(i);
                client.send(message.c_str(), message.length());
                
                char response[1024];
                ssize_t bytes_received = client.receive(response, sizeof(response));
                if (bytes_received > 0) {
                    std::cout << "Received response: " << std::string(response, bytes_received) << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        server_thread.join();
    } else {
        std::cout << "Failed to start server" << std::endl;
    }
}

void demo_load_balancer() {
    std::cout << "\n=== Load Balancer Demo ===" << std::endl;
    
    LoadBalancer::Config config;
    config.algorithm = LoadBalancingAlgorithm::CONSISTENT_HASH;
    config.enable_health_checks = true;
    config.health_check_config.type = HealthChecker::HealthCheckType::TCP_CONNECT;
    config.health_check_config.interval_ms = 5000;
    
    LoadBalancer lb(config);
    
    // Add backend servers
    lb.add_backend("192.168.1.10", 8080, 100);
    lb.add_backend("192.168.1.11", 8080, 150);
    lb.add_backend("192.168.1.12", 8080, 200);
    
    std::cout << "Added 3 backend servers" << std::endl;
    
    // Simulate load balancing requests
    for (int i = 0; i < 10; ++i) {
        std::string client_key = "client_" + std::to_string(i % 3);
        auto backend = lb.select_backend(client_key);
        
        if (backend) {
            std::cout << "Request " << i << " (client: " << client_key 
                     << ") -> " << backend->host << ":" << backend->port 
                     << " (weight: " << backend->weight << ")" << std::endl;
        } else {
            std::cout << "No backend available for request " << i << std::endl;
        }
    }
    
    // Show load balancer statistics
    auto stats = lb.get_stats();
    std::cout << "\nLoad Balancer Statistics:" << std::endl;
    std::cout << "  Total requests: " << stats.total_requests.load() << std::endl;
    std::cout << "  Successful requests: " << stats.successful_requests.load() << std::endl;
    std::cout << "  Failed requests: " << stats.failed_requests.load() << std::endl;
}

void demo_dpdk_network() {
    std::cout << "\n=== DPDK Network Demo ===" << std::endl;
    
    // Check if DPDK is available
    if (!DpdkNetworkEngine::is_dpdk_available()) {
        std::cout << "DPDK not available - skipping DPDK demo" << std::endl;
        return;
    }
    
    // Get available ports
    auto ports = DpdkNetworkEngine::get_available_ports();
    std::cout << "Available DPDK ports: ";
    for (u16 port : ports) {
        std::cout << port << " ";
    }
    std::cout << std::endl;
    
    if (ports.empty()) {
        std::cout << "No DPDK ports available - skipping DPDK demo" << std::endl;
        return;
    }
    
    // Create DPDK network engine
    DpdkNetworkEngine::EngineConfig config;
    config.port_ids = {ports[0]}; // Use first available port
    config.worker_lcores = {1, 2}; // Use lcores 1 and 2
    
    DpdkNetworkEngine engine(config);
    
    // Initialize with dummy arguments
    const char* argv[] = {"network_demo", "-l", "0-3", "-n", "4"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    
    if (engine.initialize(argc, const_cast<char**>(argv))) {
        std::cout << "DPDK engine initialized successfully" << std::endl;
        
        // Set packet processor
        auto processor = std::make_shared<DemoHttpProcessor>();
        engine.set_packet_processor(processor);
        
        // Start engine
        if (engine.start()) {
            std::cout << "DPDK engine started - processing packets for 5 seconds" << std::endl;
            
            // Run for 5 seconds
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            // Show statistics
            auto stats = engine.get_engine_stats();
            std::cout << "Engine Statistics:" << std::endl;
            std::cout << "  Packets processed: " << stats.total_packets_processed << std::endl;
            std::cout << "  Bytes processed: " << stats.total_bytes_processed << std::endl;
            std::cout << "  Packets/sec: " << stats.packets_per_second << std::endl;
            std::cout << "  Active workers: " << stats.active_workers << std::endl;
            
            engine.stop();
        } else {
            std::cout << "Failed to start DPDK engine" << std::endl;
        }
    } else {
        std::cout << "Failed to initialize DPDK engine" << std::endl;
    }
}

int main() {
    std::cout << "Ultra Low-Latency Network Protocol Optimization Demo" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    try {
        demo_http_parsing();
        demo_tcp_connection();
        demo_load_balancer();
        demo_dpdk_network();
        
        std::cout << "\n=== Demo completed successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}