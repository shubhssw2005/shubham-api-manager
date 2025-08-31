#include "network/http_protocol.hpp"
#include "network/tcp_connection.hpp"
#include "network/load_balancer.hpp"
#include <iostream>

int main() {
    std::cout << "Testing network protocol optimization compilation..." << std::endl;
    
    // Test HTTP protocol parsing
    ultra::network::Http11Parser parser;
    ultra::network::ProtocolDetector detector;
    
    const char* request = "GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n";
    auto protocol = detector.detect_protocol(request, std::strlen(request));
    
    std::cout << "Detected protocol: " << static_cast<int>(protocol) << std::endl;
    
    // Test TCP connection
    ultra::network::TcpConnection::Config config;
    config.enable_reuseport = true;
    config.enable_nodelay = true;
    
    std::cout << "TCP connection config created" << std::endl;
    
    // Test load balancer
    ultra::network::LoadBalancer::Config lb_config;
    lb_config.algorithm = ultra::network::LoadBalancingAlgorithm::CONSISTENT_HASH;
    
    std::cout << "Load balancer config created" << std::endl;
    
    std::cout << "Network protocol optimization compilation test passed!" << std::endl;
    return 0;
}