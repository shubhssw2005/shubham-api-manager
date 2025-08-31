#include "include/cache/ultra_cache.hpp"
#include "include/common/types.hpp"
#include <iostream>
#include <string>

int main() {
    try {
        // Test basic cache creation
        ultra::cache::UltraCache<std::string, std::string>::Config config;
        config.capacity = 1000;
        config.shard_count = 4;
        config.enable_rdma = false;
        
        std::cout << "Creating cache with capacity: " << config.capacity << std::endl;
        
        // This will test if the headers compile correctly
        // We can't fully test without the implementation being compiled
        std::cout << "Cache configuration created successfully!" << std::endl;
        std::cout << "Shard count: " << config.shard_count << std::endl;
        std::cout << "RDMA enabled: " << (config.enable_rdma ? "yes" : "no") << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}