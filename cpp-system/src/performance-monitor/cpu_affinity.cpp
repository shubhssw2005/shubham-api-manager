#include "performance-monitor/cpu_affinity.hpp"
#include "common/logger.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <filesystem>

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/getcpu.h>
#endif

namespace ultra_cpp {
namespace performance {

CPUAffinityManager::CPUAffinityManager() 
    : numa_available_(false) {
}

CPUAffinityManager::~CPUAffinityManager() = default;

bool CPUAffinityManager::initialize() {
    LOG_INFO("Initializing CPU affinity manager");
    
    if (!discover_cpu_topology()) {
        LOG_ERROR("Failed to discover CPU topology");
        return false;
    }
    
    if (!discover_numa_topology()) {
        LOG_WARN("NUMA topology discovery failed, continuing without NUMA support");
        numa_available_ = false;
    } else {
        numa_available_ = true;
    }
    
    LOG_INFO("CPU topology discovered: {} logical CPUs, {} physical cores, {} NUMA nodes",
             topology_.total_logical_cpus, topology_.total_physical_cores, 
             topology_.total_numa_nodes);
    
    return true;
}

bool CPUAffinityManager::discover_cpu_topology() {
#ifdef __linux__
    topology_.total_logical_cpus = std::thread::hardware_concurrency();
    
    // Parse /proc/cpuinfo for detailed CPU information
    parse_proc_cpuinfo();
    
    // Parse /sys/devices/system/cpu for additional topology info
    parse_sys_devices();
    
    // Count physical cores (unique physical_id + core_id combinations)
    std::set<std::pair<int, int>> unique_cores;
    for (const auto& core : topology_.cores) {
        unique_cores.insert({core.physical_id, core.core_id});
    }
    topology_.total_physical_cores = unique_cores.size();
    
    return true;
#else
    // Fallback for non-Linux systems
    topology_.total_logical_cpus = std::thread::hardware_concurrency();
    topology_.total_physical_cores = topology_.total_logical_cpus;
    topology_.total_numa_nodes = 1;
    
    // Create simple topology
    for (int i = 0; i < topology_.total_logical_cpus; ++i) {
        CPUTopology::Core core;
        core.physical_id = i;
        core.core_id = i;
        core.numa_node = 0;
        core.logical_cpus = {i};
        core.is_hyperthread_sibling = false;
        topology_.cores.push_back(core);
    }
    
    return true;
#endif
}

void CPUAffinityManager::parse_proc_cpuinfo() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        LOG_ERROR("Cannot open /proc/cpuinfo");
        return;
    }
    
    std::string line;
    CPUTopology::Core current_core;
    bool core_valid = false;
    
    while (std::getline(cpuinfo, line)) {
        if (line.empty()) {
            if (core_valid) {
                topology_.cores.push_back(current_core);
                current_core = CPUTopology::Core{};
                core_valid = false;
            }
            continue;
        }
        
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, ':') && std::getline(iss, value)) {
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            
            if (key == "processor") {
                int cpu_id = std::stoi(value);
                current_core.logical_cpus = {cpu_id};
                core_valid = true;
            } else if (key == "physical id") {
                current_core.physical_id = std::stoi(value);
            } else if (key == "core id") {
                current_core.core_id = std::stoi(value);
            }
        }
    }
    
    // Add the last core if valid
    if (core_valid) {
        topology_.cores.push_back(current_core);
    }
#endif
}

void CPUAffinityManager::parse_sys_devices() {
#ifdef __linux__
    namespace fs = std::filesystem;
    
    for (auto& core : topology_.cores) {
        if (core.logical_cpus.empty()) continue;
        
        int cpu_id = core.logical_cpus[0];
        std::string cpu_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id);
        
        // Read NUMA node
        std::string numa_path = cpu_path + "/node";
        if (fs::exists(numa_path)) {
            try {
                auto numa_link = fs::read_symlink(numa_path);
                std::string numa_str = numa_link.filename().string();
                if (numa_str.substr(0, 4) == "node") {
                    core.numa_node = std::stoi(numa_str.substr(4));
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to read NUMA node for CPU {}: {}", cpu_id, e.what());
                core.numa_node = 0;
            }
        }
        
        // Check for hyperthread siblings
        std::string siblings_path = cpu_path + "/topology/thread_siblings_list";
        std::ifstream siblings_file(siblings_path);
        if (siblings_file.is_open()) {
            std::string siblings_str;
            std::getline(siblings_file, siblings_str);
            
            // Parse comma-separated list or ranges
            std::vector<int> siblings;
            std::regex range_regex(R"((\d+)(?:-(\d+))?)");
            std::sregex_iterator iter(siblings_str.begin(), siblings_str.end(), range_regex);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                int start = std::stoi((*iter)[1]);
                int end_val = (*iter)[2].matched ? std::stoi((*iter)[2]) : start;
                
                for (int i = start; i <= end_val; ++i) {
                    siblings.push_back(i);
                }
            }
            
            core.is_hyperthread_sibling = siblings.size() > 1;
        }
    }
#endif
}

bool CPUAffinityManager::discover_numa_topology() {
#ifdef __linux__
    if (numa_available() < 0) {
        return false;
    }
    
    int max_node = numa_max_node();
    topology_.total_numa_nodes = max_node + 1;
    
    for (int node = 0; node <= max_node; ++node) {
        if (numa_bitmask_isbitset(numa_nodes_ptr, node)) {
            CPUTopology::NUMANode numa_node;
            numa_node.node_id = node;
            
            // Get CPUs for this NUMA node
            struct bitmask* cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, cpus) == 0) {
                for (int cpu = 0; cpu < numa_num_possible_cpus(); ++cpu) {
                    if (numa_bitmask_isbitset(cpus, cpu)) {
                        numa_node.cpus.push_back(cpu);
                    }
                }
            }
            numa_free_cpumask(cpus);
            
            // Get memory size
            long long memory_size = numa_node_size64(node, nullptr);
            numa_node.memory_size_mb = memory_size / (1024 * 1024);
            
            // Get distances to other nodes
            numa_node.distances.resize(topology_.total_numa_nodes);
            for (int other_node = 0; other_node <= max_node; ++other_node) {
                numa_node.distances[other_node] = numa_distance(node, other_node);
            }
            
            topology_.numa_nodes.push_back(numa_node);
        }
    }
    
    return true;
#else
    return false;
#endif
}

bool CPUAffinityManager::set_thread_affinity(const AffinityConfig& config) {
#ifdef __linux__
    std::vector<int> cpus = get_optimal_cpus(config);
    if (cpus.empty()) {
        LOG_ERROR("No CPUs available for affinity setting");
        return false;
    }
    
    cpu_set_t cpu_set = create_cpu_set(cpus);
    
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        LOG_ERROR("Failed to set CPU affinity: {}", strerror(errno));
        return false;
    }
    
    // Set NUMA memory policy if requested
    if (config.strategy == AffinityStrategy::NUMA_LOCAL && numa_available_) {
        int numa_node = config.preferred_numa_node;
        if (numa_node == -1) {
            numa_node = find_best_numa_node();
        }
        set_numa_memory_policy(numa_node);
    }
    
    LOG_DEBUG("Set thread affinity to CPUs: [{}]", 
              [&cpus]() {
                  std::ostringstream oss;
                  for (size_t i = 0; i < cpus.size(); ++i) {
                      if (i > 0) oss << ", ";
                      oss << cpus[i];
                  }
                  return oss.str();
              }());
    
    return true;
#else
    LOG_WARN("CPU affinity not supported on this platform");
    return false;
#endif
}

bool CPUAffinityManager::set_thread_affinity(std::thread& thread, const AffinityConfig& config) {
#ifdef __linux__
    std::vector<int> cpus = get_optimal_cpus(config);
    if (cpus.empty()) {
        return false;
    }
    
    cpu_set_t cpu_set = create_cpu_set(cpus);
    
    if (pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set), &cpu_set) != 0) {
        LOG_ERROR("Failed to set thread CPU affinity: {}", strerror(errno));
        return false;
    }
    
    return true;
#else
    return false;
#endif
}

bool CPUAffinityManager::set_numa_memory_policy(int numa_node, bool strict) {
#ifdef __linux__
    if (!numa_available_) {
        return false;
    }
    
    struct bitmask* nodes = numa_allocate_nodemask();
    numa_bitmask_setbit(nodes, numa_node);
    
    int policy = strict ? MPOL_BIND : MPOL_PREFERRED;
    int result = set_mempolicy(policy, nodes->maskp, nodes->size + 1);
    
    numa_free_nodemask(nodes);
    
    if (result != 0) {
        LOG_ERROR("Failed to set NUMA memory policy: {}", strerror(errno));
        return false;
    }
    
    LOG_DEBUG("Set NUMA memory policy to node {} (strict: {})", numa_node, strict);
    return true;
#else
    return false;
#endif
}

std::vector<int> CPUAffinityManager::get_optimal_cpus(const AffinityConfig& config) const {
    std::vector<int> cpus;
    
    switch (config.strategy) {
        case AffinityStrategy::NONE:
            // Return all available CPUs
            for (int i = 0; i < topology_.total_logical_cpus; ++i) {
                cpus.push_back(i);
            }
            break;
            
        case AffinityStrategy::SINGLE_CORE:
            // Return first available CPU
            if (topology_.total_logical_cpus > 0) {
                cpus.push_back(0);
            }
            break;
            
        case AffinityStrategy::NUMA_LOCAL: {
            int numa_node = config.preferred_numa_node;
            if (numa_node == -1) {
                numa_node = find_best_numa_node();
            }
            
            // Find CPUs in the specified NUMA node
            for (const auto& node : topology_.numa_nodes) {
                if (node.node_id == numa_node) {
                    cpus = node.cpus;
                    break;
                }
            }
            break;
        }
        
        case AffinityStrategy::PERFORMANCE_CORES:
            cpus = get_performance_cores();
            break;
            
        case AffinityStrategy::ISOLATED_CORES:
            cpus = get_isolated_cores();
            if (cpus.empty()) {
                // Fallback to performance cores
                cpus = get_performance_cores();
            }
            break;
            
        case AffinityStrategy::CUSTOM:
            cpus = config.custom_cpus;
            break;
    }
    
    // Filter hyperthread siblings if requested
    if (config.avoid_hyperthread_siblings) {
        cpus = filter_hyperthread_siblings(cpus);
    }
    
    return cpus;
}

std::vector<int> CPUAffinityManager::get_performance_cores() const {
    std::vector<int> cpus;
    
    // Prefer cores that are not hyperthread siblings
    for (const auto& core : topology_.cores) {
        if (!core.is_hyperthread_sibling && !core.logical_cpus.empty()) {
            cpus.insert(cpus.end(), core.logical_cpus.begin(), core.logical_cpus.end());
        }
    }
    
    // If no non-HT cores found, use all cores
    if (cpus.empty()) {
        for (const auto& core : topology_.cores) {
            cpus.insert(cpus.end(), core.logical_cpus.begin(), core.logical_cpus.end());
        }
    }
    
    return cpus;
}

std::vector<int> CPUAffinityManager::get_isolated_cores() const {
#ifdef __linux__
    std::vector<int> isolated_cpus;
    
    // Read isolated CPUs from /sys/devices/system/cpu/isolated
    std::ifstream isolated_file("/sys/devices/system/cpu/isolated");
    if (isolated_file.is_open()) {
        std::string isolated_str;
        std::getline(isolated_file, isolated_str);
        
        // Parse comma-separated list or ranges
        std::regex range_regex(R"((\d+)(?:-(\d+))?)");
        std::sregex_iterator iter(isolated_str.begin(), isolated_str.end(), range_regex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            int start = std::stoi((*iter)[1]);
            int end_val = (*iter)[2].matched ? std::stoi((*iter)[2]) : start;
            
            for (int i = start; i <= end_val; ++i) {
                isolated_cpus.push_back(i);
            }
        }
    }
    
    return isolated_cpus;
#else
    return {};
#endif
}

std::vector<int> CPUAffinityManager::filter_hyperthread_siblings(const std::vector<int>& cpus) const {
    std::vector<int> filtered_cpus;
    std::set<std::pair<int, int>> used_cores; // (physical_id, core_id)
    
    for (int cpu : cpus) {
        // Find the core for this CPU
        for (const auto& core : topology_.cores) {
            auto it = std::find(core.logical_cpus.begin(), core.logical_cpus.end(), cpu);
            if (it != core.logical_cpus.end()) {
                std::pair<int, int> core_id = {core.physical_id, core.core_id};
                
                // Only use the first logical CPU from each physical core
                if (used_cores.find(core_id) == used_cores.end()) {
                    filtered_cpus.push_back(cpu);
                    used_cores.insert(core_id);
                }
                break;
            }
        }
    }
    
    return filtered_cpus;
}

int CPUAffinityManager::find_best_numa_node() const {
    if (topology_.numa_nodes.empty()) {
        return 0;
    }
    
    // Find NUMA node with most available memory
    int best_node = 0;
    size_t max_memory = 0;
    
    for (const auto& node : topology_.numa_nodes) {
        if (node.memory_size_mb > max_memory) {
            max_memory = node.memory_size_mb;
            best_node = node.node_id;
        }
    }
    
    return best_node;
}

int CPUAffinityManager::get_current_cpu() const {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

int CPUAffinityManager::get_current_numa_node() const {
#ifdef __linux__
    if (!numa_available_) {
        return -1;
    }
    
    int cpu = get_current_cpu();
    if (cpu < 0) {
        return -1;
    }
    
    return numa_node_of_cpu(cpu);
#else
    return -1;
#endif
}

#ifdef __linux__
cpu_set_t CPUAffinityManager::create_cpu_set(const std::vector<int>& cpus) const {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &cpu_set);
        }
    }
    
    return cpu_set;
}
#endif

// ScopedCPUAffinity implementation
ScopedCPUAffinity::ScopedCPUAffinity(const CPUAffinityManager::AffinityConfig& config)
    : valid_(false) {
#ifdef __linux__
    // Save current affinity
    if (sched_getaffinity(0, sizeof(original_affinity_), &original_affinity_) == 0) {
        CPUAffinityManager manager;
        if (manager.initialize() && manager.set_thread_affinity(config)) {
            valid_ = true;
        }
    }
#endif
}

ScopedCPUAffinity::~ScopedCPUAffinity() {
#ifdef __linux__
    if (valid_) {
        sched_setaffinity(0, sizeof(original_affinity_), &original_affinity_);
    }
#endif
}

} // namespace performance
} // namespace ultra_cpp