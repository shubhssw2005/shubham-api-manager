#include "performance-monitor/hardware_counters.hpp"
#include "common/logger.hpp"

#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <thread>

namespace ultra {
namespace monitor {

class HardwareCounters::PlatformImpl {
public:
    static constexpr int INVALID_FD = -1;
    
    struct PerfEventAttr {
        uint32_t type;
        uint64_t config;
        const char* name;
    };
    
    static const PerfEventAttr PERF_EVENTS[];
    static size_t get_perf_events_count();
};

const HardwareCounters::PlatformImpl::PerfEventAttr HardwareCounters::PlatformImpl::PERF_EVENTS[] = {
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cpu_cycles"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, "cache_references"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "cache_misses"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branch_instructions"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch_misses"},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "page_faults"},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "context_switches"},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "cpu_migrations"}
};

size_t HardwareCounters::PlatformImpl::get_perf_events_count() {
    return sizeof(PERF_EVENTS) / sizeof(PERF_EVENTS[0]);
}

HardwareCounters::HardwareCounters() 
    : impl_(std::make_unique<PlatformImpl>()) {
}

HardwareCounters::~HardwareCounters() {
    shutdown();
}

bool HardwareCounters::initialize(const std::vector<CounterConfig>& configs) {
    if (initialized_) {
        LOG_WARN("HardwareCounters already initialized");
        return true;
    }

    if (!is_pmu_available()) {
        last_error_ = "Performance monitoring unit not available";
        LOG_ERROR("{}", last_error_);
        return false;
    }

    // Get number of CPUs
    uint32_t num_cpus = std::thread::hardware_concurrency();
    
    // Setup counters for each CPU
    for (const auto& config : configs) {
        if (!config.enabled) continue;
        
        for (uint32_t cpu_id = 0; cpu_id < num_cpus; ++cpu_id) {
            if (!setup_counter(config.type, cpu_id)) {
                LOG_WARN("Failed to setup counter {} on CPU {}: {}", 
                        counter_type_to_string(config.type), cpu_id, last_error_);
                // Continue with other counters
            }
        }
    }

    if (counter_fds_.empty()) {
        last_error_ = "No performance counters could be initialized";
        LOG_ERROR("{}", last_error_);
        return false;
    }

    initialized_ = true;
    LOG_INFO("Initialized {} hardware performance counters across {} CPUs", 
             counter_fds_.size(), num_cpus);
    return true;
}

void HardwareCounters::shutdown() {
    if (!initialized_) return;
    
    stop_counting();
    cleanup_counters();
    initialized_ = false;
    
    LOG_INFO("Hardware counters shutdown complete");
}

bool HardwareCounters::start_counting() {
    if (!initialized_) {
        last_error_ = "Counters not initialized";
        return false;
    }

    if (counting_.exchange(true, std::memory_order_acq_rel)) {
        LOG_WARN("Counters already counting");
        return true;
    }

    // Enable all counters
    for (auto& counter : counter_fds_) {
        if (counter.fd != -1) {
            if (ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
                LOG_WARN("Failed to enable counter {} on CPU {}: {}", 
                        counter_type_to_string(counter.type), counter.cpu_id, strerror(errno));
            } else {
                counter.active = true;
            }
        }
    }

    LOG_INFO("Hardware counters started");
    return true;
}

bool HardwareCounters::stop_counting() {
    if (!counting_.exchange(false, std::memory_order_acq_rel)) {
        return true;
    }

    // Disable all counters
    for (auto& counter : counter_fds_) {
        if (counter.fd != -1 && counter.active) {
            if (ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {
                LOG_WARN("Failed to disable counter {} on CPU {}: {}", 
                        counter_type_to_string(counter.type), counter.cpu_id, strerror(errno));
            }
            counter.active = false;
        }
    }

    LOG_INFO("Hardware counters stopped");
    return true;
}

bool HardwareCounters::reset_counters() {
    if (!initialized_) {
        last_error_ = "Counters not initialized";
        return false;
    }

    // Reset all counters
    for (auto& counter : counter_fds_) {
        if (counter.fd != -1) {
            if (ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0) == -1) {
                LOG_WARN("Failed to reset counter {} on CPU {}: {}", 
                        counter_type_to_string(counter.type), counter.cpu_id, strerror(errno));
            }
        }
    }

    LOG_INFO("Hardware counters reset");
    return true;
}

HardwareCounters::CounterGroup HardwareCounters::read_counters() {
    CounterGroup group;
    group.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    group.cpu_id = sched_getcpu();  // Current CPU

    if (!initialized_ || !counting_.load(std::memory_order_acquire)) {
        return group;
    }

    // Read counters for current CPU
    for (const auto& counter : counter_fds_) {
        if (counter.fd != -1 && counter.active && counter.cpu_id == group.cpu_id) {
            uint64_t value = read_counter_value(counter.fd);
            
            CounterValue counter_value;
            counter_value.type = counter.type;
            counter_value.value = value;
            counter_value.time_enabled_ns = group.timestamp_ns;
            counter_value.time_running_ns = group.timestamp_ns;
            counter_value.scaling_factor = 1.0;
            
            group.counters.push_back(counter_value);
        }
    }

    return group;
}

std::vector<HardwareCounters::CounterGroup> HardwareCounters::read_all_cpus() {
    std::vector<CounterGroup> groups;
    
    if (!initialized_ || !counting_.load(std::memory_order_acquire)) {
        return groups;
    }

    uint32_t num_cpus = std::thread::hardware_concurrency();
    groups.reserve(num_cpus);

    for (uint32_t cpu_id = 0; cpu_id < num_cpus; ++cpu_id) {
        CounterGroup group;
        group.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
        group.cpu_id = cpu_id;

        // Read counters for this CPU
        for (const auto& counter : counter_fds_) {
            if (counter.fd != -1 && counter.active && counter.cpu_id == cpu_id) {
                uint64_t value = read_counter_value(counter.fd);
                
                CounterValue counter_value;
                counter_value.type = counter.type;
                counter_value.value = value;
                counter_value.time_enabled_ns = group.timestamp_ns;
                counter_value.time_running_ns = group.timestamp_ns;
                counter_value.scaling_factor = 1.0;
                
                group.counters.push_back(counter_value);
            }
        }

        if (!group.counters.empty()) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

double HardwareCounters::calculate_ipc(const CounterGroup& group) const {
    uint64_t cycles = 0, instructions = 0;
    
    for (const auto& counter : group.counters) {
        if (counter.type == CounterType::CPU_CYCLES) {
            cycles = counter.value;
        } else if (counter.type == CounterType::INSTRUCTIONS) {
            instructions = counter.value;
        }
    }
    
    return cycles > 0 ? static_cast<double>(instructions) / cycles : 0.0;
}

double HardwareCounters::calculate_cache_hit_rate(const CounterGroup& group) const {
    uint64_t cache_refs = 0, cache_misses = 0;
    
    for (const auto& counter : group.counters) {
        if (counter.type == CounterType::CACHE_REFERENCES) {
            cache_refs = counter.value;
        } else if (counter.type == CounterType::CACHE_MISSES) {
            cache_misses = counter.value;
        }
    }
    
    if (cache_refs > 0) {
        return 1.0 - (static_cast<double>(cache_misses) / cache_refs);
    }
    return 0.0;
}

double HardwareCounters::calculate_branch_prediction_rate(const CounterGroup& group) const {
    uint64_t branch_insts = 0, branch_misses = 0;
    
    for (const auto& counter : group.counters) {
        if (counter.type == CounterType::BRANCH_INSTRUCTIONS) {
            branch_insts = counter.value;
        } else if (counter.type == CounterType::BRANCH_MISSES) {
            branch_misses = counter.value;
        }
    }
    
    if (branch_insts > 0) {
        return 1.0 - (static_cast<double>(branch_misses) / branch_insts);
    }
    return 0.0;
}

double HardwareCounters::calculate_memory_bandwidth_gbps(const CounterGroup& group, uint64_t duration_ns) const {
    // This is a simplified calculation - actual memory bandwidth calculation
    // would require more specific counters and platform knowledge
    uint64_t cache_misses = 0;
    
    for (const auto& counter : group.counters) {
        if (counter.type == CounterType::CACHE_MISSES) {
            cache_misses = counter.value;
        }
    }
    
    if (duration_ns > 0 && cache_misses > 0) {
        // Assume 64 bytes per cache line
        constexpr uint64_t CACHE_LINE_SIZE = 64;
        uint64_t bytes_transferred = cache_misses * CACHE_LINE_SIZE;
        double seconds = static_cast<double>(duration_ns) / 1e9;
        double gbps = (bytes_transferred / seconds) / (1024 * 1024 * 1024);
        return gbps;
    }
    
    return 0.0;
}

bool HardwareCounters::is_pmu_available() {
    // Try to create a simple perf event to test availability
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd >= 0) {
        close(fd);
        return true;
    }
    
    return false;
}

std::vector<HardwareCounters::CounterType> HardwareCounters::get_available_counters() {
    std::vector<CounterType> available;
    
    // Test each counter type
    CounterType types[] = {
        CounterType::CPU_CYCLES,
        CounterType::INSTRUCTIONS,
        CounterType::CACHE_REFERENCES,
        CounterType::CACHE_MISSES,
        CounterType::BRANCH_INSTRUCTIONS,
        CounterType::BRANCH_MISSES,
        CounterType::PAGE_FAULTS,
        CounterType::CONTEXT_SWITCHES
    };
    
    for (auto type : types) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.size = sizeof(pe);
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        
        // Set type and config based on counter type
        switch (type) {
            case CounterType::CPU_CYCLES:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_CPU_CYCLES;
                break;
            case CounterType::INSTRUCTIONS:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                break;
            case CounterType::CACHE_REFERENCES:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
                break;
            case CounterType::CACHE_MISSES:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_CACHE_MISSES;
                break;
            case CounterType::BRANCH_INSTRUCTIONS:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
                break;
            case CounterType::BRANCH_MISSES:
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_BRANCH_MISSES;
                break;
            case CounterType::PAGE_FAULTS:
                pe.type = PERF_TYPE_SOFTWARE;
                pe.config = PERF_COUNT_SW_PAGE_FAULTS;
                break;
            case CounterType::CONTEXT_SWITCHES:
                pe.type = PERF_TYPE_SOFTWARE;
                pe.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
                break;
            default:
                continue;
        }
        
        int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        if (fd >= 0) {
            available.push_back(type);
            close(fd);
        }
    }
    
    return available;
}

std::string HardwareCounters::counter_type_to_string(CounterType type) {
    switch (type) {
        case CounterType::CPU_CYCLES: return "cpu_cycles";
        case CounterType::INSTRUCTIONS: return "instructions";
        case CounterType::CACHE_REFERENCES: return "cache_references";
        case CounterType::CACHE_MISSES: return "cache_misses";
        case CounterType::BRANCH_INSTRUCTIONS: return "branch_instructions";
        case CounterType::BRANCH_MISSES: return "branch_misses";
        case CounterType::PAGE_FAULTS: return "page_faults";
        case CounterType::CONTEXT_SWITCHES: return "context_switches";
        case CounterType::CPU_MIGRATIONS: return "cpu_migrations";
        case CounterType::L1D_CACHE_MISSES: return "l1d_cache_misses";
        case CounterType::L1I_CACHE_MISSES: return "l1i_cache_misses";
        case CounterType::LLC_CACHE_MISSES: return "llc_cache_misses";
        case CounterType::DTLB_MISSES: return "dtlb_misses";
        case CounterType::ITLB_MISSES: return "itlb_misses";
        default: return "unknown";
    }
}

bool HardwareCounters::setup_counter(CounterType type, uint32_t cpu_id) {
    int fd = create_perf_event(type, cpu_id);
    if (fd == -1) {
        last_error_ = "Failed to create perf event: " + std::string(strerror(errno));
        return false;
    }

    CounterDescriptor desc;
    desc.fd = fd;
    desc.type = type;
    desc.cpu_id = cpu_id;
    desc.active = false;

    counter_fds_.push_back(desc);
    return true;
}

void HardwareCounters::cleanup_counters() {
    for (auto& counter : counter_fds_) {
        if (counter.fd != -1) {
            close(counter.fd);
            counter.fd = -1;
        }
    }
    counter_fds_.clear();
}

uint64_t HardwareCounters::read_counter_value(int fd) const {
    uint64_t value = 0;
    ssize_t bytes_read = read(fd, &value, sizeof(value));
    if (bytes_read != sizeof(value)) {
        LOG_WARN("Failed to read counter value: {}", strerror(errno));
        return 0;
    }
    return value;
}

int HardwareCounters::create_perf_event(CounterType type, uint32_t cpu_id) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = get_perf_event_type(type);
    pe.config = get_perf_event_config(type);

    return syscall(__NR_perf_event_open, &pe, -1, cpu_id, -1, 0);
}

uint32_t HardwareCounters::get_perf_event_type(CounterType type) const {
    switch (type) {
        case CounterType::CPU_CYCLES:
        case CounterType::INSTRUCTIONS:
        case CounterType::CACHE_REFERENCES:
        case CounterType::CACHE_MISSES:
        case CounterType::BRANCH_INSTRUCTIONS:
        case CounterType::BRANCH_MISSES:
            return PERF_TYPE_HARDWARE;
        case CounterType::PAGE_FAULTS:
        case CounterType::CONTEXT_SWITCHES:
        case CounterType::CPU_MIGRATIONS:
            return PERF_TYPE_SOFTWARE;
        default:
            return PERF_TYPE_HARDWARE;
    }
}

uint64_t HardwareCounters::get_perf_event_config(CounterType type) const {
    switch (type) {
        case CounterType::CPU_CYCLES: return PERF_COUNT_HW_CPU_CYCLES;
        case CounterType::INSTRUCTIONS: return PERF_COUNT_HW_INSTRUCTIONS;
        case CounterType::CACHE_REFERENCES: return PERF_COUNT_HW_CACHE_REFERENCES;
        case CounterType::CACHE_MISSES: return PERF_COUNT_HW_CACHE_MISSES;
        case CounterType::BRANCH_INSTRUCTIONS: return PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        case CounterType::BRANCH_MISSES: return PERF_COUNT_HW_BRANCH_MISSES;
        case CounterType::PAGE_FAULTS: return PERF_COUNT_SW_PAGE_FAULTS;
        case CounterType::CONTEXT_SWITCHES: return PERF_COUNT_SW_CONTEXT_SWITCHES;
        case CounterType::CPU_MIGRATIONS: return PERF_COUNT_SW_CPU_MIGRATIONS;
        default: return 0;
    }
}

// ScopedCounterCollection implementation
ScopedCounterCollection::ScopedCounterCollection(HardwareCounters& counters)
    : counters_(counters) {
    if (counters_.is_initialized()) {
        start_values_ = counters_.read_counters();
        started_ = true;
    }
}

ScopedCounterCollection::~ScopedCounterCollection() {
    // Destructor - no cleanup needed for counters
}

HardwareCounters::CounterGroup ScopedCounterCollection::get_results() {
    if (!started_) {
        return HardwareCounters::CounterGroup{};
    }

    auto end_values = counters_.read_counters();
    
    // Calculate deltas
    HardwareCounters::CounterGroup result;
    result.timestamp_ns = end_values.timestamp_ns;
    result.cpu_id = end_values.cpu_id;
    
    for (const auto& end_counter : end_values.counters) {
        for (const auto& start_counter : start_values_.counters) {
            if (end_counter.type == start_counter.type) {
                HardwareCounters::CounterValue delta_counter;
                delta_counter.type = end_counter.type;
                delta_counter.value = end_counter.value - start_counter.value;
                delta_counter.time_enabled_ns = end_counter.time_enabled_ns - start_counter.time_enabled_ns;
                delta_counter.time_running_ns = end_counter.time_running_ns - start_counter.time_running_ns;
                delta_counter.scaling_factor = end_counter.scaling_factor;
                
                result.counters.push_back(delta_counter);
                break;
            }
        }
    }
    
    return result;
}

} // namespace monitor
} // namespace ultra