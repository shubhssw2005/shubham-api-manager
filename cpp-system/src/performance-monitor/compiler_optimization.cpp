#include "performance-monitor/compiler_optimization.hpp"
#include "common/logger.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <filesystem>
#include <regex>
#include <cstdlib>

#ifdef __linux__
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ultra_cpp {
namespace performance {

CompilerOptimizationManager::CompilerOptimizationManager() {
    compiler_info_ = detect_compiler();
    create_builtin_profiles();
}

CompilerOptimizationManager::~CompilerOptimizationManager() = default;

void CompilerOptimizationManager::create_builtin_profiles() {
    // Debug profile
    OptimizationProfile debug_profile;
    debug_profile.name = "debug";
    debug_profile.level = OptimizationLevel::DEBUG;
    debug_profile.arch = TargetArchitecture::GENERIC;
    debug_profile.compiler_flags = {"-O0", "-g", "-DDEBUG"};
    debug_profile.defines["DEBUG"] = "1";
    register_profile(debug_profile);
    
    // Release profile
    OptimizationProfile release_profile;
    release_profile.name = "release";
    release_profile.level = OptimizationLevel::AGGRESSIVE;
    release_profile.arch = TargetArchitecture::NATIVE;
    release_profile.compiler_flags = {"-O3", "-march=native", "-DNDEBUG"};
    release_profile.enable_lto = true;
    release_profile.defines["NDEBUG"] = "1";
    register_profile(release_profile);
    
    // Performance profile with PGO
    OptimizationProfile perf_profile;
    perf_profile.name = "performance";
    perf_profile.level = OptimizationLevel::AGGRESSIVE;
    perf_profile.arch = TargetArchitecture::NATIVE;
    perf_profile.compiler_flags = {
        "-O3", "-march=native", "-mtune=native",
        "-ffast-math", "-funroll-loops", "-fvectorize",
        "-DNDEBUG"
    };
    perf_profile.enable_lto = true;
    perf_profile.enable_pgo = true;
    perf_profile.defines["NDEBUG"] = "1";
    register_profile(perf_profile);
    
    // Size-optimized profile
    OptimizationProfile size_profile;
    size_profile.name = "size";
    size_profile.level = OptimizationLevel::SIZE;
    size_profile.arch = TargetArchitecture::GENERIC;
    size_profile.compiler_flags = {"-Os", "-DNDEBUG"};
    size_profile.enable_lto = true;
    size_profile.defines["NDEBUG"] = "1";
    register_profile(size_profile);
    
    // Ultra-performance profile
    OptimizationProfile ultra_profile;
    ultra_profile.name = "ultra";
    ultra_profile.level = OptimizationLevel::FAST;
    ultra_profile.arch = TargetArchitecture::NATIVE;
    ultra_profile.compiler_flags = {
        "-Ofast", "-march=native", "-mtune=native",
        "-ffast-math", "-funroll-loops", "-fvectorize",
        "-flto", "-fwhole-program-vtables",
        "-DNDEBUG", "-DULTRA_PERFORMANCE"
    };
    ultra_profile.enable_lto = true;
    ultra_profile.enable_pgo = true;
    ultra_profile.enable_bolt = true;
    ultra_profile.defines["NDEBUG"] = "1";
    ultra_profile.defines["ULTRA_PERFORMANCE"] = "1";
    register_profile(ultra_profile);
}

void CompilerOptimizationManager::register_profile(const OptimizationProfile& profile) {
    profiles_[profile.name] = profile;
    LOG_DEBUG("Registered optimization profile: {}", profile.name);
}

CompilerOptimizationManager::OptimizationProfile 
CompilerOptimizationManager::get_profile(const std::string& name) const {
    auto it = profiles_.find(name);
    if (it != profiles_.end()) {
        return it->second;
    }
    
    LOG_WARN("Profile '{}' not found, returning default profile", name);
    return profiles_.at("release");
}

std::vector<std::string> CompilerOptimizationManager::list_profiles() const {
    std::vector<std::string> names;
    names.reserve(profiles_.size());
    
    for (const auto& [name, profile] : profiles_) {
        names.push_back(name);
    }
    
    return names;
}

CompilerOptimizationManager::CompilerInfo CompilerOptimizationManager::detect_compiler() const {
    CompilerInfo info;
    info.type = CompilerType::UNKNOWN;
    
    // Try to detect compiler
    std::string output;
    if (execute_command("gcc --version", output)) {
        if (output.find("gcc") != std::string::npos) {
            info.type = CompilerType::GCC;
            
            // Extract version
            std::regex version_regex(R"(gcc.*?(\d+\.\d+\.\d+))");
            std::smatch match;
            if (std::regex_search(output, match, version_regex)) {
                info.version = match[1].str();
            }
        }
    } else if (execute_command("clang --version", output)) {
        if (output.find("clang") != std::string::npos) {
            info.type = CompilerType::CLANG;
            
            // Extract version
            std::regex version_regex(R"(clang version (\d+\.\d+\.\d+))");
            std::smatch match;
            if (std::regex_search(output, match, version_regex)) {
                info.version = match[1].str();
            }
        }
    }
    
    // Check capabilities based on compiler type
    switch (info.type) {
        case CompilerType::GCC:
            info.supports_pgo = true;
            info.supports_lto = true;
            info.supports_thinlto = false;
            info.supports_bolt = false;
            info.supported_flags = {
                "-O0", "-O1", "-O2", "-O3", "-Os", "-Ofast",
                "-march=native", "-mtune=native", "-flto",
                "-fprofile-generate", "-fprofile-use",
                "-funroll-loops", "-fvectorize"
            };
            break;
            
        case CompilerType::CLANG:
            info.supports_pgo = true;
            info.supports_lto = true;
            info.supports_thinlto = true;
            info.supports_bolt = true;
            info.supported_flags = {
                "-O0", "-O1", "-O2", "-O3", "-Os", "-Ofast",
                "-march=native", "-mtune=native", "-flto", "-flto=thin",
                "-fprofile-instr-generate", "-fprofile-instr-use",
                "-funroll-loops", "-fvectorize"
            };
            break;
            
        default:
            info.supports_pgo = false;
            info.supports_lto = false;
            info.supports_thinlto = false;
            info.supports_bolt = false;
            break;
    }
    
    LOG_INFO("Detected compiler: {} version {}", 
             info.type == CompilerType::GCC ? "GCC" : 
             info.type == CompilerType::CLANG ? "Clang" : "Unknown",
             info.version);
    
    return info;
}

bool CompilerOptimizationManager::generate_pgo_profile(const PGOWorkflow& workflow) {
    LOG_INFO("Generating PGO profile for: {}", workflow.profile_name);
    
    // Create profile output directory
    std::filesystem::create_directories(workflow.profile_output_dir);
    
    // Build with profile generation
    std::string build_cmd;
    if (compiler_info_.type == CompilerType::GCC) {
        build_cmd = "cd " + workflow.build_dir + " && "
                   "cmake -DCMAKE_CXX_FLAGS=\"-fprofile-generate=" + 
                   workflow.profile_output_dir + "\" " + workflow.source_dir + " && "
                   "make -j$(nproc)";
    } else if (compiler_info_.type == CompilerType::CLANG) {
        build_cmd = "cd " + workflow.build_dir + " && "
                   "cmake -DCMAKE_CXX_FLAGS=\"-fprofile-instr-generate=" + 
                   workflow.profile_output_dir + "/default.profraw\" " + 
                   workflow.source_dir + " && "
                   "make -j$(nproc)";
    } else {
        LOG_ERROR("PGO not supported for this compiler");
        return false;
    }
    
    std::string output;
    if (!execute_command(build_cmd, output)) {
        LOG_ERROR("Failed to build with profile generation: {}", output);
        return false;
    }
    
    // Run benchmark to generate profile data
    std::string benchmark_cmd = workflow.benchmark_executable;
    for (const auto& arg : workflow.benchmark_args) {
        benchmark_cmd += " " + arg;
    }
    
    if (!execute_command(benchmark_cmd, output)) {
        LOG_ERROR("Failed to run benchmark for profile generation: {}", output);
        return false;
    }
    
    // Process profile data (for Clang)
    if (compiler_info_.type == CompilerType::CLANG) {
        std::string merge_cmd = "llvm-profdata merge -output=" + 
                               workflow.profile_output_dir + "/default.profdata " +
                               workflow.profile_output_dir + "/default.profraw";
        
        if (!execute_command(merge_cmd, output)) {
            LOG_ERROR("Failed to merge profile data: {}", output);
            return false;
        }
    }
    
    LOG_INFO("PGO profile generation completed successfully");
    return true;
}

bool CompilerOptimizationManager::build_with_pgo(const PGOWorkflow& workflow) {
    LOG_INFO("Building with PGO profile: {}", workflow.profile_name);
    
    std::string profile_path;
    if (compiler_info_.type == CompilerType::GCC) {
        profile_path = workflow.profile_output_dir;
    } else if (compiler_info_.type == CompilerType::CLANG) {
        profile_path = workflow.profile_output_dir + "/default.profdata";
    } else {
        LOG_ERROR("PGO not supported for this compiler");
        return false;
    }
    
    // Check if profile exists
    if (!file_exists(profile_path)) {
        LOG_ERROR("PGO profile not found: {}", profile_path);
        return false;
    }
    
    // Build with profile use
    std::string build_cmd;
    if (compiler_info_.type == CompilerType::GCC) {
        build_cmd = "cd " + workflow.build_dir + " && "
                   "cmake -DCMAKE_CXX_FLAGS=\"-fprofile-use=" + profile_path + 
                   " -O3 -march=native\" " + workflow.source_dir + " && "
                   "make -j$(nproc)";
    } else if (compiler_info_.type == CompilerType::CLANG) {
        build_cmd = "cd " + workflow.build_dir + " && "
                   "cmake -DCMAKE_CXX_FLAGS=\"-fprofile-instr-use=" + profile_path + 
                   " -O3 -march=native\" " + workflow.source_dir + " && "
                   "make -j$(nproc)";
    }
    
    std::string output;
    if (!execute_command(build_cmd, output)) {
        LOG_ERROR("Failed to build with PGO: {}", output);
        return false;
    }
    
    LOG_INFO("PGO build completed successfully");
    return true;
}

std::string CompilerOptimizationManager::generate_cmake_flags(const OptimizationProfile& profile) const {
    std::ostringstream flags;
    
    // Optimization level
    flags << get_optimization_flag(profile.level) << " ";
    
    // Architecture
    flags << get_architecture_flag(profile.arch) << " ";
    
    // Additional compiler flags
    for (const auto& flag : profile.compiler_flags) {
        flags << flag << " ";
    }
    
    // LTO flags
    if (profile.enable_lto) {
        auto lto_flags = get_lto_flags();
        for (const auto& flag : lto_flags) {
            flags << flag << " ";
        }
    }
    
    // PGO flags
    if (profile.enable_pgo && !profile.pgo_profile_path.empty()) {
        auto pgo_flags = get_pgo_use_flags(profile.pgo_profile_path);
        for (const auto& flag : pgo_flags) {
            flags << flag << " ";
        }
    }
    
    // Defines
    for (const auto& [key, value] : profile.defines) {
        flags << "-D" << key;
        if (!value.empty()) {
            flags << "=" << value;
        }
        flags << " ";
    }
    
    return flags.str();
}

CompilerOptimizationManager::BenchmarkResult 
CompilerOptimizationManager::benchmark_profile(const OptimizationProfile& profile,
                                              const std::string& benchmark_command,
                                              size_t iterations) const {
    BenchmarkResult result;
    result.profile_name = profile.name;
    
    std::vector<std::chrono::nanoseconds> times;
    times.reserve(iterations);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string output;
        if (!execute_command(benchmark_command, output)) {
            LOG_WARN("Benchmark iteration {} failed", i);
            continue;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
    }
    
    if (!times.empty()) {
        // Calculate median time
        std::sort(times.begin(), times.end());
        result.execution_time = times[times.size() / 2];
        
        // Calculate performance score (inverse of execution time)
        result.performance_score = 1.0 / result.execution_time.count();
    }
    
    return result;
}

std::string CompilerOptimizationManager::get_optimization_flag(OptimizationLevel level) const {
    switch (level) {
        case OptimizationLevel::DEBUG: return "-O0";
        case OptimizationLevel::BASIC: return "-O1";
        case OptimizationLevel::STANDARD: return "-O2";
        case OptimizationLevel::AGGRESSIVE: return "-O3";
        case OptimizationLevel::SIZE: return "-Os";
        case OptimizationLevel::FAST: return "-Ofast";
        case OptimizationLevel::PROFILE_GEN:
            return compiler_info_.type == CompilerType::GCC ? 
                   "-fprofile-generate" : "-fprofile-instr-generate";
        case OptimizationLevel::PROFILE_USE:
            return compiler_info_.type == CompilerType::GCC ? 
                   "-fprofile-use" : "-fprofile-instr-use";
        default: return "-O2";
    }
}

std::string CompilerOptimizationManager::get_architecture_flag(TargetArchitecture arch) const {
    switch (arch) {
        case TargetArchitecture::GENERIC: return "";
        case TargetArchitecture::NATIVE: return "-march=native -mtune=native";
        case TargetArchitecture::HASWELL: return "-march=haswell -mtune=haswell";
        case TargetArchitecture::SKYLAKE: return "-march=skylake -mtune=skylake";
        case TargetArchitecture::ZEN2: return "-march=znver2 -mtune=znver2";
        case TargetArchitecture::ZEN3: return "-march=znver3 -mtune=znver3";
        case TargetArchitecture::ARM_CORTEX_A72: return "-march=armv8-a+crc -mtune=cortex-a72";
        case TargetArchitecture::ARM_NEOVERSE_N1: return "-march=armv8.2-a -mtune=neoverse-n1";
        default: return "";
    }
}

std::vector<std::string> CompilerOptimizationManager::get_lto_flags() const {
    if (compiler_info_.type == CompilerType::GCC) {
        return {"-flto", "-fuse-linker-plugin"};
    } else if (compiler_info_.type == CompilerType::CLANG) {
        return {"-flto"};
    }
    return {};
}

bool CompilerOptimizationManager::execute_command(const std::string& command, std::string& output) const {
#ifdef __linux__
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }
    
    char buffer[128];
    output.clear();
    
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    
    int status = pclose(pipe);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
    // Fallback for non-Linux systems
    int result = std::system(command.c_str());
    return result == 0;
#endif
}

bool CompilerOptimizationManager::file_exists(const std::string& path) const {
    return std::filesystem::exists(path);
}

size_t CompilerOptimizationManager::get_file_size(const std::string& path) const {
    try {
        return std::filesystem::file_size(path);
    } catch (const std::filesystem::filesystem_error&) {
        return 0;
    }
}

// PGOHelper implementation
PGOHelper::PGOHelper(const std::string& profile_dir) 
    : profile_dir_(profile_dir), profiling_active_(false) {
    std::filesystem::create_directories(profile_dir_);
}

PGOHelper::~PGOHelper() {
    if (profiling_active_) {
        stop_profiling();
    }
}

void PGOHelper::add_training_workload(const TrainingWorkload& workload) {
    workloads_.push_back(workload);
    LOG_DEBUG("Added training workload: {}", workload.name);
}

void PGOHelper::run_training_workloads() {
    LOG_INFO("Running {} training workloads for PGO", workloads_.size());
    
    start_profiling();
    
    for (const auto& workload : workloads_) {
        LOG_DEBUG("Running workload: {}", workload.name);
        
        try {
            workload.workload();
        } catch (const std::exception& e) {
            LOG_ERROR("Workload '{}' failed: {}", workload.name, e.what());
        }
    }
    
    stop_profiling();
    LOG_INFO("Training workloads completed");
}

void PGOHelper::start_profiling() {
    if (profiling_active_) return;
    
    // Set environment variable for profile output
    std::string profile_path = profile_dir_ + "/default.profraw";
    setenv("LLVM_PROFILE_FILE", profile_path.c_str(), 1);
    
    profiling_active_ = true;
    LOG_DEBUG("Started PGO profiling, output: {}", profile_path);
}

void PGOHelper::stop_profiling() {
    if (!profiling_active_) return;
    
    unsetenv("LLVM_PROFILE_FILE");
    profiling_active_ = false;
    
    LOG_DEBUG("Stopped PGO profiling");
}

// BOLTHelper implementation
BOLTHelper::BOLTHelper(const BOLTConfig& config) : config_(config) {
    if (!check_bolt_availability()) {
        LOG_WARN("BOLT not available, binary optimization will be disabled");
    }
    
    if (!check_perf_availability()) {
        LOG_WARN("perf not available, profile collection may be limited");
    }
}

BOLTHelper::~BOLTHelper() = default;

bool BOLTHelper::check_bolt_availability() const {
    std::string output;
    return std::system("which llvm-bolt > /dev/null 2>&1") == 0;
}

bool BOLTHelper::check_perf_availability() const {
    return std::system("which perf > /dev/null 2>&1") == 0;
}

bool BOLTHelper::collect_perf_profile(const std::string& binary_path,
                                     const std::string& workload_command,
                                     const std::string& output_profile) {
    std::string perf_cmd = "perf record -e cycles:u -j any,u -o " + output_profile + 
                          " -- " + workload_command;
    
    LOG_INFO("Collecting perf profile: {}", perf_cmd);
    
    int result = std::system(perf_cmd.c_str());
    if (result != 0) {
        LOG_ERROR("Failed to collect perf profile");
        return false;
    }
    
    LOG_INFO("Perf profile collected: {}", output_profile);
    return true;
}

bool BOLTHelper::optimize_binary(const std::string& input_binary,
                                const std::string& profile_path,
                                const std::string& output_binary) {
    std::ostringstream bolt_cmd;
    bolt_cmd << config_.bolt_binary_path << " " << input_binary
             << " -data=" << profile_path
             << " -o " << output_binary;
    
    for (const auto& flag : config_.optimization_flags) {
        bolt_cmd << " " << flag;
    }
    
    LOG_INFO("Optimizing binary with BOLT: {}", bolt_cmd.str());
    
    int result = std::system(bolt_cmd.str().c_str());
    if (result != 0) {
        LOG_ERROR("BOLT optimization failed");
        return false;
    }
    
    LOG_INFO("BOLT optimization completed: {}", output_binary);
    return true;
}

} // namespace performance
} // namespace ultra_cpp