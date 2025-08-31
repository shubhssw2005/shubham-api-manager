#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>

namespace ultra_cpp {
namespace performance {

/**
 * Compiler optimization profile manager
 */
class CompilerOptimizationManager {
public:
    enum class OptimizationLevel {
        DEBUG,          // -O0, debug symbols, no optimization
        BASIC,          // -O1, basic optimization
        STANDARD,       // -O2, standard optimization
        AGGRESSIVE,     // -O3, aggressive optimization
        SIZE,           // -Os, optimize for size
        FAST,           // -Ofast, fastest possible (may break standards)
        PROFILE_GEN,    // Profile generation build
        PROFILE_USE     // Profile-guided optimization build
    };
    
    enum class TargetArchitecture {
        GENERIC,        // Generic x86_64
        NATIVE,         // -march=native
        HASWELL,        // Intel Haswell
        SKYLAKE,        // Intel Skylake
        ZEN2,           // AMD Zen 2
        ZEN3,           // AMD Zen 3
        ARM_CORTEX_A72, // ARM Cortex-A72
        ARM_NEOVERSE_N1 // ARM Neoverse N1
    };
    
    struct OptimizationProfile {
        std::string name;
        OptimizationLevel level;
        TargetArchitecture arch;
        std::vector<std::string> compiler_flags;
        std::vector<std::string> linker_flags;
        std::unordered_map<std::string, std::string> defines;
        bool enable_lto = false;           // Link Time Optimization
        bool enable_pgo = false;           // Profile Guided Optimization
        bool enable_bolt = false;          // Binary Optimization and Layout Tool
        bool enable_thinlto = false;       // ThinLTO
        std::string pgo_profile_path;
        std::string bolt_profile_path;
    };
    
    CompilerOptimizationManager();
    ~CompilerOptimizationManager();
    
    // Profile management
    void register_profile(const OptimizationProfile& profile);
    OptimizationProfile get_profile(const std::string& name) const;
    std::vector<std::string> list_profiles() const;
    
    // Built-in profiles
    void create_builtin_profiles();
    
    // PGO workflow
    struct PGOWorkflow {
        std::string profile_name;
        std::string source_dir;
        std::string build_dir;
        std::string benchmark_executable;
        std::vector<std::string> benchmark_args;
        std::string profile_output_dir;
    };
    
    bool generate_pgo_profile(const PGOWorkflow& workflow);
    bool build_with_pgo(const PGOWorkflow& workflow);
    
    // BOLT workflow
    struct BOLTWorkflow {
        std::string binary_path;
        std::string perf_data_path;
        std::string optimized_binary_path;
        std::vector<std::string> bolt_flags;
    };
    
    bool optimize_with_bolt(const BOLTWorkflow& workflow);
    
    // Compiler detection and capabilities
    enum class CompilerType {
        GCC,
        CLANG,
        ICC,        // Intel C++ Compiler
        MSVC,       // Microsoft Visual C++
        UNKNOWN
    };
    
    struct CompilerInfo {
        CompilerType type;
        std::string version;
        std::vector<std::string> supported_flags;
        bool supports_pgo;
        bool supports_lto;
        bool supports_thinlto;
        bool supports_bolt;
    };
    
    CompilerInfo detect_compiler() const;
    
    // Build system integration
    std::string generate_cmake_flags(const OptimizationProfile& profile) const;
    std::string generate_makefile_flags(const OptimizationProfile& profile) const;
    std::string generate_ninja_flags(const OptimizationProfile& profile) const;
    
    // Performance measurement
    struct BenchmarkResult {
        std::string profile_name;
        std::chrono::nanoseconds execution_time;
        size_t binary_size;
        double performance_score;
        std::unordered_map<std::string, double> metrics;
    };
    
    BenchmarkResult benchmark_profile(const OptimizationProfile& profile,
                                    const std::string& benchmark_command,
                                    size_t iterations = 10) const;
    
    // Auto-tuning
    struct AutoTuneConfig {
        std::vector<OptimizationLevel> levels_to_test;
        std::vector<TargetArchitecture> architectures_to_test;
        std::vector<std::vector<std::string>> flag_combinations;
        std::string benchmark_command;
        size_t benchmark_iterations = 5;
        std::string output_profile_name;
    };
    
    OptimizationProfile auto_tune(const AutoTuneConfig& config);
    
    // Profile validation
    bool validate_profile(const OptimizationProfile& profile) const;
    std::vector<std::string> get_validation_errors(const OptimizationProfile& profile) const;

private:
    std::unordered_map<std::string, OptimizationProfile> profiles_;
    CompilerInfo compiler_info_;
    
    // Helper methods
    std::string get_optimization_flag(OptimizationLevel level) const;
    std::string get_architecture_flag(TargetArchitecture arch) const;
    std::vector<std::string> get_lto_flags() const;
    std::vector<std::string> get_pgo_generation_flags() const;
    std::vector<std::string> get_pgo_use_flags(const std::string& profile_path) const;
    
    bool execute_command(const std::string& command, std::string& output) const;
    bool file_exists(const std::string& path) const;
    size_t get_file_size(const std::string& path) const;
};

/**
 * Profile-guided optimization helper
 */
class PGOHelper {
public:
    struct TrainingWorkload {
        std::string name;
        std::function<void()> workload;
        double weight = 1.0;  // Relative importance
    };
    
    PGOHelper(const std::string& profile_dir);
    ~PGOHelper();
    
    // Training workload management
    void add_training_workload(const TrainingWorkload& workload);
    void run_training_workloads();
    
    // Profile merging (for multiple training runs)
    bool merge_profiles(const std::vector<std::string>& profile_paths,
                       const std::string& output_path);
    
    // Profile analysis
    struct ProfileStats {
        size_t total_functions;
        size_t profiled_functions;
        double coverage_percentage;
        std::vector<std::pair<std::string, uint64_t>> hot_functions;
    };
    
    ProfileStats analyze_profile(const std::string& profile_path) const;
    
    // Instrumentation control
    void start_profiling();
    void stop_profiling();
    void reset_counters();

private:
    std::string profile_dir_;
    std::vector<TrainingWorkload> workloads_;
    bool profiling_active_;
};

/**
 * BOLT (Binary Optimization and Layout Tool) helper
 */
class BOLTHelper {
public:
    struct BOLTConfig {
        std::string bolt_binary_path = "llvm-bolt";
        std::vector<std::string> optimization_flags = {
            "-reorder-blocks=ext-tsp",
            "-reorder-functions=hfsort+",
            "-split-functions",
            "-split-all-cold",
            "-dyno-stats"
        };
        bool enable_instrumentation = true;
        bool enable_sampling = true;
    };
    
    BOLTHelper(const BOLTConfig& config = BOLTConfig{});
    ~BOLTHelper();
    
    // Profile collection
    bool collect_perf_profile(const std::string& binary_path,
                             const std::string& workload_command,
                             const std::string& output_profile);
    
    // Binary optimization
    bool optimize_binary(const std::string& input_binary,
                        const std::string& profile_path,
                        const std::string& output_binary);
    
    // Analysis
    struct BOLTStats {
        size_t original_size;
        size_t optimized_size;
        double size_reduction_percent;
        size_t functions_optimized;
        size_t basic_blocks_reordered;
    };
    
    BOLTStats analyze_optimization(const std::string& original_binary,
                                  const std::string& optimized_binary) const;

private:
    BOLTConfig config_;
    
    bool check_bolt_availability() const;
    bool check_perf_availability() const;
};

/**
 * Compiler flag optimizer using genetic algorithm
 */
class CompilerFlagOptimizer {
public:
    struct FlagGene {
        std::string flag;
        std::vector<std::string> possible_values;
        std::string current_value;
        double mutation_rate = 0.1;
    };
    
    struct OptimizationTarget {
        std::string name;
        std::function<double(const std::vector<std::string>&)> fitness_function;
        double weight = 1.0;
    };
    
    struct GAConfig {
        size_t population_size = 50;
        size_t generations = 100;
        double crossover_rate = 0.8;
        double mutation_rate = 0.1;
        size_t elite_size = 5;
        double convergence_threshold = 0.001;
    };
    
    CompilerFlagOptimizer(const GAConfig& config = GAConfig{});
    ~CompilerFlagOptimizer();
    
    // Gene pool management
    void add_flag_gene(const FlagGene& gene);
    void add_optimization_target(const OptimizationTarget& target);
    
    // Optimization process
    std::vector<std::string> optimize_flags();
    
    // Results analysis
    struct OptimizationResult {
        std::vector<std::string> best_flags;
        double best_fitness;
        std::vector<double> fitness_history;
        size_t generations_run;
        bool converged;
    };
    
    OptimizationResult get_last_result() const;

private:
    GAConfig config_;
    std::vector<FlagGene> genes_;
    std::vector<OptimizationTarget> targets_;
    OptimizationResult last_result_;
    
    struct Individual {
        std::vector<std::string> chromosome;
        double fitness = 0.0;
    };
    
    std::vector<Individual> create_initial_population();
    double evaluate_fitness(const Individual& individual);
    Individual crossover(const Individual& parent1, const Individual& parent2);
    void mutate(Individual& individual);
    std::vector<Individual> select_parents(const std::vector<Individual>& population);
};

} // namespace performance
} // namespace ultra_cpp