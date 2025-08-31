#include "chaos_testing_framework.hpp"
#include <iostream>
#include <string>

using namespace ultra::testing::chaos;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --experiment <name>    Experiment name: network_latency|memory_pressure|exception_injection|all\n";
    std::cout << "  --duration <seconds>   Test duration in seconds (default: 300)\n";
    std::cout << "  --probability <float>  Failure probability 0.0-1.0 (default: 0.1)\n";
    std::cout << "  --enable-destructive   Enable destructive tests (default: false)\n";
    std::cout << "  --output <file>        Output report file (default: chaos_test_report.json)\n";
    std::cout << "  --list-experiments     List available experiments\n";
    std::cout << "  --help                 Show this help message\n";
}

void list_experiments() {
    std::cout << "Available Chaos Experiments:\n";
    std::cout << "============================\n";
    std::cout << "network_latency      - Inject network latency\n";
    std::cout << "network_packet_loss  - Inject packet loss\n";
    std::cout << "memory_pressure      - Create memory pressure\n";
    std::cout << "cpu_stress          - Create CPU stress\n";
    std::cout << "exception_injection  - Inject exceptions\n";
    std::cout << "timeout_injection   - Inject timeouts\n";
    std::cout << "memory_leak         - Inject memory leaks\n";
    std::cout << "disk_io_failure     - Simulate disk I/O failures\n";
    std::cout << "all                 - Run all experiments\n";
}

int main(int argc, char* argv[]) {
    std::string experiment_name = "network_latency";
    int duration = 300;
    double probability = 0.1;
    bool enable_destructive = false;
    std::string output_file = "chaos_test_report.json";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--list-experiments") {
            list_experiments();
            return 0;
        } else if (arg == "--experiment" && i + 1 < argc) {
            experiment_name = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stoi(argv[++i]);
        } else if (arg == "--probability" && i + 1 < argc) {
            probability = std::stod(argv[++i]);
        } else if (arg == "--enable-destructive") {
            enable_destructive = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!enable_destructive) {
        std::cout << "WARNING: Destructive tests are disabled.\n";
        std::cout << "Use --enable-destructive to run actual chaos experiments.\n";
        std::cout << "Running in simulation mode...\n\n";
    }
    
    try {
        ChaosTestingEngine engine;
        
        // Set global configuration
        engine.set_global_probability(probability);
        
        // Set safety limits
        std::map<std::string, double> safety_limits = {
            {"cpu_usage", 95.0},
            {"memory_usage", 16384.0}, // 16GB
            {"error_rate", 0.8}         // 80% max error rate
        };
        engine.set_safety_limits(safety_limits);
        engine.enable_auto_recovery(true);
        
        // Register experiments
        std::vector<ChaosExperiment> experiments;
        
        if (experiment_name == "network_latency" || experiment_name == "all") {
            auto exp = ChaosExperimentLibrary::network_latency_experiment();
            exp.probability = probability;
            experiments.push_back(exp);
            engine.register_experiment(exp);
        }
        
        if (experiment_name == "memory_pressure" || experiment_name == "all") {
            auto exp = ChaosExperimentLibrary::memory_pressure_experiment();
            exp.probability = probability;
            experiments.push_back(exp);
            engine.register_experiment(exp);
        }
        
        if (experiment_name == "exception_injection" || experiment_name == "all") {
            auto exp = ChaosExperimentLibrary::exception_injection_experiment();
            exp.probability = probability;
            experiments.push_back(exp);
            engine.register_experiment(exp);
        }
        
        if (experiments.empty()) {
            std::cerr << "Unknown experiment: " << experiment_name << std::endl;
            std::cout << "Use --list-experiments to see available experiments.\n";
            return 1;
        }
        
        std::cout << "Starting chaos testing...\n";
        std::cout << "Experiments: ";
        for (const auto& exp : experiments) {
            std::cout << exp.name << " ";
        }
        std::cout << "\n";
        std::cout << "Duration: " << duration << " seconds\n";
        std::cout << "Probability: " << probability << "\n";
        std::cout << "Destructive mode: " << (enable_destructive ? "ENABLED" : "DISABLED") << "\n\n";
        
        if (experiment_name == "all") {
            // Run test suite
            ChaosTestingEngine::ChaosTestSuite suite;
            suite.name = "comprehensive_chaos_test";
            for (const auto& exp : experiments) {
                suite.experiment_names.push_back(exp.name);
            }
            suite.total_duration = std::chrono::seconds(duration);
            suite.stop_on_failure = false;
            
            auto results = engine.run_test_suite(suite);
            
            std::cout << "\nChaos Test Suite Results:\n";
            std::cout << "=========================\n";
            std::cout << "Suite: " << results.suite_name << "\n";
            std::cout << "Total Duration: " << results.total_duration.count() << "ms\n";
            std::cout << "Passed Experiments: " << results.passed_experiments << "\n";
            std::cout << "Failed Experiments: " << results.failed_experiments << "\n";
            std::cout << "Overall Success: " << (results.overall_success ? "YES" : "NO") << "\n\n";
            
            for (const auto& exp_result : results.experiment_results) {
                std::cout << "Experiment: " << exp_result.experiment_name << "\n";
                std::cout << "  Total Injections: " << exp_result.total_injections << "\n";
                std::cout << "  Successful: " << exp_result.successful_injections << "\n";
                std::cout << "  Failed: " << exp_result.failed_injections << "\n";
                std::cout << "  Passed: " << (exp_result.experiment_passed ? "YES" : "NO") << "\n";
                if (!exp_result.failure_reason.empty()) {
                    std::cout << "  Failure Reason: " << exp_result.failure_reason << "\n";
                }
                std::cout << "\n";
            }
            
            // Generate report
            ChaosTestingUtils::generate_chaos_report(results.experiment_results, output_file);
            std::cout << "Report generated: " << output_file << "\n";
            
            return results.overall_success ? 0 : 1;
            
        } else {
            // Run single experiment
            auto result = engine.run_experiment(experiment_name, std::chrono::seconds(duration));
            
            std::cout << "\nChaos Experiment Results:\n";
            std::cout << "=========================\n";
            std::cout << "Experiment: " << result.experiment_name << "\n";
            std::cout << "Total Injections: " << result.total_injections << "\n";
            std::cout << "Successful: " << result.successful_injections << "\n";
            std::cout << "Failed: " << result.failed_injections << "\n";
            std::cout << "System Recoveries: " << result.system_recoveries << "\n";
            std::cout << "Passed: " << (result.experiment_passed ? "YES" : "NO") << "\n";
            
            if (!result.failure_reason.empty()) {
                std::cout << "Failure Reason: " << result.failure_reason << "\n";
            }
            
            std::cout << "\nSystem Metrics:\n";
            std::cout << "  Avg Response Time: " << result.system_metrics.avg_response_time_ms << "ms\n";
            std::cout << "  Error Rate: " << (result.system_metrics.error_rate * 100) << "%\n";
            std::cout << "  Throughput Degradation: " << (result.system_metrics.throughput_degradation * 100) << "%\n";
            std::cout << "  Peak Memory Usage: " << result.system_metrics.memory_usage_peak_mb << "MB\n";
            std::cout << "  Peak CPU Usage: " << result.system_metrics.cpu_usage_peak_percent << "%\n";
            
            // Generate report
            std::vector<ChaosExperimentResult> results_vec = {result};
            ChaosTestingUtils::generate_chaos_report(results_vec, output_file);
            std::cout << "\nReport generated: " << output_file << "\n";
            
            return result.experiment_passed ? 0 : 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Chaos test error: " << e.what() << std::endl;
        return 1;
    }
}