#include "framework/test_framework.hpp"
#include "load/load_test_suite.hpp"
#include "chaos/chaos_testing_framework.hpp"
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

using namespace ultra::testing;

class UltraTestRunner {
public:
    enum class TestType {
        UNIT_TESTS,
        BENCHMARKS,
        LOAD_TESTS,
        CHAOS_TESTS,
        ALL_TESTS
    };
    
    struct Config {
        TestType test_type = TestType::ALL_TESTS;
        std::string output_dir = "./test_results";
        bool generate_reports = true;
        bool verbose = false;
        
        // Load test specific
        std::string target_endpoint = "http://localhost:8080";
        std::chrono::seconds load_test_duration{60};
        
        // Chaos test specific
        std::chrono::seconds chaos_test_duration{300};
        bool enable_destructive_tests = false;
    };
    
    explicit UltraTestRunner(const Config& config) : config_(config) {}
    
    int run() {
        std::cout << "Ultra Low-Latency System Test Runner\n";
        std::cout << "====================================\n\n";
        
        int result = 0;
        
        switch (config_.test_type) {
            case TestType::UNIT_TESTS:
                result = run_unit_tests();
                break;
            case TestType::BENCHMARKS:
                result = run_benchmarks();
                break;
            case TestType::LOAD_TESTS:
                result = run_load_tests();
                break;
            case TestType::CHAOS_TESTS:
                result = run_chaos_tests();
                break;
            case TestType::ALL_TESTS:
                result = run_all_tests();
                break;
        }
        
        if (config_.generate_reports) {
            generate_summary_report();
        }
        
        return result;
    }

private:
    Config config_;
    std::vector<std::string> test_results_;
    
    int run_unit_tests() {
        std::cout << "Running Unit Tests...\n";
        
        // Initialize Google Test
        int argc = 1;
        char* argv[] = {const_cast<char*>("test_runner")};
        ::testing::InitGoogleTest(&argc, argv);
        
        // Configure test output
        if (config_.verbose) {
            ::testing::FLAGS_gtest_print_time = true;
        }
        
        // Run tests
        int result = RUN_ALL_TESTS();
        
        test_results_.push_back("Unit Tests: " + 
                               (result == 0 ? "PASSED" : "FAILED"));
        
        return result;
    }
    
    int run_benchmarks() {
        std::cout << "Running Performance Benchmarks...\n";
        
        // Initialize Google Benchmark
        int argc = 1;
        char* argv[] = {const_cast<char*>("test_runner")};
        ::benchmark::Initialize(&argc, argv);
        
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }
        
        // Configure benchmark output
        std::string output_file = config_.output_dir + "/benchmark_results.json";
        ::benchmark::RegisterBenchmark("DummyBenchmark", [](::benchmark::State& state) {
            for (auto _ : state) {
                // Dummy benchmark to ensure framework works
                volatile int x = 42;
                ::benchmark::DoNotOptimize(x);
            }
        });
        
        ::benchmark::RunSpecifiedBenchmarks();
        ::benchmark::Shutdown();
        
        test_results_.push_back("Benchmarks: COMPLETED");
        return 0;
    }
    
    int run_load_tests() {
        std::cout << "Running Load Tests...\n";
        
        using namespace ultra::testing::load;
        
        try {
            // Create load test suite
            LoadTestSuite suite;
            
            // Add basic load tests
            suite.add_test_case("constant_load", 
                LoadTestScenarios::constant_load(1000, config_.load_test_duration));
            
            suite.add_test_case("ramp_up_test",
                LoadTestScenarios::ramp_up_test(100, 2000, config_.load_test_duration));
            
            suite.add_test_case("spike_test",
                LoadTestScenarios::spike_test(500, 5000, config_.load_test_duration));
            
            // Run the test suite
            auto results = suite.run_suite();
            
            // Generate reports
            if (config_.generate_reports) {
                suite.generate_html_report(results, config_.output_dir + "/load_test_report.html");
                suite.generate_json_report(results, config_.output_dir + "/load_test_results.json");
            }
            
            std::cout << "Load Tests Completed: " << results.passed_tests 
                      << " passed, " << results.failed_tests << " failed\n";
            
            test_results_.push_back("Load Tests: " + 
                                   (results.overall_success ? "PASSED" : "FAILED"));
            
            return results.overall_success ? 0 : 1;
            
        } catch (const std::exception& e) {
            std::cerr << "Load test error: " << e.what() << std::endl;
            test_results_.push_back("Load Tests: ERROR");
            return 1;
        }
    }
    
    int run_chaos_tests() {
        std::cout << "Running Chaos Tests...\n";
        
        if (!config_.enable_destructive_tests) {
            std::cout << "Destructive tests disabled. Use --enable-destructive to run chaos tests.\n";
            test_results_.push_back("Chaos Tests: SKIPPED");
            return 0;
        }
        
        using namespace ultra::testing::chaos;
        
        try {
            ChaosTestingEngine engine;
            
            // Register basic chaos experiments
            engine.register_experiment(ChaosExperimentLibrary::network_latency_experiment());
            engine.register_experiment(ChaosExperimentLibrary::memory_pressure_experiment());
            engine.register_experiment(ChaosExperimentLibrary::exception_injection_experiment());
            
            // Set safety limits
            std::map<std::string, double> safety_limits = {
                {"cpu_usage", 90.0},
                {"memory_usage", 8192.0}, // 8GB
                {"error_rate", 0.5}        // 50% max error rate
            };
            engine.set_safety_limits(safety_limits);
            engine.enable_auto_recovery(true);
            
            // Create test suite
            ChaosTestingEngine::ChaosTestSuite suite;
            suite.name = "basic_chaos_tests";
            suite.experiment_names = {"network_latency", "memory_pressure", "exception_injection"};
            suite.total_duration = config_.chaos_test_duration;
            suite.stop_on_failure = false;
            
            // Run chaos tests
            auto results = suite_results = engine.run_test_suite(suite);
            
            std::cout << "Chaos Tests Completed: " << results.passed_experiments 
                      << " passed, " << results.failed_experiments << " failed\n";
            
            test_results_.push_back("Chaos Tests: " + 
                                   (results.overall_success ? "PASSED" : "FAILED"));
            
            return results.overall_success ? 0 : 1;
            
        } catch (const std::exception& e) {
            std::cerr << "Chaos test error: " << e.what() << std::endl;
            test_results_.push_back("Chaos Tests: ERROR");
            return 1;
        }
    }
    
    int run_all_tests() {
        std::cout << "Running All Test Suites...\n\n";
        
        int overall_result = 0;
        
        // Run unit tests first
        std::cout << "=== Unit Tests ===\n";
        int unit_result = run_unit_tests();
        if (unit_result != 0) overall_result = 1;
        
        std::cout << "\n=== Benchmarks ===\n";
        int bench_result = run_benchmarks();
        if (bench_result != 0) overall_result = 1;
        
        std::cout << "\n=== Load Tests ===\n";
        int load_result = run_load_tests();
        if (load_result != 0) overall_result = 1;
        
        if (config_.enable_destructive_tests) {
            std::cout << "\n=== Chaos Tests ===\n";
            int chaos_result = run_chaos_tests();
            if (chaos_result != 0) overall_result = 1;
        }
        
        return overall_result;
    }
    
    void generate_summary_report() {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "TEST SUMMARY REPORT\n";
        std::cout << std::string(50, '=') << "\n";
        
        for (const auto& result : test_results_) {
            std::cout << result << "\n";
        }
        
        // Generate JSON summary
        std::ofstream summary_file(config_.output_dir + "/test_summary.json");
        if (summary_file.is_open()) {
            summary_file << "{\n";
            summary_file << "  \"test_results\": [\n";
            
            for (size_t i = 0; i < test_results_.size(); ++i) {
                summary_file << "    \"" << test_results_[i] << "\"";
                if (i < test_results_.size() - 1) {
                    summary_file << ",";
                }
                summary_file << "\n";
            }
            
            summary_file << "  ],\n";
            summary_file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
            summary_file << "  \"config\": {\n";
            summary_file << "    \"output_dir\": \"" << config_.output_dir << "\",\n";
            summary_file << "    \"target_endpoint\": \"" << config_.target_endpoint << "\",\n";
            summary_file << "    \"destructive_tests_enabled\": " 
                         << (config_.enable_destructive_tests ? "true" : "false") << "\n";
            summary_file << "  }\n";
            summary_file << "}\n";
        }
        
        std::cout << "\nReports generated in: " << config_.output_dir << "\n";
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --test-type <type>     Test type: unit|benchmark|load|chaos|all (default: all)\n";
    std::cout << "  --output-dir <dir>     Output directory for reports (default: ./test_results)\n";
    std::cout << "  --target <endpoint>    Target endpoint for load tests (default: http://localhost:8080)\n";
    std::cout << "  --duration <seconds>   Test duration in seconds (default: 60)\n";
    std::cout << "  --enable-destructive   Enable destructive chaos tests\n";
    std::cout << "  --verbose              Enable verbose output\n";
    std::cout << "  --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
    UltraTestRunner::Config config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--test-type" && i + 1 < argc) {
            std::string type = argv[++i];
            if (type == "unit") {
                config.test_type = UltraTestRunner::TestType::UNIT_TESTS;
            } else if (type == "benchmark") {
                config.test_type = UltraTestRunner::TestType::BENCHMARKS;
            } else if (type == "load") {
                config.test_type = UltraTestRunner::TestType::LOAD_TESTS;
            } else if (type == "chaos") {
                config.test_type = UltraTestRunner::TestType::CHAOS_TESTS;
            } else if (type == "all") {
                config.test_type = UltraTestRunner::TestType::ALL_TESTS;
            } else {
                std::cerr << "Invalid test type: " << type << std::endl;
                return 1;
            }
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            config.target_endpoint = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            int duration = std::stoi(argv[++i]);
            config.load_test_duration = std::chrono::seconds(duration);
            config.chaos_test_duration = std::chrono::seconds(duration * 5); // 5x for chaos
        } else if (arg == "--enable-destructive") {
            config.enable_destructive_tests = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Create output directory
    std::system(("mkdir -p " + config.output_dir).c_str());
    
    // Run tests
    UltraTestRunner runner(config);
    return runner.run();
}