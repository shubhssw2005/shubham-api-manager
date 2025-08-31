#include "load_test_suite.hpp"
#include <iostream>
#include <string>

using namespace ultra::testing::load;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --test-type <type>     Test type: constant|ramp|spike|burst|stress|realistic (default: constant)\n";
    std::cout << "  --target <endpoint>    Target endpoint (default: http://localhost:8080)\n";
    std::cout << "  --rps <number>         Requests per second (default: 1000)\n";
    std::cout << "  --peak-rps <number>    Peak requests per second (default: 5000)\n";
    std::cout << "  --duration <seconds>   Test duration in seconds (default: 60)\n";
    std::cout << "  --threads <number>     Number of worker threads (default: 4)\n";
    std::cout << "  --output <file>        Output report file (default: load_test_report.html)\n";
    std::cout << "  --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string test_type = "constant";
    std::string target = "http://localhost:8080";
    size_t rps = 1000;
    size_t peak_rps = 5000;
    int duration = 60;
    size_t threads = 4;
    std::string output_file = "load_test_report.html";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--test-type" && i + 1 < argc) {
            test_type = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            target = argv[++i];
        } else if (arg == "--rps" && i + 1 < argc) {
            rps = std::stoul(argv[++i]);
        } else if (arg == "--peak-rps" && i + 1 < argc) {
            peak_rps = std::stoul(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoul(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        LoadTestConfig config;
        config.target_endpoint = target;
        config.thread_count = threads;
        config.duration = std::chrono::seconds(duration);
        
        // Configure test type
        if (test_type == "constant") {
            config = LoadTestScenarios::constant_load(rps, std::chrono::seconds(duration));
        } else if (test_type == "ramp") {
            config = LoadTestScenarios::ramp_up_test(rps, peak_rps, std::chrono::seconds(duration));
        } else if (test_type == "spike") {
            config = LoadTestScenarios::spike_test(rps, peak_rps, std::chrono::seconds(duration));
        } else if (test_type == "burst") {
            config = LoadTestScenarios::burst_test(rps, peak_rps, 
                                                 std::chrono::seconds(10),
                                                 std::chrono::seconds(2),
                                                 std::chrono::seconds(duration));
        } else if (test_type == "stress") {
            config = LoadTestScenarios::stress_test(target);
        } else if (test_type == "realistic") {
            config = LoadTestScenarios::realistic_web_traffic(target);
        } else {
            std::cerr << "Invalid test type: " << test_type << std::endl;
            return 1;
        }
        
        config.target_endpoint = target;
        config.thread_count = threads;
        
        std::cout << "Starting load test...\n";
        std::cout << "Test type: " << test_type << "\n";
        std::cout << "Target: " << target << "\n";
        std::cout << "RPS: " << rps << " (peak: " << peak_rps << ")\n";
        std::cout << "Duration: " << duration << " seconds\n";
        std::cout << "Threads: " << threads << "\n\n";
        
        // Run the load test
        LoadTestRunner runner(config);
        
        // Set up progress callback
        runner.set_progress_callback([](double progress, const LoadTestRunner::RealTimeStats& stats) {
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                      << (progress * 100) << "% | "
                      << "RPS: " << std::setprecision(0) << stats.current_rps << " | "
                      << "Error Rate: " << std::setprecision(2) << (stats.current_error_rate * 100) << "% | "
                      << "P99 Latency: " << stats.current_p99_latency_ms << "ms" << std::flush;
        });
        
        auto results = runner.run();
        
        std::cout << "\n\nLoad Test Results:\n";
        std::cout << "==================\n";
        std::cout << "Total Requests: " << results.total_requests << "\n";
        std::cout << "Successful: " << results.successful_requests << "\n";
        std::cout << "Failed: " << results.failed_requests << "\n";
        std::cout << "Timeouts: " << results.timeout_requests << "\n";
        std::cout << "Actual RPS: " << std::fixed << std::setprecision(2) << results.throughput_stats.actual_rps << "\n";
        std::cout << "Error Rate: " << std::setprecision(2) << (results.error_stats.error_rate * 100) << "%\n";
        std::cout << "Latency P50: " << results.latency_stats.p50_ms << "ms\n";
        std::cout << "Latency P95: " << results.latency_stats.p95_ms << "ms\n";
        std::cout << "Latency P99: " << results.latency_stats.p99_ms << "ms\n";
        std::cout << "SLA Passed: " << (results.passed_sla ? "YES" : "NO") << "\n";
        
        // Generate report
        LoadTestSuite suite;
        LoadTestSuite::SuiteResults suite_results;
        suite_results.test_results.push_back({test_type, results});
        suite_results.passed_tests = results.passed_sla ? 1 : 0;
        suite_results.failed_tests = results.passed_sla ? 0 : 1;
        suite_results.overall_success = results.passed_sla;
        
        suite.generate_html_report(suite_results, output_file);
        std::cout << "\nReport generated: " << output_file << "\n";
        
        return results.passed_sla ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Load test error: " << e.what() << std::endl;
        return 1;
    }
}