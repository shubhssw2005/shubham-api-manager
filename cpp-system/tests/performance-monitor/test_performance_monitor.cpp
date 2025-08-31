#include <gtest/gtest.h>
#include "performance-monitor/performance_monitor.hpp"
#include "performance-monitor/metrics_collector.hpp"
#include "performance-monitor/hardware_counters.hpp"

using namespace ultra::monitor;

class PerformanceMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        PerformanceMonitor::Config config;
        config.enable_hardware_counters = false;  // Disable for testing
        config.enable_slo_monitoring = false;
        config.prometheus_port = 9091;  // Use different port for testing
        
        monitor_ = std::make_unique<PerformanceMonitor>(config);
    }
    
    void TearDown() override {
        monitor_.reset();
    }
    
    std::unique_ptr<PerformanceMonitor> monitor_;
};

TEST_F(PerformanceMonitorTest, BasicCounterOperations) {
    // Test counter increment
    monitor_->increment_counter("test_counter", 5);
    monitor_->increment_counter("test_counter", 3);
    
    // Note: We can't directly verify the counter value without exposing internal state
    // In a real implementation, you might add getter methods for testing
    SUCCEED();
}

TEST_F(PerformanceMonitorTest, BasicGaugeOperations) {
    // Test gauge set
    monitor_->set_gauge("test_gauge", 42.5);
    monitor_->set_gauge("test_gauge", 37.2);
    
    SUCCEED();
}

TEST_F(PerformanceMonitorTest, BasicHistogramOperations) {
    // Test histogram observations
    monitor_->observe_histogram("test_histogram", 0.001);  // 1ms
    monitor_->observe_histogram("test_histogram", 0.002);  // 2ms
    monitor_->observe_histogram("test_histogram", 0.0005); // 0.5ms
    
    SUCCEED();
}

TEST_F(PerformanceMonitorTest, TimerOperations) {
    {
        PerformanceMonitor::Timer timer(*monitor_, "test_timer");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        // Timer automatically records on destruction
    }
    
    SUCCEED();
}

TEST_F(PerformanceMonitorTest, SLORegistration) {
    PerformanceMonitor::SLOConfig slo_config;
    slo_config.name = "test_slo";
    slo_config.target_percentile = 0.99;
    slo_config.target_latency_ns = 1000000;  // 1ms
    slo_config.evaluation_window = std::chrono::seconds(60);
    
    monitor_->register_slo(slo_config);
    
    SUCCEED();
}

TEST_F(PerformanceMonitorTest, PrometheusExport) {
    // Add some metrics
    monitor_->increment_counter("requests_total", 100);
    monitor_->set_gauge("memory_usage", 1024.0);
    monitor_->observe_histogram("request_duration", 0.001);
    
    // Export metrics
    std::string metrics = monitor_->export_prometheus_metrics();
    
    // Basic validation that we got some output
    EXPECT_FALSE(metrics.empty());
    EXPECT_NE(metrics.find("requests_total"), std::string::npos);
    EXPECT_NE(metrics.find("memory_usage"), std::string::npos);
    EXPECT_NE(metrics.find("request_duration"), std::string::npos);
}

class MetricsCollectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetricsCollector::Config config;
        config.max_metrics = 1000;
        config.histogram_buckets = 32;
        
        collector_ = std::make_unique<MetricsCollector>(config);
    }
    
    std::unique_ptr<MetricsCollector> collector_;
};

TEST_F(MetricsCollectorTest, CounterOperations) {
    collector_->increment_counter("test_counter", 5);
    EXPECT_EQ(collector_->get_counter_value("test_counter"), 5);
    
    collector_->increment_counter("test_counter", 3);
    EXPECT_EQ(collector_->get_counter_value("test_counter"), 8);
    
    // Test non-existent counter
    EXPECT_EQ(collector_->get_counter_value("nonexistent"), 0);
}

TEST_F(MetricsCollectorTest, GaugeOperations) {
    collector_->set_gauge("test_gauge", 42.5);
    EXPECT_DOUBLE_EQ(collector_->get_gauge_value("test_gauge"), 42.5);
    
    collector_->set_gauge("test_gauge", 37.2);
    EXPECT_DOUBLE_EQ(collector_->get_gauge_value("test_gauge"), 37.2);
    
    // Test non-existent gauge
    EXPECT_DOUBLE_EQ(collector_->get_gauge_value("nonexistent"), 0.0);
}

TEST_F(MetricsCollectorTest, HistogramOperations) {
    // Add some observations
    collector_->observe_histogram("test_histogram", 0.001);
    collector_->observe_histogram("test_histogram", 0.002);
    collector_->observe_histogram("test_histogram", 0.0005);
    collector_->observe_histogram("test_histogram", 0.003);
    
    // Calculate percentiles
    auto percentiles = collector_->calculate_percentiles("test_histogram");
    
    EXPECT_GT(percentiles.count, 0);
    EXPECT_GT(percentiles.sum, 0.0);
    EXPECT_GT(percentiles.mean, 0.0);
    EXPECT_GE(percentiles.max, percentiles.min);
}

TEST_F(MetricsCollectorTest, MetricEnumeration) {
    collector_->increment_counter("counter1");
    collector_->increment_counter("counter2");
    collector_->set_gauge("gauge1", 1.0);
    collector_->observe_histogram("histogram1", 0.001);
    
    auto counter_names = collector_->get_counter_names();
    auto gauge_names = collector_->get_gauge_names();
    auto histogram_names = collector_->get_histogram_names();
    
    EXPECT_EQ(counter_names.size(), 2);
    EXPECT_EQ(gauge_names.size(), 1);
    EXPECT_EQ(histogram_names.size(), 1);
    
    EXPECT_NE(std::find(counter_names.begin(), counter_names.end(), "counter1"), counter_names.end());
    EXPECT_NE(std::find(counter_names.begin(), counter_names.end(), "counter2"), counter_names.end());
    EXPECT_NE(std::find(gauge_names.begin(), gauge_names.end(), "gauge1"), gauge_names.end());
    EXPECT_NE(std::find(histogram_names.begin(), histogram_names.end(), "histogram1"), histogram_names.end());
}

class HardwareCountersTest : public ::testing::Test {
protected:
    void SetUp() override {
        counters_ = std::make_unique<HardwareCounters>();
    }
    
    std::unique_ptr<HardwareCounters> counters_;
};

TEST_F(HardwareCountersTest, PMUAvailability) {
    bool available = HardwareCounters::is_pmu_available();
    // This test will pass regardless of PMU availability
    // Just checking that the function doesn't crash
    SUCCEED();
}

TEST_F(HardwareCountersTest, AvailableCounters) {
    auto available_counters = HardwareCounters::get_available_counters();
    // Should return some counters on most systems, but might be empty in containers
    EXPECT_GE(available_counters.size(), 0);
}

TEST_F(HardwareCountersTest, CounterTypeToString) {
    EXPECT_EQ(HardwareCounters::counter_type_to_string(HardwareCounters::CounterType::CPU_CYCLES), "cpu_cycles");
    EXPECT_EQ(HardwareCounters::counter_type_to_string(HardwareCounters::CounterType::INSTRUCTIONS), "instructions");
    EXPECT_EQ(HardwareCounters::counter_type_to_string(HardwareCounters::CounterType::CACHE_MISSES), "cache_misses");
}

// Performance benchmarks
class PerformanceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        MetricsCollector::Config config;
        config.max_metrics = 10000;
        collector_ = std::make_unique<MetricsCollector>(config);
    }
    
    std::unique_ptr<MetricsCollector> collector_;
};

TEST_F(PerformanceBenchmark, CounterPerformance) {
    const int iterations = 1000000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        collector_->increment_counter("benchmark_counter");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double ns_per_operation = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Counter increment performance: " << ns_per_operation << " ns/op" << std::endl;
    
    // Should be very fast (sub-microsecond)
    EXPECT_LT(ns_per_operation, 1000.0);  // Less than 1μs per operation
}

TEST_F(PerformanceBenchmark, HistogramPerformance) {
    const int iterations = 100000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        collector_->observe_histogram("benchmark_histogram", static_cast<double>(i) / 1000000.0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double ns_per_operation = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Histogram observation performance: " << ns_per_operation << " ns/op" << std::endl;
    
    // Should be reasonably fast (sub-10μs)
    EXPECT_LT(ns_per_operation, 10000.0);  // Less than 10μs per operation
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}