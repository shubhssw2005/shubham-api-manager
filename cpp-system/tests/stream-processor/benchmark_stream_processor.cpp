#include "../../include/stream-processor/stream_processor.hpp"
#include "../../include/stream-processor/event_ingestion.hpp"
#include "../../include/stream-processor/windowed_aggregator.hpp"
#include "../../include/stream-processor/anomaly_detector.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace ultra_cpp::stream;

class StreamProcessorBenchmark {
public:
    struct BenchmarkConfig {
        size_t num_events = 1000000;
        size_t num_threads = std::thread::hardware_concurrency();
        size_t batch_size = 256;
        bool enable_simd = true;
        bool enable_aggregation = true;
        bool enable_anomaly_detection = true;
        double anomaly_rate = 0.01; // 1% anomalies
    };
    
    explicit StreamProcessorBenchmark(const BenchmarkConfig& config = {})
        : config_(config) {}
    
    void run_ingestion_benchmark() {
        std::cout << "\n=== Event Ingestion Benchmark ===" << std::endl;
        std::cout << "Events: " << config_.num_events << std::endl;
        std::cout << "Threads: " << config_.num_threads << std::endl;
        std::cout << "Batch size: " << config_.batch_size << std::endl;
        
        IngestionConfig ingestion_config;
        ingestion_config.batch_size = config_.batch_size;
        ingestion_config.worker_threads = config_.num_threads;
        ingestion_config.enable_batching = true;
        
        EventIngestion ingestion(ingestion_config);
        
        std::atomic<uint64_t> batches_processed{0};
        ingestion.register_batch_callback([&batches_processed](const EventBatch& batch) {
            batches_processed.fetch_add(1);
        });
        
        ingestion.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate events from multiple threads
        std::vector<std::thread> producer_threads;
        size_t events_per_thread = config_.num_events / config_.num_threads;
        
        for (size_t t = 0; t < config_.num_threads; ++t) {
            producer_threads.emplace_back([&ingestion, events_per_thread, t]() {
                std::random_device rd;
                std::mt19937 gen(rd() + t);
                std::uniform_real_distribution<double> dist(0.0, 1000.0);
                
                for (size_t i = 0; i < events_per_thread; ++i) {
                    double value = dist(gen);
                    std::string data = "benchmark_" + std::to_string(value);
                    
                    while (!ingestion.ingest_event(1, 1, static_cast<uint32_t>(t), 
                                                  data.c_str(), data.size())) {
                        std::this_thread::yield(); // Retry if queue is full
                    }
                }
            });
        }
        
        // Wait for all producers to finish
        for (auto& thread : producer_threads) {
            thread.join();
        }
        
        // Wait for all events to be processed
        ingestion.flush_batch();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        auto stats = ingestion.get_stats();
        
        double events_per_second = static_cast<double>(config_.num_events) / (duration.count() / 1000000.0);
        double avg_latency_ns = stats.get_average_latency_ns();
        
        std::cout << "Duration: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << events_per_second << " events/sec" << std::endl;
        std::cout << "Average latency: " << avg_latency_ns << " ns" << std::endl;
        std::cout << "Batches processed: " << batches_processed.load() << std::endl;
        std::cout << "Events ingested: " << stats.events_ingested.load() << std::endl;
        std::cout << "Events dropped: " << stats.events_dropped.load() << std::endl;
        
        ingestion.stop();
    }
    
    void run_stream_processing_benchmark() {
        std::cout << "\n=== Stream Processing Benchmark ===" << std::endl;
        
        StreamConfig stream_config;
        stream_config.batch_size = config_.batch_size;
        stream_config.worker_threads = config_.num_threads;
        stream_config.enable_simd = config_.enable_simd;
        
        StreamProcessor processor(stream_config);
        
        std::atomic<uint64_t> events_processed{0};
        processor.subscribe(1, [&events_processed](const StreamEvent& event) {
            events_processed.fetch_add(1);
        });
        
        processor.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Publish events
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        
        for (size_t i = 0; i < config_.num_events; ++i) {
            double value = dist(gen);
            std::string data = "stream_benchmark_" + std::to_string(value);
            
            while (!processor.publish(1, 1, 1, data.c_str(), data.size())) {
                std::this_thread::yield();
            }
        }
        
        // Wait for processing to complete
        while (events_processed.load() < config_.num_events) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        auto metrics = processor.get_metrics();
        
        double events_per_second = static_cast<double>(config_.num_events) / (duration.count() / 1000000.0);
        
        std::cout << "Duration: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << events_per_second << " events/sec" << std::endl;
        std::cout << "Average latency: " << metrics.get_average_latency_ns() << " ns" << std::endl;
        std::cout << "Average batch size: " << metrics.get_average_batch_size() << std::endl;
        std::cout << "Events processed: " << metrics.events_processed.load() << std::endl;
        std::cout << "Events dropped: " << metrics.events_dropped.load() << std::endl;
        
        processor.stop();
    }
    
    void run_windowed_aggregation_benchmark() {
        if (!config_.enable_aggregation) return;
        
        std::cout << "\n=== Windowed Aggregation Benchmark ===" << std::endl;
        
        WindowConfig window_config;
        window_config.window_size = std::chrono::milliseconds(1000);
        window_config.slide_interval = std::chrono::milliseconds(100);
        window_config.enable_simd = config_.enable_simd;
        
        WindowedAggregator aggregator(window_config);
        
        std::atomic<uint64_t> results_received{0};
        aggregator.add_aggregation(1, 1, AggregationType::AVERAGE,
            [&results_received](const AggregationResult& result) {
                results_received.fetch_add(1);
            });
        
        aggregator.add_aggregation(1, 1, AggregationType::PERCENTILE_95,
            [](const AggregationResult& result) {
                // Just consume the result
            });
        
        aggregator.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate events with timestamps
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(100.0, 15.0);
        
        uint64_t base_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        for (size_t i = 0; i < config_.num_events; ++i) {
            StreamEvent event;
            event.timestamp_ns = base_time + i * 1000000ULL; // 1ms apart
            event.event_type = 1;
            event.tenant_id = 1;
            event.user_id = 1;
            event.data_size = 0;
            
            double value = dist(gen);
            aggregator.process_event(event, value);
        }
        
        // Wait for window processing
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        aggregator.flush_windows();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double events_per_second = static_cast<double>(config_.num_events) / (duration.count() / 1000000.0);
        
        std::cout << "Duration: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << events_per_second << " events/sec" << std::endl;
        std::cout << "Aggregation results: " << results_received.load() << std::endl;
        
        aggregator.stop();
    }
    
    void run_anomaly_detection_benchmark() {
        if (!config_.enable_anomaly_detection) return;
        
        std::cout << "\n=== Anomaly Detection Benchmark ===" << std::endl;
        
        AnomalyConfig anomaly_config;
        anomaly_config.algorithm = AnomalyAlgorithm::Z_SCORE;
        anomaly_config.threshold = 3.0;
        anomaly_config.window_size = 1000;
        anomaly_config.enable_simd = config_.enable_simd;
        
        AnomalyDetector detector(anomaly_config);
        
        std::atomic<uint64_t> anomalies_detected{0};
        detector.register_detector(1, 1, [&anomalies_detected](const AnomalyResult& result) {
            anomalies_detected.fetch_add(1);
        });
        
        detector.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate events with some anomalies
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> normal_dist(100.0, 15.0);
        std::uniform_real_distribution<double> anomaly_dist(300.0, 400.0);
        std::uniform_real_distribution<double> anomaly_chance(0.0, 1.0);
        
        uint64_t base_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        for (size_t i = 0; i < config_.num_events; ++i) {
            StreamEvent event;
            event.timestamp_ns = base_time + i * 1000000ULL;
            event.event_type = 1;
            event.tenant_id = 1;
            event.user_id = 1;
            event.data_size = 0;
            
            // Generate anomaly based on configured rate
            double value = (anomaly_chance(gen) < config_.anomaly_rate) ? 
                          anomaly_dist(gen) : normal_dist(gen);
            
            detector.process_event(event, value);
        }
        
        // Wait for processing
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double events_per_second = static_cast<double>(config_.num_events) / (duration.count() / 1000000.0);
        
        std::cout << "Duration: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << events_per_second << " events/sec" << std::endl;
        std::cout << "Anomalies detected: " << anomalies_detected.load() << std::endl;
        std::cout << "Expected anomalies: " << static_cast<size_t>(config_.num_events * config_.anomaly_rate) << std::endl;
        
        detector.stop();
    }
    
    void run_full_pipeline_benchmark() {
        std::cout << "\n=== Full Pipeline Benchmark ===" << std::endl;
        std::cout << "Testing complete stream processing pipeline..." << std::endl;
        
        // Configure all components
        IngestionConfig ingestion_config;
        ingestion_config.batch_size = config_.batch_size;
        ingestion_config.worker_threads = config_.num_threads;
        
        StreamConfig stream_config;
        stream_config.batch_size = config_.batch_size;
        stream_config.worker_threads = config_.num_threads;
        stream_config.enable_simd = config_.enable_simd;
        
        WindowConfig window_config;
        window_config.window_size = std::chrono::milliseconds(1000);
        window_config.enable_simd = config_.enable_simd;
        
        AnomalyConfig anomaly_config;
        anomaly_config.enable_simd = config_.enable_simd;
        
        // Create components
        EventIngestion ingestion(ingestion_config);
        StreamProcessor processor(stream_config);
        WindowedAggregator aggregator(window_config);
        AnomalyDetector detector(anomaly_config);
        
        // Set up pipeline
        std::atomic<uint64_t> events_processed{0};
        std::atomic<uint64_t> aggregations_computed{0};
        std::atomic<uint64_t> anomalies_detected{0};
        
        ingestion.register_batch_callback([&](const EventBatch& batch) {
            auto events = batch.get_events();
            for (StreamEvent* event : events) {
                if (!event) continue;
                
                // Extract value from event data
                double value = static_cast<double>(event->timestamp_ns % 1000000) / 1000.0;
                
                // Process through aggregator and detector
                aggregator.process_event(*event, value);
                detector.process_event(*event, value);
                
                events_processed.fetch_add(1);
            }
        });
        
        aggregator.add_aggregation(1, 1, AggregationType::AVERAGE,
            [&aggregations_computed](const AggregationResult& result) {
                aggregations_computed.fetch_add(1);
            });
        
        detector.register_detector(1, 1, [&anomalies_detected](const AnomalyResult& result) {
            anomalies_detected.fetch_add(1);
        });
        
        // Start all components
        ingestion.start();
        processor.start();
        aggregator.start();
        detector.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate events
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(100.0, 15.0);
        
        for (size_t i = 0; i < config_.num_events; ++i) {
            double value = dist(gen);
            std::string data = "pipeline_test_" + std::to_string(value);
            
            while (!ingestion.ingest_event(1, 1, 1, data.c_str(), data.size())) {
                std::this_thread::yield();
            }
        }
        
        // Wait for processing to complete
        ingestion.flush_batch();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double events_per_second = static_cast<double>(config_.num_events) / (duration.count() / 1000000.0);
        
        std::cout << "Duration: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << events_per_second << " events/sec" << std::endl;
        std::cout << "Events processed: " << events_processed.load() << std::endl;
        std::cout << "Aggregations computed: " << aggregations_computed.load() << std::endl;
        std::cout << "Anomalies detected: " << anomalies_detected.load() << std::endl;
        
        // Stop components
        detector.stop();
        aggregator.stop();
        processor.stop();
        ingestion.stop();
    }
    
private:
    BenchmarkConfig config_;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --help                Show this help message\n"
              << "  --events N            Number of events to process (default: 1000000)\n"
              << "  --threads N           Number of worker threads (default: auto)\n"
              << "  --batch-size N        Batch size (default: 256)\n"
              << "  --disable-simd        Disable SIMD acceleration\n"
              << "  --disable-aggregation Disable aggregation benchmark\n"
              << "  --disable-anomaly     Disable anomaly detection benchmark\n"
              << "  --anomaly-rate R      Anomaly rate (0.0-1.0, default: 0.01)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    StreamProcessorBenchmark::BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--events" && i + 1 < argc) {
            config.num_events = std::stoull(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoull(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoull(argv[++i]);
        } else if (arg == "--disable-simd") {
            config.enable_simd = false;
        } else if (arg == "--disable-aggregation") {
            config.enable_aggregation = false;
        } else if (arg == "--disable-anomaly") {
            config.enable_anomaly_detection = false;
        } else if (arg == "--anomaly-rate" && i + 1 < argc) {
            config.anomaly_rate = std::stod(argv[++i]);
        }
    }
    
    std::cout << "Ultra Low-Latency Stream Processor Benchmark" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Events: " << config.num_events << std::endl;
    std::cout << "  Threads: " << config.num_threads << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  SIMD: " << (config.enable_simd ? "enabled" : "disabled") << std::endl;
    std::cout << "  Aggregation: " << (config.enable_aggregation ? "enabled" : "disabled") << std::endl;
    std::cout << "  Anomaly detection: " << (config.enable_anomaly_detection ? "enabled" : "disabled") << std::endl;
    std::cout << "  Anomaly rate: " << config.anomaly_rate << std::endl;
    
    StreamProcessorBenchmark benchmark(config);
    
    try {
        benchmark.run_ingestion_benchmark();
        benchmark.run_stream_processing_benchmark();
        benchmark.run_windowed_aggregation_benchmark();
        benchmark.run_anomaly_detection_benchmark();
        benchmark.run_full_pipeline_benchmark();
        
        std::cout << "\n=== Benchmark Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}