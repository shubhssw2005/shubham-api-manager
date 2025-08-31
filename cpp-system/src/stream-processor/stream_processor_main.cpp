#include "../../include/stream-processor/stream_processor.hpp"
#include "../../include/stream-processor/event_ingestion.hpp"
#include "../../include/stream-processor/windowed_aggregator.hpp"
#include "../../include/stream-processor/anomaly_detector.hpp"
#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>

using namespace ultra_cpp::stream;

// Global flag for graceful shutdown
std::atomic<bool> g_shutdown{false};

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down gracefully..." << std::endl;
    g_shutdown.store(true);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --help                Show this help message\n"
              << "  --port PORT          Set ingestion port (default: 8090)\n"
              << "  --threads N          Set number of worker threads (default: auto)\n"
              << "  --batch-size N       Set batch size (default: 256)\n"
              << "  --window-size MS     Set window size in milliseconds (default: 1000)\n"
              << "  --enable-simd        Enable SIMD acceleration (default: true)\n"
              << "  --enable-anomaly     Enable anomaly detection (default: true)\n"
              << "  --demo               Run demonstration mode\n"
              << std::endl;
}

struct Config {
    uint16_t port = 8090;
    size_t threads = 0; // 0 = auto-detect
    size_t batch_size = 256;
    uint32_t window_size_ms = 1000;
    bool enable_simd = true;
    bool enable_anomaly = true;
    bool demo_mode = false;
};

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--window-size" && i + 1 < argc) {
            config.window_size_ms = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--enable-simd") {
            config.enable_simd = true;
        } else if (arg == "--disable-simd") {
            config.enable_simd = false;
        } else if (arg == "--enable-anomaly") {
            config.enable_anomaly = true;
        } else if (arg == "--disable-anomaly") {
            config.enable_anomaly = false;
        } else if (arg == "--demo") {
            config.demo_mode = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

void run_demo() {
    std::cout << "Running Ultra Low-Latency Stream Processor Demo..." << std::endl;
    
    // Configure components
    StreamConfig stream_config;
    stream_config.batch_size = 128;
    stream_config.enable_simd = true;
    
    IngestionConfig ingestion_config;
    ingestion_config.batch_size = 128;
    ingestion_config.enable_batching = true;
    
    WindowConfig window_config;
    window_config.window_size = std::chrono::milliseconds(1000);
    window_config.slide_interval = std::chrono::milliseconds(100);
    window_config.enable_simd = true;
    
    AnomalyConfig anomaly_config;
    anomaly_config.algorithm = AnomalyAlgorithm::Z_SCORE;
    anomaly_config.threshold = 3.0;
    anomaly_config.enable_simd = true;
    
    // Create components
    StreamProcessor processor(stream_config);
    EventIngestion ingestion(ingestion_config);
    WindowedAggregator aggregator(window_config);
    AnomalyDetector detector(anomaly_config);
    
    // Set up event handlers
    processor.subscribe(1, [](const StreamEvent& event) {
        std::cout << "Processed event type " << event.event_type 
                  << " from tenant " << event.tenant_id << std::endl;
    });
    
    // Set up aggregation callbacks
    aggregator.add_aggregation(1, 1, AggregationType::AVERAGE, 
        [](const AggregationResult& result) {
            std::cout << "Average: " << result.value 
                      << " (count: " << result.count << ")" << std::endl;
        });
    
    aggregator.add_aggregation(1, 1, AggregationType::PERCENTILE_95,
        [](const AggregationResult& result) {
            std::cout << "P95: " << result.value << std::endl;
        });
    
    // Set up anomaly detection
    detector.register_detector(1, 1, [](const AnomalyResult& result) {
        std::cout << "ANOMALY: Score " << result.anomaly_score 
                  << ", Value " << result.actual_value 
                  << ", Expected " << result.expected_value << std::endl;
    });
    
    // Start all components
    std::cout << "Starting components..." << std::endl;
    processor.start();
    ingestion.start();
    aggregator.start();
    detector.start();
    
    // Generate demo data
    std::cout << "Generating demo events..." << std::endl;
    std::thread data_generator([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> normal_dist(100.0, 15.0);
        std::uniform_real_distribution<double> anomaly_dist(200.0, 300.0);
        
        int event_count = 0;
        while (!g_shutdown.load() && event_count < 10000) {
            // Generate mostly normal data with occasional anomalies
            double value = (event_count % 100 == 0) ? anomaly_dist(gen) : normal_dist(gen);
            
            // Create event data
            std::string data = "metric_value:" + std::to_string(value);
            
            // Ingest event
            ingestion.ingest_event(1, 1, 0, data.c_str(), data.size());
            
            // Process through aggregator and detector
            StreamEvent demo_event;
            demo_event.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            demo_event.event_type = 1;
            demo_event.tenant_id = 1;
            demo_event.user_id = 0;
            demo_event.data_size = static_cast<uint32_t>(data.size());
            demo_event.sequence_id = event_count;
            
            aggregator.process_event(demo_event, value);
            detector.process_event(demo_event, value);
            
            ++event_count;
            
            // Sleep to simulate realistic event rate
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    // Print statistics periodically
    std::thread stats_printer([&]() {
        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            auto ingestion_stats = ingestion.get_stats();
            auto stream_metrics = processor.get_metrics();
            
            std::cout << "\n=== Statistics ===" << std::endl;
            std::cout << "Ingested events: " << ingestion_stats.events_ingested.load() << std::endl;
            std::cout << "Processed events: " << stream_metrics.events_processed.load() << std::endl;
            std::cout << "Dropped events: " << ingestion_stats.events_dropped.load() << std::endl;
            std::cout << "Queue depth: " << ingestion_stats.queue_depth.load() << std::endl;
            std::cout << "Average latency: " << stream_metrics.get_average_latency_ns() << " ns" << std::endl;
            std::cout << "Events/sec: " << ingestion_stats.events_per_second.load() << std::endl;
            std::cout << "==================\n" << std::endl;
        }
    });
    
    // Wait for shutdown signal
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Stopping demo..." << std::endl;
    
    // Stop data generation
    if (data_generator.joinable()) {
        data_generator.join();
    }
    if (stats_printer.joinable()) {
        stats_printer.join();
    }
    
    // Stop components
    detector.stop();
    aggregator.stop();
    ingestion.stop();
    processor.stop();
    
    std::cout << "Demo completed." << std::endl;
}

void run_production(const Config& config) {
    std::cout << "Starting Ultra Low-Latency Stream Processor..." << std::endl;
    std::cout << "Port: " << config.port << std::endl;
    std::cout << "Threads: " << (config.threads == 0 ? std::thread::hardware_concurrency() : config.threads) << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Window size: " << config.window_size_ms << "ms" << std::endl;
    std::cout << "SIMD enabled: " << (config.enable_simd ? "yes" : "no") << std::endl;
    std::cout << "Anomaly detection: " << (config.enable_anomaly ? "yes" : "no") << std::endl;
    
    // Configure components
    StreamConfig stream_config;
    stream_config.worker_threads = config.threads == 0 ? std::thread::hardware_concurrency() : config.threads;
    stream_config.batch_size = config.batch_size;
    stream_config.enable_simd = config.enable_simd;
    
    IngestionConfig ingestion_config;
    ingestion_config.batch_size = config.batch_size;
    ingestion_config.worker_threads = stream_config.worker_threads;
    ingestion_config.enable_batching = true;
    
    WindowConfig window_config;
    window_config.window_size = std::chrono::milliseconds(config.window_size_ms);
    window_config.enable_simd = config.enable_simd;
    
    AnomalyConfig anomaly_config;
    anomaly_config.enable_simd = config.enable_simd;
    
    // Create components
    StreamProcessor processor(stream_config);
    EventIngestion ingestion(ingestion_config);
    WindowedAggregator aggregator(window_config);
    std::unique_ptr<AnomalyDetector> detector;
    
    if (config.enable_anomaly) {
        detector = std::make_unique<AnomalyDetector>(anomaly_config);
    }
    
    // Set up network ingestion
    NetworkEventIngestion::NetworkConfig network_config;
    network_config.port = config.port;
    NetworkEventIngestion network_ingestion(network_config, ingestion);
    
    // Start all components
    std::cout << "Starting components..." << std::endl;
    processor.start();
    ingestion.start();
    aggregator.start();
    if (detector) {
        detector->start();
    }
    network_ingestion.start();
    
    std::cout << "Stream processor is running. Press Ctrl+C to stop." << std::endl;
    
    // Statistics reporting thread
    std::thread stats_thread([&]() {
        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            auto ingestion_stats = ingestion.get_stats();
            auto stream_metrics = processor.get_metrics();
            auto network_stats = network_ingestion.get_network_stats();
            
            std::cout << "\n=== Performance Statistics ===" << std::endl;
            std::cout << "Network connections: " << network_stats.connections_active.load() << std::endl;
            std::cout << "Bytes received: " << network_stats.bytes_received.load() << std::endl;
            std::cout << "Events ingested: " << ingestion_stats.events_ingested.load() << std::endl;
            std::cout << "Events processed: " << stream_metrics.events_processed.load() << std::endl;
            std::cout << "Events dropped: " << ingestion_stats.events_dropped.load() << std::endl;
            std::cout << "Queue depth: " << ingestion_stats.queue_depth.load() << std::endl;
            std::cout << "Average latency: " << stream_metrics.get_average_latency_ns() << " ns" << std::endl;
            std::cout << "Throughput: " << ingestion_stats.events_per_second.load() << " events/sec" << std::endl;
            std::cout << "Bandwidth: " << ingestion_stats.bytes_per_second.load() << " bytes/sec" << std::endl;
            std::cout << "==============================\n" << std::endl;
        }
    });
    
    // Wait for shutdown signal
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Shutting down..." << std::endl;
    
    // Stop statistics thread
    if (stats_thread.joinable()) {
        stats_thread.join();
    }
    
    // Stop components in reverse order
    network_ingestion.stop();
    if (detector) {
        detector->stop();
    }
    aggregator.stop();
    ingestion.stop();
    processor.stop();
    
    std::cout << "Shutdown complete." << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        Config config = parse_args(argc, argv);
        
        if (config.demo_mode) {
            run_demo();
        } else {
            run_production(config);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}