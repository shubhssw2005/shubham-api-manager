#include <gtest/gtest.h>
#include "../../include/stream-processor/stream_processor.hpp"
#include "../../include/stream-processor/event_ingestion.hpp"
#include "../../include/stream-processor/windowed_aggregator.hpp"
#include "../../include/stream-processor/anomaly_detector.hpp"
#include <chrono>
#include <thread>

using namespace ultra_cpp::stream;

class StreamProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.batch_size = 4;
        config_.worker_threads = 2;
        config_.enable_simd = true;
        
        processor_ = std::make_unique<StreamProcessor>(config_);
    }
    
    void TearDown() override {
        if (processor_) {
            processor_->stop();
        }
    }
    
    StreamConfig config_;
    std::unique_ptr<StreamProcessor> processor_;
};

TEST_F(StreamProcessorTest, BasicEventProcessing) {
    std::atomic<int> events_received{0};
    
    // Subscribe to events
    processor_->subscribe(1, [&events_received](const StreamEvent& event) {
        events_received.fetch_add(1);
        EXPECT_EQ(event.event_type, 1u);
    });
    
    processor_->start();
    
    // Publish some events
    for (int i = 0; i < 10; ++i) {
        std::string data = "test_data_" + std::to_string(i);
        EXPECT_TRUE(processor_->publish(1, 1, 1, data.c_str(), data.size()));
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(events_received.load(), 10);
    
    auto metrics = processor_->get_metrics();
    EXPECT_GE(metrics.events_processed.load(), 10u);
}

TEST_F(StreamProcessorTest, BatchProcessing) {
    std::atomic<int> batches_received{0};
    std::atomic<int> total_events{0};
    
    // Subscribe to batch processing
    processor_->subscribe_batch(1, [&](const std::vector<const StreamEvent*>& events) {
        batches_received.fetch_add(1);
        total_events.fetch_add(events.size());
    });
    
    processor_->start();
    
    // Publish events to trigger batching
    for (int i = 0; i < 20; ++i) {
        std::string data = "batch_test_" + std::to_string(i);
        processor_->publish(1, 1, 1, data.c_str(), data.size());
    }
    
    // Wait for batch processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_GT(batches_received.load(), 0);
    EXPECT_EQ(total_events.load(), 20);
}

class EventIngestionTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.batch_size = 4;
        config_.enable_batching = true;
        
        ingestion_ = std::make_unique<EventIngestion>(config_);
    }
    
    void TearDown() override {
        if (ingestion_) {
            ingestion_->stop();
        }
    }
    
    IngestionConfig config_;
    std::unique_ptr<EventIngestion> ingestion_;
};

TEST_F(EventIngestionTest, BasicIngestion) {
    std::atomic<int> batches_processed{0};
    
    ingestion_->register_batch_callback([&batches_processed](const EventBatch& batch) {
        batches_processed.fetch_add(1);
        EXPECT_GT(batch.get_count(), 0u);
    });
    
    ingestion_->start();
    
    // Ingest events
    for (int i = 0; i < 10; ++i) {
        std::string data = "ingestion_test_" + std::to_string(i);
        EXPECT_TRUE(ingestion_->ingest_event(1, 1, 1, data.c_str(), data.size()));
    }
    
    // Force flush to ensure processing
    ingestion_->flush_batch();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    auto stats = ingestion_->get_stats();
    EXPECT_GE(stats.events_ingested.load(), 10u);
    EXPECT_GT(batches_processed.load(), 0);
}

class WindowedAggregatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.window_size = std::chrono::milliseconds(100);
        config_.slide_interval = std::chrono::milliseconds(50);
        config_.enable_simd = true;
        
        aggregator_ = std::make_unique<WindowedAggregator>(config_);
    }
    
    void TearDown() override {
        if (aggregator_) {
            aggregator_->stop();
        }
    }
    
    WindowConfig config_;
    std::unique_ptr<WindowedAggregator> aggregator_;
};

TEST_F(WindowedAggregatorTest, BasicAggregation) {
    std::atomic<int> results_received{0};
    
    aggregator_->add_aggregation(1, 1, AggregationType::AVERAGE, 
        [&results_received](const AggregationResult& result) {
            results_received.fetch_add(1);
            EXPECT_GT(result.count, 0u);
            EXPECT_GT(result.value, 0.0);
        });
    
    aggregator_->start();
    
    // Generate test events
    uint64_t base_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    for (int i = 0; i < 10; ++i) {
        StreamEvent event;
        event.timestamp_ns = base_time + i * 10000000ULL; // 10ms apart
        event.event_type = 1;
        event.tenant_id = 1;
        event.user_id = 1;
        event.data_size = 0;
        
        double value = 100.0 + i; // Increasing values
        aggregator_->process_event(event, value);
    }
    
    // Wait for window processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    aggregator_->flush_windows();
    
    EXPECT_GT(results_received.load(), 0);
}

class AnomalyDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.algorithm = AnomalyAlgorithm::Z_SCORE;
        config_.threshold = 2.0;
        config_.window_size = 10;
        config_.enable_simd = true;
        
        detector_ = std::make_unique<AnomalyDetector>(config_);
    }
    
    void TearDown() override {
        if (detector_) {
            detector_->stop();
        }
    }
    
    AnomalyConfig config_;
    std::unique_ptr<AnomalyDetector> detector_;
};

TEST_F(AnomalyDetectorTest, AnomalyDetection) {
    std::atomic<int> anomalies_detected{0};
    
    detector_->register_detector(1, 1, [&anomalies_detected](const AnomalyResult& result) {
        anomalies_detected.fetch_add(1);
        EXPECT_GT(result.anomaly_score, 0.0);
        EXPECT_EQ(result.event_type, 1u);
        EXPECT_EQ(result.tenant_id, 1u);
    });
    
    detector_->start();
    
    uint64_t base_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Generate normal data first
    for (int i = 0; i < 15; ++i) {
        StreamEvent event;
        event.timestamp_ns = base_time + i * 1000000ULL;
        event.event_type = 1;
        event.tenant_id = 1;
        event.user_id = 1;
        event.data_size = 0;
        
        double value = 100.0 + (i % 3); // Normal variation
        detector_->process_event(event, value);
    }
    
    // Generate anomalous data
    for (int i = 0; i < 3; ++i) {
        StreamEvent event;
        event.timestamp_ns = base_time + (15 + i) * 1000000ULL;
        event.event_type = 1;
        event.tenant_id = 1;
        event.user_id = 1;
        event.data_size = 0;
        
        double value = 200.0; // Anomalous value
        detector_->process_event(event, value);
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_GT(anomalies_detected.load(), 0);
}

// Performance benchmark test
TEST(StreamProcessorPerformance, HighThroughputTest) {
    StreamConfig config;
    config.batch_size = 256;
    config.worker_threads = 4;
    config.enable_simd = true;
    
    StreamProcessor processor(config);
    
    std::atomic<uint64_t> events_processed{0};
    processor.subscribe(1, [&events_processed](const StreamEvent& event) {
        events_processed.fetch_add(1);
    });
    
    processor.start();
    
    const int NUM_EVENTS = 100000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Publish events as fast as possible
    for (int i = 0; i < NUM_EVENTS; ++i) {
        std::string data = "perf_test_" + std::to_string(i);
        processor.publish(1, 1, 1, data.c_str(), data.size());
    }
    
    // Wait for all events to be processed
    while (events_processed.load() < NUM_EVENTS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double events_per_second = static_cast<double>(NUM_EVENTS) / (duration.count() / 1000000.0);
    
    std::cout << "Processed " << NUM_EVENTS << " events in " << duration.count() 
              << " microseconds (" << events_per_second << " events/sec)" << std::endl;
    
    // Expect at least 10K events per second (very conservative for this system)
    EXPECT_GT(events_per_second, 10000.0);
    
    auto metrics = processor.get_metrics();
    EXPECT_EQ(metrics.events_processed.load(), NUM_EVENTS);
    EXPECT_LT(metrics.get_average_latency_ns(), 1000000.0); // Less than 1ms average
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}