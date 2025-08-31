#include "../../include/stream-processor/stream_processor.hpp"
#include "../../include/stream-processor/event_ingestion.hpp"
#include "../../include/stream-processor/windowed_aggregator.hpp"
#include "../../include/stream-processor/anomaly_detector.hpp"

namespace ultra_cpp {
namespace stream {

/**
 * Integrated event handler that combines ingestion, aggregation, and anomaly detection
 */
class IntegratedEventHandler {
public:
    struct Config {
        IngestionConfig ingestion;
        WindowConfig windowing;
        AnomalyConfig anomaly;
        bool enable_aggregation = true;
        bool enable_anomaly_detection = true;
    };
    
    explicit IntegratedEventHandler(const Config& config = {})
        : config_(config)
        , ingestion_(config_.ingestion)
        , aggregator_(config_.windowing)
        , detector_(config_.anomaly) {
        
        // Set up event flow pipeline
        setup_pipeline();
    }
    
    ~IntegratedEventHandler() {
        stop();
    }
    
    void start() {
        if (running_.exchange(true)) {
            return;
        }
        
        ingestion_.start();
        
        if (config_.enable_aggregation) {
            aggregator_.start();
        }
        
        if (config_.enable_anomaly_detection) {
            detector_.start();
        }
    }
    
    void stop() {
        running_.store(false);
        
        ingestion_.stop();
        
        if (config_.enable_aggregation) {
            aggregator_.stop();
        }
        
        if (config_.enable_anomaly_detection) {
            detector_.stop();
        }
    }
    
    // Convenience methods for event processing
    bool process_metric_event(uint32_t tenant_id, const std::string& metric_name, 
                             double value, uint64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        
        // Create event data
        std::string data = metric_name + ":" + std::to_string(value);
        
        return ingestion_.ingest_event(
            EVENT_TYPE_METRIC,
            tenant_id,
            0, // user_id not relevant for metrics
            data.c_str(),
            data.size()
        );
    }
    
    bool process_log_event(uint32_t tenant_id, uint32_t user_id, 
                          const std::string& log_message, uint64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        
        return ingestion_.ingest_event(
            EVENT_TYPE_LOG,
            tenant_id,
            user_id,
            log_message.c_str(),
            log_message.size()
        );
    }
    
    bool process_user_action(uint32_t tenant_id, uint32_t user_id, 
                            const std::string& action, uint64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        
        return ingestion_.ingest_event(
            EVENT_TYPE_USER_ACTION,
            tenant_id,
            user_id,
            action.c_str(),
            action.size()
        );
    }
    
    // Register callbacks for different event types
    void register_aggregation_callback(uint32_t event_type, uint32_t tenant_id,
                                     AggregationType agg_type, AggregationCallback callback) {
        if (config_.enable_aggregation) {
            aggregator_.add_aggregation(event_type, tenant_id, agg_type, std::move(callback));
        }
    }
    
    void register_anomaly_callback(uint32_t event_type, uint32_t tenant_id,
                                  AnomalyCallback callback) {
        if (config_.enable_anomaly_detection) {
            detector_.register_detector(event_type, tenant_id, std::move(callback));
        }
    }
    
    // Get statistics
    IngestionStats get_ingestion_stats() const {
        return ingestion_.get_stats();
    }
    
    std::vector<AggregationResult> get_aggregation_results() const {
        if (config_.enable_aggregation) {
            return aggregator_.get_current_results();
        }
        return {};
    }
    
private:
    // Event type constants
    static constexpr uint32_t EVENT_TYPE_METRIC = 1;
    static constexpr uint32_t EVENT_TYPE_LOG = 2;
    static constexpr uint32_t EVENT_TYPE_USER_ACTION = 3;
    static constexpr uint32_t EVENT_TYPE_SYSTEM = 4;
    
    Config config_;
    std::atomic<bool> running_{false};
    
    EventIngestion ingestion_;
    WindowedAggregator aggregator_;
    AnomalyDetector detector_;
    
    void setup_pipeline() {
        // Register batch callback to process events through aggregation and anomaly detection
        ingestion_.register_batch_callback([this](const EventBatch& batch) {
            process_batch(batch);
        });
    }
    
    void process_batch(const EventBatch& batch) {
        auto events = batch.get_events();
        if (events.empty()) {
            return;
        }
        
        // Process events for aggregation and anomaly detection
        for (StreamEvent* event : events) {
            if (!event) continue;
            
            // Extract numeric value from event data for processing
            double value = extract_numeric_value(*event);
            
            // Process through aggregator
            if (config_.enable_aggregation) {
                aggregator_.process_event(*event, value);
            }
            
            // Process through anomaly detector
            if (config_.enable_anomaly_detection) {
                detector_.process_event(*event, value);
            }
        }
    }
    
    double extract_numeric_value(const StreamEvent& event) {
        // Simple extraction logic - in practice would be more sophisticated
        if (event.data_size == 0) {
            return 0.0;
        }
        
        std::string data(event.data, event.data_size);
        
        // Look for numeric patterns in the data
        size_t colon_pos = data.find(':');
        if (colon_pos != std::string::npos && colon_pos + 1 < data.size()) {
            try {
                return std::stod(data.substr(colon_pos + 1));
            } catch (const std::exception&) {
                // Fallback to data size as a metric
                return static_cast<double>(event.data_size);
            }
        }
        
        // Default to timestamp-based value for demonstration
        return static_cast<double>(event.timestamp_ns % 1000000) / 1000.0;
    }
};

/**
 * Factory function to create a complete stream processing pipeline
 */
std::unique_ptr<IntegratedEventHandler> create_stream_pipeline(
    const IntegratedEventHandler::Config& config = {}) {
    return std::make_unique<IntegratedEventHandler>(config);
}

/**
 * High-level event processing functions for common use cases
 */
namespace event_handlers {

// Real-time metrics processing
void setup_metrics_processing(IntegratedEventHandler& handler, uint32_t tenant_id) {
    // Set up aggregations for metrics
    handler.register_aggregation_callback(
        1, // EVENT_TYPE_METRIC
        tenant_id,
        AggregationType::AVERAGE,
        [](const AggregationResult& result) {
            // Handle average metric aggregation
            printf("Average metric for tenant %u: %.2f (window: %lu-%lu)\n",
                   result.tenant_id, result.value, 
                   result.window_start_ns, result.window_end_ns);
        }
    );
    
    handler.register_aggregation_callback(
        1, // EVENT_TYPE_METRIC
        tenant_id,
        AggregationType::PERCENTILE_95,
        [](const AggregationResult& result) {
            // Handle P95 metric aggregation
            printf("P95 metric for tenant %u: %.2f\n", result.tenant_id, result.value);
        }
    );
    
    // Set up anomaly detection for metrics
    handler.register_anomaly_callback(
        1, // EVENT_TYPE_METRIC
        tenant_id,
        [](const AnomalyResult& result) {
            printf("ANOMALY DETECTED: Tenant %u, Score: %.2f, Value: %.2f, Expected: %.2f\n",
                   result.tenant_id, result.anomaly_score, 
                   result.actual_value, result.expected_value);
        }
    );
}

// User behavior analysis
void setup_user_behavior_analysis(IntegratedEventHandler& handler, uint32_t tenant_id) {
    // Count user actions per window
    handler.register_aggregation_callback(
        3, // EVENT_TYPE_USER_ACTION
        tenant_id,
        AggregationType::COUNT,
        [](const AggregationResult& result) {
            printf("User actions count for tenant %u: %.0f\n", result.tenant_id, result.value);
        }
    );
    
    // Detect unusual user activity patterns
    handler.register_anomaly_callback(
        3, // EVENT_TYPE_USER_ACTION
        tenant_id,
        [](const AnomalyResult& result) {
            printf("Unusual user activity detected: User %u, Tenant %u\n",
                   result.user_id, result.tenant_id);
        }
    );
}

// System monitoring
void setup_system_monitoring(IntegratedEventHandler& handler) {
    // Monitor system events across all tenants
    handler.register_aggregation_callback(
        4, // EVENT_TYPE_SYSTEM
        0, // All tenants
        AggregationType::COUNT,
        [](const AggregationResult& result) {
            printf("System events count: %.0f\n", result.value);
        }
    );
    
    handler.register_anomaly_callback(
        4, // EVENT_TYPE_SYSTEM
        0, // All tenants
        [](const AnomalyResult& result) {
            printf("System anomaly detected: Score %.2f\n", result.anomaly_score);
        }
    );
}

} // namespace event_handlers

} // namespace stream
} // namespace ultra_cpp