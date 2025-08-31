# Integration Examples

## Overview

This document provides practical integration examples for common use cases of the Ultra Low-Latency C++ System. These examples demonstrate how to integrate the high-performance C++ components with existing Node.js infrastructure and external systems.

## Table of Contents

1. [Blog Post Caching Integration](#blog-post-caching-integration)
2. [Real-time Analytics Pipeline](#real-time-analytics-pipeline)
3. [Media Processing Acceleration](#media-processing-acceleration)
4. [User Authentication Fast Path](#user-authentication-fast-path)
5. [Database Query Acceleration](#database-query-acceleration)
6. [API Rate Limiting](#api-rate-limiting)
7. [Content Delivery Optimization](#content-delivery-optimization)
8. [Monitoring and Alerting Integration](#monitoring-and-alerting-integration)

## Blog Post Caching Integration

### Scenario
Accelerate blog post retrieval by caching frequently accessed posts in the ultra-fast C++ cache layer while maintaining consistency with the Node.js backend.

### Implementation

**C++ Cache Service:**
```cpp
#include "cache/ultra_cache.hpp"
#include "api-gateway/fast_api_gateway.hpp"
#include "common/config_manager.hpp"

class BlogPostCacheService {
private:
    UltraCache<std::string, std::string> post_cache_;
    UltraCache<std::string, std::string> metadata_cache_;
    NodeJSProxy nodejs_proxy_;
    
public:
    BlogPostCacheService() 
        : post_cache_({
            .capacity = 100000,
            .shard_count = 64,
            .default_ttl = std::chrono::minutes(30)
          }),
          metadata_cache_({
            .capacity = 50000,
            .shard_count = 32,
            .default_ttl = std::chrono::minutes(10)
          }),
          nodejs_proxy_({
            .upstream_host = "localhost",
            .upstream_port = 3005,
            .timeout = std::chrono::seconds(5)
          }) {}
    
    void setup_routes(FastAPIGateway& gateway) {
        // Fast path for individual blog posts
        gateway.register_fast_route("/api/posts/{id}", 
            [this](const HttpRequest& req, HttpResponse& resp) {
                handle_get_post(req, resp);
            });
        
        // Fast path for post metadata (title, author, date)
        gateway.register_fast_route("/api/posts/{id}/metadata",
            [this](const HttpRequest& req, HttpResponse& resp) {
                handle_get_metadata(req, resp);
            });
        
        // Fast path for popular posts list
        gateway.register_fast_route("/api/posts/popular",
            [this](const HttpRequest& req, HttpResponse& resp) {
                handle_get_popular_posts(req, resp);
            });
    }
    
private:
    void handle_get_post(const HttpRequest& req, HttpResponse& resp) {
        auto post_id = req.path_params.at("id");
        
        // Try cache first
        if (auto cached_post = post_cache_.get(post_id)) {
            resp.set_body(*cached_post);
            resp.set_status(200);
            resp.set_header("Content-Type", "application/json");
            resp.set_header("X-Cache", "HIT");
            resp.set_header("X-Cache-Age", get_cache_age(post_id));
            return;
        }
        
        // Cache miss - fetch from Node.js backend
        try {
            auto nodejs_response = nodejs_proxy_.forward_request(req);
            
            if (nodejs_response.status == 200) {
                // Cache the response for future requests
                post_cache_.put(post_id, nodejs_response.body, std::chrono::minutes(30));
                
                resp.set_body(nodejs_response.body);
                resp.set_status(200);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("X-Cache", "MISS");
            } else {
                resp.set_status(nodejs_response.status);
                resp.set_body(nodejs_response.body);
            }
        } catch (const std::exception& e) {
            resp.set_status(503);
            resp.set_body(R"({"error": "Backend service unavailable"})");
            resp.set_header("X-Error", e.what());
        }
    }
    
    void handle_get_metadata(const HttpRequest& req, HttpResponse& resp) {
        auto post_id = req.path_params.at("id");
        std::string cache_key = "metadata:" + post_id;
        
        if (auto cached_metadata = metadata_cache_.get(cache_key)) {
            resp.set_body(*cached_metadata);
            resp.set_status(200);
            resp.set_header("Content-Type", "application/json");
            resp.set_header("X-Cache", "HIT");
            return;
        }
        
        // Extract metadata from full post cache or fetch from backend
        if (auto cached_post = post_cache_.get(post_id)) {
            auto metadata = extract_metadata(*cached_post);
            metadata_cache_.put(cache_key, metadata, std::chrono::minutes(60));
            
            resp.set_body(metadata);
            resp.set_status(200);
            resp.set_header("Content-Type", "application/json");
            resp.set_header("X-Cache", "DERIVED");
        } else {
            // Fallback to Node.js
            auto nodejs_response = nodejs_proxy_.forward_request(req);
            resp.set_body(nodejs_response.body);
            resp.set_status(nodejs_response.status);
            resp.set_header("X-Cache", "MISS");
        }
    }
    
    std::string extract_metadata(const std::string& post_json) {
        // Fast JSON parsing to extract only metadata fields
        // Using SIMD-accelerated JSON parser for performance
        rapidjson::Document doc;
        doc.Parse(post_json.c_str());
        
        rapidjson::Document metadata;
        metadata.SetObject();
        auto& allocator = metadata.GetAllocator();
        
        if (doc.HasMember("id")) {
            metadata.AddMember("id", doc["id"], allocator);
        }
        if (doc.HasMember("title")) {
            metadata.AddMember("title", doc["title"], allocator);
        }
        if (doc.HasMember("author")) {
            metadata.AddMember("author", doc["author"], allocator);
        }
        if (doc.HasMember("created_at")) {
            metadata.AddMember("created_at", doc["created_at"], allocator);
        }
        if (doc.HasMember("tags")) {
            metadata.AddMember("tags", doc["tags"], allocator);
        }
        
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        metadata.Accept(writer);
        
        return buffer.GetString();
    }
    
    std::string get_cache_age(const std::string& post_id) {
        // Implementation to calculate cache age
        return "120"; // seconds
    }
};
```

**Node.js Integration:**
```javascript
// Node.js cache invalidation service
const Redis = require('redis');
const redis = Redis.createClient();

class CacheInvalidationService {
    constructor() {
        this.cppServiceUrl = 'http://localhost:8080';
    }
    
    async invalidatePost(postId) {
        try {
            // Invalidate in C++ cache
            await fetch(`${this.cppServiceUrl}/internal/cache/invalidate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    keys: [`post:${postId}`, `metadata:${postId}`] 
                })
            });
            
            // Also invalidate Redis cache
            await redis.del(`post:${postId}`);
            
            console.log(`Invalidated cache for post ${postId}`);
        } catch (error) {
            console.error('Cache invalidation failed:', error);
        }
    }
    
    async warmCache(postId) {
        try {
            // Pre-warm the C++ cache by making a request
            await fetch(`${this.cppServiceUrl}/api/posts/${postId}`);
            console.log(`Warmed cache for post ${postId}`);
        } catch (error) {
            console.error('Cache warming failed:', error);
        }
    }
}

// Usage in existing Node.js routes
app.put('/api/posts/:id', async (req, res) => {
    try {
        // Update post in database
        const updatedPost = await Post.findByIdAndUpdate(req.params.id, req.body);
        
        // Invalidate caches
        await cacheInvalidationService.invalidatePost(req.params.id);
        
        res.json(updatedPost);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```

## Real-time Analytics Pipeline

### Scenario
Process user interaction events in real-time to generate analytics dashboards and trigger automated responses.

### Implementation

**C++ Event Processing:**
```cpp
#include "stream-processor/stream_processor.hpp"
#include "gpu-compute/gpu_compute_engine.hpp"

class AnalyticsEventProcessor {
private:
    StreamProcessor stream_processor_;
    GPUComputeEngine gpu_engine_;
    UltraCache<std::string, std::string> analytics_cache_;
    
    // Event types
    enum EventType : uint32_t {
        PAGE_VIEW = 1,
        POST_LIKE = 2,
        COMMENT_CREATED = 3,
        USER_SIGNUP = 4,
        SEARCH_QUERY = 5
    };
    
    struct PageViewEvent {
        uint32_t user_id;
        uint32_t post_id;
        uint32_t session_duration_ms;
        char referrer[128];
        char user_agent[256];
    };
    
    struct PostLikeEvent {
        uint32_t user_id;
        uint32_t post_id;
        uint32_t author_id;
    };
    
public:
    AnalyticsEventProcessor() 
        : stream_processor_({
            .ring_buffer_size = 2 * 1024 * 1024,
            .worker_threads = 8,
            .enable_simd = true
          }),
          gpu_engine_({
            .device_id = 0,
            .memory_pool_size = 512 * 1024 * 1024
          }),
          analytics_cache_({
            .capacity = 1000000,
            .shard_count = 128
          }) {
        setup_event_handlers();
    }
    
    void start() {
        stream_processor_.start_processing();
        
        // Create sliding windows for real-time metrics
        stream_processor_.create_sliding_window("page_views_5min", 
            std::chrono::minutes(5), std::chrono::seconds(10));
        stream_processor_.create_sliding_window("user_activity_1hour",
            std::chrono::hours(1), std::chrono::minutes(1));
    }
    
    void process_event(const std::string& event_json) {
        // Parse event and convert to internal format
        auto event = parse_json_event(event_json);
        stream_processor_.publish(event);
    }
    
private:
    void setup_event_handlers() {
        // Page view analytics
        stream_processor_.subscribe(PAGE_VIEW, [this](const StreamProcessor::Event& event) {
            auto page_view = event.get_data<PageViewEvent>();
            
            // Update real-time counters
            increment_counter("page_views_total");
            increment_counter("page_views_post_" + std::to_string(page_view->post_id));
            
            // Track user engagement
            if (page_view->session_duration_ms > 30000) { // 30 seconds
                increment_counter("engaged_views_total");
            }
            
            // Update popular posts ranking
            update_popular_posts_ranking(page_view->post_id);
            
            // Detect trending content
            detect_trending_content(page_view->post_id);
        });
        
        // Post like analytics
        stream_processor_.subscribe(POST_LIKE, [this](const StreamProcessor::Event& event) {
            auto like_event = event.get_data<PostLikeEvent>();
            
            increment_counter("likes_total");
            increment_counter("likes_post_" + std::to_string(like_event->post_id));
            increment_counter("likes_author_" + std::to_string(like_event->author_id));
            
            // Update author engagement metrics
            update_author_metrics(like_event->author_id);
            
            // Trigger notifications for popular posts
            check_viral_threshold(like_event->post_id);
        });
        
        // User signup analytics
        stream_processor_.subscribe(USER_SIGNUP, [this](const StreamProcessor::Event& event) {
            increment_counter("signups_total");
            increment_counter("signups_today");
            
            // Update conversion funnel metrics
            update_conversion_metrics();
        });
    }
    
    void increment_counter(const std::string& key) {
        auto current_value = analytics_cache_.get(key).value_or("0");
        auto new_value = std::to_string(std::stoll(current_value) + 1);
        analytics_cache_.put(key, new_value, std::chrono::minutes(60));
    }
    
    void update_popular_posts_ranking(uint32_t post_id) {
        // Use GPU for parallel ranking computation
        std::vector<float> post_scores = get_current_post_scores();
        
        // Update score for this post
        auto it = std::find_if(post_scores.begin(), post_scores.end(),
            [post_id](float score) { /* find post logic */ return false; });
        
        if (it != post_scores.end()) {
            *it += 1.0f; // Increment score
        }
        
        // GPU-accelerated sorting for top posts
        auto top_posts = gpu_engine_.compute_top_k(post_scores, 100);
        
        // Cache the updated rankings
        std::string rankings_json = serialize_rankings(top_posts);
        analytics_cache_.put("popular_posts", rankings_json, std::chrono::minutes(5));
    }
    
    void detect_trending_content(uint32_t post_id) {
        // Get recent view counts for this post
        auto recent_views = get_recent_view_count(post_id, std::chrono::minutes(15));
        auto historical_avg = get_historical_average(post_id);
        
        // Detect if trending (views > 3x historical average)
        if (recent_views > historical_avg * 3.0f) {
            // Trigger trending notification
            StreamProcessor::Event trending_event;
            trending_event.type = 999; // TRENDING_DETECTED
            trending_event.timestamp_ns = std::chrono::high_resolution_clock::now()
                                        .time_since_epoch().count();
            
            stream_processor_.publish(trending_event);
            
            // Cache trending status
            analytics_cache_.put("trending_post_" + std::to_string(post_id), 
                               "true", std::chrono::minutes(30));
        }
    }
    
    StreamProcessor::Event parse_json_event(const std::string& json) {
        // Fast JSON parsing using SIMD
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        
        StreamProcessor::Event event;
        event.timestamp_ns = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
        event.type = doc["type"].GetUint();
        
        // Serialize event data based on type
        switch (event.type) {
            case PAGE_VIEW: {
                PageViewEvent page_view;
                page_view.user_id = doc["user_id"].GetUint();
                page_view.post_id = doc["post_id"].GetUint();
                page_view.session_duration_ms = doc["duration"].GetUint();
                
                std::memcpy(event.data, &page_view, sizeof(PageViewEvent));
                event.size = sizeof(PageViewEvent);
                break;
            }
            // Handle other event types...
        }
        
        return event;
    }
    
    std::vector<float> get_current_post_scores() {
        // Implementation to retrieve current post scores
        return {};
    }
    
    float get_recent_view_count(uint32_t post_id, std::chrono::minutes window) {
        // Implementation to get recent view count
        return 0.0f;
    }
    
    float get_historical_average(uint32_t post_id) {
        // Implementation to get historical average
        return 0.0f;
    }
    
    std::string serialize_rankings(const std::vector<uint32_t>& top_posts) {
        // Implementation to serialize rankings to JSON
        return "{}";
    }
    
    void update_author_metrics(uint32_t author_id) {
        // Implementation to update author engagement metrics
    }
    
    void check_viral_threshold(uint32_t post_id) {
        // Implementation to check if post has gone viral
    }
    
    void update_conversion_metrics() {
        // Implementation to update conversion funnel metrics
    }
};
```

**Node.js Event Publisher:**
```javascript
// Node.js event publisher
const EventEmitter = require('events');
const WebSocket = require('ws');

class AnalyticsEventPublisher extends EventEmitter {
    constructor() {
        super();
        this.cppAnalyticsWs = new WebSocket('ws://localhost:8081/analytics');
        this.eventBuffer = [];
        this.batchSize = 100;
        this.flushInterval = 100; // ms
        
        this.setupBatchProcessing();
    }
    
    publishEvent(eventType, eventData) {
        const event = {
            type: eventType,
            timestamp: Date.now(),
            ...eventData
        };
        
        this.eventBuffer.push(event);
        
        if (this.eventBuffer.length >= this.batchSize) {
            this.flushEvents();
        }
    }
    
    setupBatchProcessing() {
        setInterval(() => {
            if (this.eventBuffer.length > 0) {
                this.flushEvents();
            }
        }, this.flushInterval);
    }
    
    flushEvents() {
        if (this.cppAnalyticsWs.readyState === WebSocket.OPEN) {
            this.cppAnalyticsWs.send(JSON.stringify({
                events: this.eventBuffer
            }));
            this.eventBuffer = [];
        }
    }
}

// Usage in existing routes
const analyticsPublisher = new AnalyticsEventPublisher();

app.get('/api/posts/:id', async (req, res) => {
    const startTime = Date.now();
    
    try {
        const post = await Post.findById(req.params.id);
        
        // Publish page view event
        analyticsPublisher.publishEvent('PAGE_VIEW', {
            user_id: req.user?.id || 0,
            post_id: req.params.id,
            session_duration_ms: 0, // Will be updated by frontend
            referrer: req.get('Referer') || '',
            user_agent: req.get('User-Agent') || ''
        });
        
        res.json(post);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/posts/:id/like', async (req, res) => {
    try {
        const post = await Post.findById(req.params.id);
        
        // Update like count
        post.likes += 1;
        await post.save();
        
        // Publish like event
        analyticsPublisher.publishEvent('POST_LIKE', {
            user_id: req.user.id,
            post_id: req.params.id,
            author_id: post.author_id
        });
        
        res.json({ likes: post.likes });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```

## Media Processing Acceleration

### Scenario
Accelerate image and video processing using GPU compute for thumbnails, transcoding, and content analysis.

### Implementation

**C++ Media Processor:**
```cpp
#include "gpu-compute/gpu_compute_engine.hpp"
#include "memory/numa_allocator.hpp"

class MediaProcessingService {
private:
    GPUComputeEngine gpu_engine_;
    MemoryPool<uint8_t> image_pool_;
    
public:
    MediaProcessingService() 
        : gpu_engine_({
            .device_id = 0,
            .memory_pool_size = 2ULL * 1024 * 1024 * 1024, // 2GB
            .enable_tensorrt = true
          }) {
        
        // Load ML models for content analysis
        gpu_engine_.load_model("nsfw_classifier", "/models/nsfw_detection.onnx");
        gpu_engine_.load_model("object_detector", "/models/yolo_v8.trt");
        gpu_engine_.load_model("face_detector", "/models/face_detection.trt");
    }
    
    struct ProcessingResult {
        std::vector<uint8_t> thumbnail_data;
        std::vector<uint8_t> compressed_data;
        ContentAnalysis analysis;
        ProcessingMetrics metrics;
    };
    
    struct ContentAnalysis {
        float nsfw_score;
        std::vector<DetectedObject> objects;
        std::vector<DetectedFace> faces;
        ImageQualityMetrics quality;
    };
    
    ProcessingResult process_image(const std::vector<uint8_t>& image_data,
                                 const ProcessingOptions& options) {
        PerformanceMonitor::Timer timer("image_processing");
        
        ProcessingResult result;
        
        // Decode image on GPU
        auto gpu_image = decode_image_gpu(image_data);
        
        // Parallel processing pipeline
        std::vector<std::future<void>> tasks;
        
        // Task 1: Generate thumbnails
        tasks.emplace_back(std::async(std::launch::async, [&]() {
            result.thumbnail_data = generate_thumbnails_gpu(gpu_image, options.thumbnail_sizes);
        }));
        
        // Task 2: Compress image
        tasks.emplace_back(std::async(std::launch::async, [&]() {
            result.compressed_data = compress_image_gpu(gpu_image, options.quality);
        }));
        
        // Task 3: Content analysis
        tasks.emplace_back(std::async(std::launch::async, [&]() {
            result.analysis = analyze_content_gpu(gpu_image);
        }));
        
        // Wait for all tasks to complete
        for (auto& task : tasks) {
            task.wait();
        }
        
        result.metrics.processing_time_ms = timer.elapsed_ns() / 1000000;
        return result;
    }
    
    ProcessingResult process_video(const std::vector<uint8_t>& video_data,
                                 const VideoProcessingOptions& options) {
        PerformanceMonitor::Timer timer("video_processing");
        
        ProcessingResult result;
        
        // Hardware-accelerated video decoding
        auto video_frames = decode_video_hw(video_data);
        
        // Parallel frame processing
        std::vector<std::future<FrameResult>> frame_tasks;
        
        for (size_t i = 0; i < video_frames.size(); i += options.keyframe_interval) {
            frame_tasks.emplace_back(std::async(std::launch::async, [&, i]() {
                return process_video_frame(video_frames[i], options);
            }));
        }
        
        // Collect results
        std::vector<FrameResult> frame_results;
        for (auto& task : frame_tasks) {
            frame_results.push_back(task.get());
        }
        
        // Generate video thumbnails and previews
        result.thumbnail_data = generate_video_thumbnails(frame_results);
        result.compressed_data = encode_video_hw(video_frames, options);
        result.analysis = analyze_video_content(frame_results);
        
        result.metrics.processing_time_ms = timer.elapsed_ns() / 1000000;
        return result;
    }
    
private:
    GPUImage decode_image_gpu(const std::vector<uint8_t>& image_data) {
        // Use GPU-accelerated image decoding (NVJPEG, etc.)
        GPUImage gpu_image;
        
        // Allocate GPU memory
        cudaMalloc(&gpu_image.data, image_data.size());
        cudaMemcpy(gpu_image.data, image_data.data(), image_data.size(), 
                  cudaMemcpyHostToDevice);
        
        // Decode using hardware acceleration
        nvjpegDecode(gpu_image.data, image_data.size(), &gpu_image);
        
        return gpu_image;
    }
    
    std::vector<uint8_t> generate_thumbnails_gpu(const GPUImage& image,
                                               const std::vector<ThumbnailSize>& sizes) {
        std::vector<uint8_t> thumbnails;
        
        for (const auto& size : sizes) {
            // GPU-accelerated resize using CUDA kernels
            GPUImage resized = resize_image_cuda(image, size.width, size.height);
            
            // Encode thumbnail
            auto encoded = encode_image_gpu(resized, ImageFormat::JPEG, 85);
            
            // Append to result
            thumbnails.insert(thumbnails.end(), encoded.begin(), encoded.end());
        }
        
        return thumbnails;
    }
    
    ContentAnalysis analyze_content_gpu(const GPUImage& image) {
        ContentAnalysis analysis;
        
        // Prepare input for ML models
        auto ml_input = preprocess_for_ml(image);
        
        // Run NSFW classification
        auto nsfw_result = gpu_engine_.infer("nsfw_classifier", ml_input);
        analysis.nsfw_score = nsfw_result[0];
        
        // Run object detection
        auto object_result = gpu_engine_.infer("object_detector", ml_input);
        analysis.objects = parse_object_detection_result(object_result);
        
        // Run face detection
        auto face_result = gpu_engine_.infer("face_detector", ml_input);
        analysis.faces = parse_face_detection_result(face_result);
        
        // Calculate image quality metrics
        analysis.quality = calculate_quality_metrics_gpu(image);
        
        return analysis;
    }
    
    std::vector<float> preprocess_for_ml(const GPUImage& image) {
        // Normalize and resize for ML model input
        std::vector<float> input;
        
        // GPU kernel for preprocessing
        preprocess_image_cuda(image, input.data(), 224, 224);
        
        return input;
    }
    
    std::vector<DetectedObject> parse_object_detection_result(
        const std::vector<float>& result) {
        std::vector<DetectedObject> objects;
        
        // Parse YOLO output format
        for (size_t i = 0; i < result.size(); i += 6) {
            if (result[i + 4] > 0.5f) { // Confidence threshold
                DetectedObject obj;
                obj.class_id = static_cast<int>(result[i + 5]);
                obj.confidence = result[i + 4];
                obj.bbox = {result[i], result[i + 1], result[i + 2], result[i + 3]};
                objects.push_back(obj);
            }
        }
        
        return objects;
    }
};
```

**Node.js Integration:**
```javascript
// Node.js media upload handler
const multer = require('multer');
const { Worker } = require('worker_threads');

class MediaUploadService {
    constructor() {
        this.cppProcessorUrl = 'http://localhost:8082';
        this.processingQueue = [];
        this.maxConcurrentJobs = 4;
        this.activeJobs = 0;
    }
    
    async handleUpload(file) {
        try {
            // Quick validation
            if (!this.isValidMediaFile(file)) {
                throw new Error('Invalid media file type');
            }
            
            // Create media record
            const media = new Media({
                filename: file.filename,
                originalName: file.originalname,
                mimeType: file.mimetype,
                size: file.size,
                status: 'processing',
                uploadedAt: new Date()
            });
            
            await media.save();
            
            // Queue for C++ processing
            this.queueProcessing(media._id, file.path);
            
            return {
                id: media._id,
                status: 'processing',
                message: 'File uploaded successfully, processing in progress'
            };
            
        } catch (error) {
            throw new Error(`Upload failed: ${error.message}`);
        }
    }
    
    async queueProcessing(mediaId, filePath) {
        const job = {
            mediaId,
            filePath,
            timestamp: Date.now()
        };
        
        this.processingQueue.push(job);
        this.processNextJob();
    }
    
    async processNextJob() {
        if (this.activeJobs >= this.maxConcurrentJobs || this.processingQueue.length === 0) {
            return;
        }
        
        const job = this.processingQueue.shift();
        this.activeJobs++;
        
        try {
            await this.processWithCpp(job);
        } catch (error) {
            console.error('Processing failed:', error);
            await this.markProcessingFailed(job.mediaId, error.message);
        } finally {
            this.activeJobs--;
            this.processNextJob(); // Process next job in queue
        }
    }
    
    async processWithCpp(job) {
        const fs = require('fs');
        const fileData = fs.readFileSync(job.filePath);
        
        // Send to C++ processor
        const response = await fetch(`${this.cppProcessorUrl}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/octet-stream',
                'X-Media-ID': job.mediaId,
                'X-Processing-Options': JSON.stringify({
                    thumbnail_sizes: [
                        { width: 150, height: 150 },
                        { width: 300, height: 300 },
                        { width: 800, height: 600 }
                    ],
                    quality: 85,
                    enable_analysis: true
                })
            },
            body: fileData
        });
        
        if (!response.ok) {
            throw new Error(`C++ processor returned ${response.status}`);
        }
        
        const result = await response.json();
        
        // Update media record with results
        await this.updateMediaRecord(job.mediaId, result);
        
        // Clean up temporary file
        fs.unlinkSync(job.filePath);
    }
    
    async updateMediaRecord(mediaId, processingResult) {
        const media = await Media.findById(mediaId);
        
        media.status = 'completed';
        media.thumbnails = processingResult.thumbnails;
        media.compressed_url = processingResult.compressed_url;
        media.analysis = {
            nsfw_score: processingResult.analysis.nsfw_score,
            detected_objects: processingResult.analysis.objects,
            quality_score: processingResult.analysis.quality.overall_score
        };
        media.processing_metrics = {
            processing_time_ms: processingResult.metrics.processing_time_ms,
            file_size_reduction: processingResult.metrics.compression_ratio
        };
        media.processedAt = new Date();
        
        await media.save();
        
        // Emit event for real-time updates
        this.emit('mediaProcessed', {
            mediaId: media._id,
            status: 'completed',
            thumbnails: media.thumbnails
        });
    }
    
    isValidMediaFile(file) {
        const allowedTypes = [
            'image/jpeg', 'image/png', 'image/webp',
            'video/mp4', 'video/webm', 'video/quicktime'
        ];
        
        return allowedTypes.includes(file.mimetype) && file.size <= 100 * 1024 * 1024; // 100MB
    }
}

// Express route setup
const upload = multer({ dest: 'temp_uploads/' });
const mediaService = new MediaUploadService();

app.post('/api/media/upload', upload.single('file'), async (req, res) => {
    try {
        const result = await mediaService.handleUpload(req.file);
        res.json(result);
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

app.get('/api/media/:id/status', async (req, res) => {
    try {
        const media = await Media.findById(req.params.id);
        res.json({
            id: media._id,
            status: media.status,
            thumbnails: media.thumbnails,
            analysis: media.analysis
        });
    } catch (error) {
        res.status(404).json({ error: 'Media not found' });
    }
});
```## User Auth
entication Fast Path

### Scenario
Accelerate JWT token validation and user session management using C++ for high-frequency authentication checks.

### Implementation

**C++ Authentication Service:**
```cpp
#include "security/jwt_validator.hpp"
#include "cache/ultra_cache.hpp"

class FastAuthenticationService {
private:
    JWTValidator jwt_validator_;
    UltraCache<std::string, std::string> token_cache_;
    UltraCache<std::string, UserSession> session_cache_;
    
public:
    FastAuthenticationService() 
        : jwt_validator_({
            .public_key_path = "/etc/ssl/jwt_public.pem",
            .algorithm = "RS256",
            .issuer = "blog-platform",
            .audience = "api"
          }),
          token_cache_({
            .capacity = 1000000,
            .shard_count = 64,
            .default_ttl = std::chrono::minutes(15)
          }),
          session_cache_({
            .capacity = 500000,
            .shard_count = 32,
            .default_ttl = std::chrono::hours(24)
          }) {}
    
    struct AuthResult {
        bool is_valid;
        uint32_t user_id;
        std::vector<std::string> permissions;
        std::chrono::seconds expires_in;
        std::string error_message;
    };
    
    AuthResult validate_token(const std::string& token) {
        PerformanceMonitor::Timer timer("token_validation");
        
        // Check token cache first
        if (auto cached_result = token_cache_.get(token)) {
            return deserialize_auth_result(*cached_result);
        }
        
        AuthResult result;
        
        try {
            // Fast JWT validation using SIMD
            auto jwt_claims = jwt_validator_.validate_fast(token);
            
            if (jwt_claims.is_valid) {
                result.is_valid = true;
                result.user_id = jwt_claims.user_id;
                result.permissions = jwt_claims.permissions;
                result.expires_in = jwt_claims.expires_in;
                
                // Cache the result
                auto serialized = serialize_auth_result(result);
                token_cache_.put(token, serialized, jwt_claims.expires_in);
                
                // Update session cache
                update_user_session(result.user_id, jwt_claims);
            } else {
                result.is_valid = false;
                result.error_message = jwt_claims.error_message;
            }
            
        } catch (const std::exception& e) {
            result.is_valid = false;
            result.error_message = e.what();
        }
        
        return result;
    }
    
    bool check_permission(uint32_t user_id, const std::string& permission) {
        std::string session_key = "session:" + std::to_string(user_id);
        
        if (auto session = session_cache_.get(session_key)) {
            return has_permission(*session, permission);
        }
        
        return false; // Session not found or expired
    }
    
    void invalidate_user_sessions(uint32_t user_id) {
        std::string session_key = "session:" + std::to_string(user_id);
        session_cache_.remove(session_key);
        
        // Also remove from token cache (requires scanning)
        // This is expensive but necessary for security
        invalidate_user_tokens(user_id);
    }
    
    struct SessionStats {
        uint64_t active_sessions;
        uint64_t token_validations_per_second;
        uint64_t cache_hit_ratio_percent;
        uint64_t avg_validation_time_ns;
    };
    
    SessionStats get_session_stats() const {
        auto token_stats = token_cache_.get_stats();
        auto session_stats = session_cache_.get_stats();
        
        return {
            .active_sessions = session_stats.size,
            .token_validations_per_second = calculate_validation_rate(),
            .cache_hit_ratio_percent = static_cast<uint64_t>(
                (token_stats.hits * 100) / (token_stats.hits + token_stats.misses)),
            .avg_validation_time_ns = get_avg_validation_time()
        };
    }
    
private:
    struct UserSession {
        uint32_t user_id;
        std::vector<std::string> permissions;
        std::chrono::system_clock::time_point last_activity;
        std::string ip_address;
        std::string user_agent;
    };
    
    void update_user_session(uint32_t user_id, const JWTClaims& claims) {
        UserSession session;
        session.user_id = user_id;
        session.permissions = claims.permissions;
        session.last_activity = std::chrono::system_clock::now();
        
        std::string session_key = "session:" + std::to_string(user_id);
        session_cache_.put(session_key, session, std::chrono::hours(24));
    }
    
    bool has_permission(const UserSession& session, const std::string& permission) {
        return std::find(session.permissions.begin(), session.permissions.end(), 
                        permission) != session.permissions.end();
    }
    
    std::string serialize_auth_result(const AuthResult& result) {
        rapidjson::Document doc;
        doc.SetObject();
        auto& allocator = doc.GetAllocator();
        
        doc.AddMember("is_valid", result.is_valid, allocator);
        doc.AddMember("user_id", result.user_id, allocator);
        doc.AddMember("expires_in", result.expires_in.count(), allocator);
        
        rapidjson::Value permissions(rapidjson::kArrayType);
        for (const auto& perm : result.permissions) {
            permissions.PushBack(rapidjson::Value(perm.c_str(), allocator), allocator);
        }
        doc.AddMember("permissions", permissions, allocator);
        
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        
        return buffer.GetString();
    }
    
    AuthResult deserialize_auth_result(const std::string& json) {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        
        AuthResult result;
        result.is_valid = doc["is_valid"].GetBool();
        result.user_id = doc["user_id"].GetUint();
        result.expires_in = std::chrono::seconds(doc["expires_in"].GetInt64());
        
        for (const auto& perm : doc["permissions"].GetArray()) {
            result.permissions.push_back(perm.GetString());
        }
        
        return result;
    }
    
    void invalidate_user_tokens(uint32_t user_id) {
        // Implementation to scan and invalidate user tokens
        // This is a heavy operation and should be used sparingly
    }
    
    uint64_t calculate_validation_rate() const {
        // Implementation to calculate validations per second
        return 0;
    }
    
    uint64_t get_avg_validation_time() const {
        // Implementation to get average validation time
        return 0;
    }
};
```

**Node.js Middleware Integration:**
```javascript
// Enhanced authentication middleware
const jwt = require('jsonwebtoken');

class AuthenticationMiddleware {
    constructor() {
        this.cppAuthUrl = 'http://localhost:8083';
        this.fallbackToNodeJS = true;
        this.cppHealthy = true;
        
        // Health check for C++ service
        this.setupHealthCheck();
    }
    
    // Fast path authentication using C++ service
    async authenticateRequest(req, res, next) {
        const token = this.extractToken(req);
        
        if (!token) {
            return res.status(401).json({ error: 'No token provided' });
        }
        
        try {
            let authResult;
            
            if (this.cppHealthy) {
                // Try C++ fast path first
                authResult = await this.validateWithCpp(token);
            } else {
                // Fallback to Node.js
                authResult = await this.validateWithNodeJS(token);
            }
            
            if (authResult.is_valid) {
                req.user = {
                    id: authResult.user_id,
                    permissions: authResult.permissions
                };
                next();
            } else {
                res.status(401).json({ error: authResult.error_message || 'Invalid token' });
            }
            
        } catch (error) {
            console.error('Authentication error:', error);
            
            if (this.fallbackToNodeJS && this.cppHealthy) {
                // C++ service failed, try Node.js fallback
                try {
                    const authResult = await this.validateWithNodeJS(token);
                    if (authResult.is_valid) {
                        req.user = {
                            id: authResult.user_id,
                            permissions: authResult.permissions
                        };
                        return next();
                    }
                } catch (fallbackError) {
                    console.error('Fallback authentication also failed:', fallbackError);
                }
            }
            
            res.status(500).json({ error: 'Authentication service unavailable' });
        }
    }
    
    async validateWithCpp(token) {
        const response = await fetch(`${this.cppAuthUrl}/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            timeout: 100 // 100ms timeout for ultra-fast response
        });
        
        if (!response.ok) {
            throw new Error(`C++ auth service returned ${response.status}`);
        }
        
        return await response.json();
    }
    
    async validateWithNodeJS(token) {
        try {
            const decoded = jwt.verify(token, process.env.JWT_PUBLIC_KEY, {
                algorithms: ['RS256'],
                issuer: 'blog-platform',
                audience: 'api'
            });
            
            // Get user permissions from database
            const user = await User.findById(decoded.user_id).select('permissions');
            
            return {
                is_valid: true,
                user_id: decoded.user_id,
                permissions: user ? user.permissions : [],
                expires_in: decoded.exp - Math.floor(Date.now() / 1000)
            };
            
        } catch (error) {
            return {
                is_valid: false,
                error_message: error.message
            };
        }
    }
    
    extractToken(req) {
        const authHeader = req.headers.authorization;
        if (authHeader && authHeader.startsWith('Bearer ')) {
            return authHeader.substring(7);
        }
        
        // Also check cookies for web clients
        return req.cookies?.auth_token;
    }
    
    setupHealthCheck() {
        setInterval(async () => {
            try {
                const response = await fetch(`${this.cppAuthUrl}/health`, {
                    timeout: 1000
                });
                this.cppHealthy = response.ok;
            } catch (error) {
                this.cppHealthy = false;
                console.warn('C++ auth service health check failed:', error.message);
            }
        }, 5000); // Check every 5 seconds
    }
    
    // Permission checking middleware
    requirePermission(permission) {
        return async (req, res, next) => {
            if (!req.user) {
                return res.status(401).json({ error: 'Authentication required' });
            }
            
            let hasPermission = false;
            
            if (this.cppHealthy) {
                // Use C++ service for fast permission check
                try {
                    const response = await fetch(`${this.cppAuthUrl}/check-permission`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: req.user.id,
                            permission: permission
                        }),
                        timeout: 50 // 50ms timeout
                    });
                    
                    const result = await response.json();
                    hasPermission = result.has_permission;
                } catch (error) {
                    // Fallback to local check
                    hasPermission = req.user.permissions.includes(permission);
                }
            } else {
                hasPermission = req.user.permissions.includes(permission);
            }
            
            if (hasPermission) {
                next();
            } else {
                res.status(403).json({ error: 'Insufficient permissions' });
            }
        };
    }
}

// Usage
const authMiddleware = new AuthenticationMiddleware();

// Apply to all protected routes
app.use('/api', authMiddleware.authenticateRequest.bind(authMiddleware));

// Specific permission requirements
app.post('/api/posts', 
    authMiddleware.requirePermission('create_post'),
    async (req, res) => {
        // Create post logic
    }
);

app.delete('/api/posts/:id',
    authMiddleware.requirePermission('delete_post'),
    async (req, res) => {
        // Delete post logic
    }
);
```

## Database Query Acceleration

### Scenario
Accelerate frequently executed database queries using C++ with connection pooling and prepared statements.

### Implementation

**C++ Database Accelerator:**
```cpp
#include "database/high_performance_connector.hpp"
#include "cache/ultra_cache.hpp"

class DatabaseAccelerator {
private:
    HighPerformanceConnector db_connector_;
    UltraCache<std::string, std::string> query_cache_;
    
public:
    DatabaseAccelerator() 
        : db_connector_({
            .host = "localhost",
            .port = 5432,
            .database = "blog_platform",
            .pool_size = 50,
            .enable_prepared_statements = true,
            .enable_async_io = true
          }),
          query_cache_({
            .capacity = 1000000,
            .shard_count = 128,
            .default_ttl = std::chrono::minutes(5)
          }) {
        
        setup_prepared_statements();
    }
    
    struct QueryResult {
        bool success;
        std::vector<std::map<std::string, std::string>> rows;
        uint64_t execution_time_us;
        bool from_cache;
        std::string error_message;
    };
    
    // Fast post retrieval
    QueryResult get_post_by_id(uint32_t post_id) {
        std::string cache_key = "post:" + std::to_string(post_id);
        
        if (auto cached = query_cache_.get(cache_key)) {
            return {
                .success = true,
                .rows = deserialize_rows(*cached),
                .execution_time_us = 0,
                .from_cache = true
            };
        }
        
        PerformanceMonitor::Timer timer("db_query_post");
        
        try {
            auto result = db_connector_.execute_prepared("get_post_by_id", {
                std::to_string(post_id)
            });
            
            if (!result.rows.empty()) {
                // Cache the result
                auto serialized = serialize_rows(result.rows);
                query_cache_.put(cache_key, serialized, std::chrono::minutes(30));
            }
            
            return {
                .success = true,
                .rows = result.rows,
                .execution_time_us = timer.elapsed_ns() / 1000,
                .from_cache = false
            };
            
        } catch (const std::exception& e) {
            return {
                .success = false,
                .execution_time_us = timer.elapsed_ns() / 1000,
                .from_cache = false,
                .error_message = e.what()
            };
        }
    }
    
    // Fast user posts retrieval with pagination
    QueryResult get_user_posts(uint32_t user_id, uint32_t limit, uint32_t offset) {
        std::string cache_key = "user_posts:" + std::to_string(user_id) + 
                               ":" + std::to_string(limit) + ":" + std::to_string(offset);
        
        if (auto cached = query_cache_.get(cache_key)) {
            return {
                .success = true,
                .rows = deserialize_rows(*cached),
                .execution_time_us = 0,
                .from_cache = true
            };
        }
        
        PerformanceMonitor::Timer timer("db_query_user_posts");
        
        try {
            auto result = db_connector_.execute_prepared("get_user_posts", {
                std::to_string(user_id),
                std::to_string(limit),
                std::to_string(offset)
            });
            
            // Cache for shorter time due to frequent updates
            auto serialized = serialize_rows(result.rows);
            query_cache_.put(cache_key, serialized, std::chrono::minutes(2));
            
            return {
                .success = true,
                .rows = result.rows,
                .execution_time_us = timer.elapsed_ns() / 1000,
                .from_cache = false
            };
            
        } catch (const std::exception& e) {
            return {
                .success = false,
                .execution_time_us = timer.elapsed_ns() / 1000,
                .from_cache = false,
                .error_message = e.what()
            };
        }
    }
    
    // Batch operations for better performance
    std::vector<QueryResult> get_posts_batch(const std::vector<uint32_t>& post_ids) {
        std::vector<QueryResult> results;
        std::vector<uint32_t> cache_misses;
        
        // Check cache for all posts first
        for (uint32_t post_id : post_ids) {
            std::string cache_key = "post:" + std::to_string(post_id);
            
            if (auto cached = query_cache_.get(cache_key)) {
                results.push_back({
                    .success = true,
                    .rows = deserialize_rows(*cached),
                    .execution_time_us = 0,
                    .from_cache = true
                });
            } else {
                cache_misses.push_back(post_id);
                results.push_back({}); // Placeholder
            }
        }
        
        // Batch fetch cache misses
        if (!cache_misses.empty()) {
            auto batch_results = fetch_posts_batch(cache_misses);
            
            // Fill in the placeholders
            size_t batch_index = 0;
            for (size_t i = 0; i < results.size(); ++i) {
                if (results[i].rows.empty() && batch_index < batch_results.size()) {
                    results[i] = batch_results[batch_index++];
                }
            }
        }
        
        return results;
    }
    
    void invalidate_cache_pattern(const std::string& pattern) {
        // Invalidate cache entries matching pattern
        // This is expensive but necessary for consistency
        query_cache_.remove_pattern(pattern);
    }
    
    struct DatabaseStats {
        uint64_t queries_per_second;
        uint64_t cache_hit_ratio_percent;
        uint64_t avg_query_time_us;
        uint64_t active_connections;
        uint64_t connection_pool_utilization_percent;
    };
    
    DatabaseStats get_database_stats() const {
        auto cache_stats = query_cache_.get_stats();
        auto db_stats = db_connector_.get_stats();
        
        return {
            .queries_per_second = db_stats.queries_per_second,
            .cache_hit_ratio_percent = static_cast<uint64_t>(
                (cache_stats.hits * 100) / (cache_stats.hits + cache_stats.misses)),
            .avg_query_time_us = db_stats.avg_query_time_us,
            .active_connections = db_stats.active_connections,
            .connection_pool_utilization_percent = 
                (db_stats.active_connections * 100) / db_stats.pool_size
        };
    }
    
private:
    void setup_prepared_statements() {
        // Prepare frequently used queries
        db_connector_.prepare_statement("get_post_by_id",
            "SELECT id, title, content, author_id, created_at, updated_at "
            "FROM posts WHERE id = $1 AND deleted_at IS NULL");
        
        db_connector_.prepare_statement("get_user_posts",
            "SELECT id, title, content, created_at "
            "FROM posts WHERE author_id = $1 AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT $2 OFFSET $3");
        
        db_connector_.prepare_statement("get_popular_posts",
            "SELECT p.id, p.title, p.author_id, p.created_at, "
            "COUNT(l.id) as like_count "
            "FROM posts p LEFT JOIN likes l ON p.id = l.post_id "
            "WHERE p.created_at > NOW() - INTERVAL '24 hours' "
            "GROUP BY p.id ORDER BY like_count DESC LIMIT $1");
        
        db_connector_.prepare_statement("search_posts",
            "SELECT id, title, content, author_id, created_at, "
            "ts_rank(search_vector, plainto_tsquery($1)) as rank "
            "FROM posts WHERE search_vector @@ plainto_tsquery($1) "
            "ORDER BY rank DESC LIMIT $2 OFFSET $3");
    }
    
    std::vector<QueryResult> fetch_posts_batch(const std::vector<uint32_t>& post_ids) {
        // Build batch query
        std::string query = "SELECT id, title, content, author_id, created_at, updated_at "
                           "FROM posts WHERE id = ANY($1) AND deleted_at IS NULL";
        
        // Convert post_ids to PostgreSQL array format
        std::string ids_array = "{";
        for (size_t i = 0; i < post_ids.size(); ++i) {
            if (i > 0) ids_array += ",";
            ids_array += std::to_string(post_ids[i]);
        }
        ids_array += "}";
        
        PerformanceMonitor::Timer timer("db_batch_query");
        
        try {
            auto result = db_connector_.execute_query(query, {ids_array});
            
            // Cache individual results
            for (const auto& row : result.rows) {
                std::string cache_key = "post:" + row.at("id");
                std::vector<std::map<std::string, std::string>> single_row = {row};
                auto serialized = serialize_rows(single_row);
                query_cache_.put(cache_key, serialized, std::chrono::minutes(30));
            }
            
            // Group results by post_id
            std::vector<QueryResult> results;
            for (uint32_t post_id : post_ids) {
                auto it = std::find_if(result.rows.begin(), result.rows.end(),
                    [post_id](const auto& row) {
                        return std::stoul(row.at("id")) == post_id;
                    });
                
                if (it != result.rows.end()) {
                    results.push_back({
                        .success = true,
                        .rows = {*it},
                        .execution_time_us = timer.elapsed_ns() / 1000,
                        .from_cache = false
                    });
                } else {
                    results.push_back({
                        .success = false,
                        .execution_time_us = 0,
                        .from_cache = false,
                        .error_message = "Post not found"
                    });
                }
            }
            
            return results;
            
        } catch (const std::exception& e) {
            // Return error for all posts
            std::vector<QueryResult> error_results;
            for (size_t i = 0; i < post_ids.size(); ++i) {
                error_results.push_back({
                    .success = false,
                    .execution_time_us = timer.elapsed_ns() / 1000,
                    .from_cache = false,
                    .error_message = e.what()
                });
            }
            return error_results;
        }
    }
    
    std::string serialize_rows(const std::vector<std::map<std::string, std::string>>& rows) {
        rapidjson::Document doc;
        doc.SetArray();
        auto& allocator = doc.GetAllocator();
        
        for (const auto& row : rows) {
            rapidjson::Value row_obj(rapidjson::kObjectType);
            for (const auto& [key, value] : row) {
                row_obj.AddMember(
                    rapidjson::Value(key.c_str(), allocator),
                    rapidjson::Value(value.c_str(), allocator),
                    allocator
                );
            }
            doc.PushBack(row_obj, allocator);
        }
        
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        
        return buffer.GetString();
    }
    
    std::vector<std::map<std::string, std::string>> deserialize_rows(const std::string& json) {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        
        std::vector<std::map<std::string, std::string>> rows;
        
        for (const auto& row_val : doc.GetArray()) {
            std::map<std::string, std::string> row;
            for (const auto& member : row_val.GetObject()) {
                row[member.name.GetString()] = member.value.GetString();
            }
            rows.push_back(row);
        }
        
        return rows;
    }
};
```

**Node.js Database Service Integration:**
```javascript
// Enhanced database service with C++ acceleration
class DatabaseService {
    constructor() {
        this.cppDbUrl = 'http://localhost:8084';
        this.fallbackEnabled = true;
        this.cppHealthy = true;
        
        this.setupHealthCheck();
    }
    
    async getPostById(postId) {
        if (this.cppHealthy) {
            try {
                const response = await fetch(`${this.cppDbUrl}/posts/${postId}`, {
                    timeout: 50 // 50ms timeout for ultra-fast response
                });
                
                if (response.ok) {
                    const result = await response.json();
                    return {
                        ...result,
                        source: 'cpp',
                        cached: result.from_cache
                    };
                }
            } catch (error) {
                console.warn('C++ database service failed, falling back to Node.js:', error.message);
            }
        }
        
        // Fallback to traditional MongoDB query
        const post = await Post.findById(postId);
        return {
            ...post.toObject(),
            source: 'nodejs',
            cached: false
        };
    }
    
    async getUserPosts(userId, limit = 20, offset = 0) {
        if (this.cppHealthy) {
            try {
                const response = await fetch(
                    `${this.cppDbUrl}/users/${userId}/posts?limit=${limit}&offset=${offset}`,
                    { timeout: 100 }
                );
                
                if (response.ok) {
                    const result = await response.json();
                    return {
                        posts: result.rows,
                        source: 'cpp',
                        cached: result.from_cache,
                        execution_time_us: result.execution_time_us
                    };
                }
            } catch (error) {
                console.warn('C++ database service failed for user posts:', error.message);
            }
        }
        
        // Fallback to MongoDB
        const posts = await Post.find({ author_id: userId })
            .sort({ created_at: -1 })
            .limit(limit)
            .skip(offset);
        
        return {
            posts: posts.map(p => p.toObject()),
            source: 'nodejs',
            cached: false
        };
    }
    
    async getPostsBatch(postIds) {
        if (this.cppHealthy && postIds.length > 1) {
            try {
                const response = await fetch(`${this.cppDbUrl}/posts/batch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ post_ids: postIds }),
                    timeout: 200
                });
                
                if (response.ok) {
                    const results = await response.json();
                    return results.map(result => ({
                        ...result,
                        source: 'cpp'
                    }));
                }
            } catch (error) {
                console.warn('C++ batch query failed:', error.message);
            }
        }
        
        // Fallback to individual MongoDB queries
        const posts = await Post.find({ _id: { $in: postIds } });
        return posts.map(post => ({
            rows: [post.toObject()],
            success: true,
            source: 'nodejs',
            cached: false
        }));
    }
    
    async invalidatePostCache(postId) {
        if (this.cppHealthy) {
            try {
                await fetch(`${this.cppDbUrl}/cache/invalidate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        pattern: `post:${postId}*` 
                    }),
                    timeout: 1000
                });
            } catch (error) {
                console.error('Failed to invalidate C++ cache:', error.message);
            }
        }
    }
    
    async getDatabaseStats() {
        try {
            const response = await fetch(`${this.cppDbUrl}/stats`, {
                timeout: 1000
            });
            
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Failed to get database stats:', error.message);
        }
        
        return null;
    }
    
    setupHealthCheck() {
        setInterval(async () => {
            try {
                const response = await fetch(`${this.cppDbUrl}/health`, {
                    timeout: 1000
                });
                this.cppHealthy = response.ok;
            } catch (error) {
                this.cppHealthy = false;
            }
        }, 5000);
    }
}

// Usage in routes
const dbService = new DatabaseService();

app.get('/api/posts/:id', async (req, res) => {
    try {
        const post = await dbService.getPostById(req.params.id);
        
        res.set('X-Data-Source', post.source);
        if (post.cached) {
            res.set('X-Cache', 'HIT');
        }
        
        res.json(post);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/users/:id/posts', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 20;
        const offset = parseInt(req.query.offset) || 0;
        
        const result = await dbService.getUserPosts(req.params.id, limit, offset);
        
        res.set('X-Data-Source', result.source);
        res.set('X-Execution-Time', `${result.execution_time_us || 0}μs`);
        
        res.json({
            posts: result.posts,
            pagination: {
                limit,
                offset,
                has_more: result.posts.length === limit
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Cache invalidation on post updates
app.put('/api/posts/:id', async (req, res) => {
    try {
        const updatedPost = await Post.findByIdAndUpdate(req.params.id, req.body);
        
        // Invalidate C++ cache
        await dbService.invalidatePostCache(req.params.id);
        
        res.json(updatedPost);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```

This comprehensive integration examples document demonstrates how to effectively integrate the Ultra Low-Latency C++ System with existing Node.js infrastructure across various use cases, providing significant performance improvements while maintaining system reliability and fallback capabilities.