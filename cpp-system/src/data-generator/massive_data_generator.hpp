#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <random>
#include <mongocxx/client.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

namespace DataGenerator {

struct User {
    std::string id;
    std::string email;
    std::string name;
};

struct PostData {
    std::string title;
    std::string slug;
    std::string content;
    std::string excerpt;
    std::vector<std::string> tags;
    std::string status;
    bool featured;
    std::string authorId;
    std::string authorEmail;
    std::string authorName;
    std::chrono::system_clock::time_point createdAt;
    int postIndex;
};

class HighPerformanceDataGenerator {
private:
    mongocxx::client client_;
    mongocxx::database db_;
    std::vector<User> users_;
    
    // Performance optimization
    static constexpr size_t BATCH_SIZE = 1000;
    static constexpr size_t THREAD_COUNT = 8;
    static constexpr size_t POSTS_PER_USER = 1000;
    
    // Thread-safe counters
    std::atomic<size_t> posts_created_{0};
    std::atomic<size_t> users_processed_{0};
    std::atomic<size_t> batches_processed_{0};
    
    // Thread synchronization
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::queue<std::vector<PostData>> batch_queue_;
    bool shutdown_{false};
    
    // Random generators (thread-local)
    thread_local static std::mt19937 rng_;
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;

public:
    explicit HighPerformanceDataGenerator(const std::string& connection_string);
    ~HighPerformanceDataGenerator();
    
    // Main operations
    bool connect();
    void disconnect();
    
    // Data generation
    std::vector<User> fetchUsers();
    void generateMassiveData();
    
    // CRUD operations
    void performCRUDOperations();
    void performSoftDeletes();
    
    // Statistics and reporting
    void generateStatistics();
    void printPerformanceMetrics();
    
private:
    // Core generation functions
    PostData generatePost(const User& user, int postIndex);
    std::string generateSlug(const std::string& title, const std::string& userId, int postIndex);
    std::string generateContent(const std::string& topic, int postIndex, const std::string& userId);
    
    // Batch processing
    void createBatchesForUser(const User& user);
    void processBatches();
    void insertBatch(const std::vector<PostData>& batch);
    
    // Thread management
    void workerThread();
    void startWorkerThreads();
    void stopWorkerThreads();
    
    // Utility functions
    std::string getRandomTopic();
    std::vector<std::string> generateTags(const std::string& topic, const std::string& userId, int postIndex);
    bsoncxx::document::value createBSONDocument(const PostData& post);
    
    // Cleanup
    void cleanupExistingPosts();
};

} // namespace DataGenerator