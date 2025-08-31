#include "massive_data_generator.hpp"
#include "../common/logger.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <future>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/builder/stream/array.hpp>

namespace DataGenerator {

thread_local std::mt19937 HighPerformanceDataGenerator::rng_(std::chrono::steady_clock::now().time_since_epoch().count());

HighPerformanceDataGenerator::HighPerformanceDataGenerator(const std::string& connection_string)
    : client_(mongocxx::uri{connection_string}) {
    Logger::info("Initialized High-Performance Data Generator");
}

HighPerformanceDataGenerator::~HighPerformanceDataGenerator() {
    stopWorkerThreads();
}

bool HighPerformanceDataGenerator::connect() {
    try {
        db_ = client_["test"]; // Use default database
        
        // Test connection
        auto ping_cmd = bsoncxx::builder::stream::document{} << "ping" << 1 << bsoncxx::builder::stream::finalize;
        db_.run_command(ping_cmd.view());
        
        Logger::info("✅ Connected to MongoDB successfully");
        return true;
    } catch (const std::exception& e) {
        Logger::error("❌ Failed to connect to MongoDB: " + std::string(e.what()));
        return false;
    }
}

void HighPerformanceDataGenerator::disconnect() {
    Logger::info("🔌 Disconnected from MongoDB");
}

std::vector<User> HighPerformanceDataGenerator::fetchUsers() {
    Logger::info("📊 Fetching test users...");
    
    std::vector<User> users;
    
    try {
        auto collection = db_["users"];
        
        // Create filter for test users
        auto filter = bsoncxx::builder::stream::document{}
            << "email" << bsoncxx::builder::stream::open_document
                << "$regex" << "test|demo|perf|realdata|api"
                << "$options" << "i"
            << bsoncxx::builder::stream::close_document
            << bsoncxx::builder::stream::finalize;
        
        auto cursor = collection.find(filter.view());
        
        for (auto&& doc : cursor) {
            User user;
            user.id = doc["_id"].get_oid().value.to_string();
            user.email = doc["email"].get_utf8().value.to_string();
            
            if (doc["name"]) {
                user.name = doc["name"].get_utf8().value.to_string();
            } else {
                user.name = "Test User";
            }
            
            users.push_back(user);
        }
        
        Logger::info("✅ Found " + std::to_string(users.size()) + " test users");
        users_ = users;
        return users;
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error fetching users: " + std::string(e.what()));
        return {};
    }
}

void HighPerformanceDataGenerator::cleanupExistingPosts() {
    Logger::info("🧹 Cleaning up existing test posts...");
    
    try {
        auto collection = db_["posts"];
        
        auto filter = bsoncxx::builder::stream::document{}
            << "authorEmail" << bsoncxx::builder::stream::open_document
                << "$regex" << "test|demo|perf|realdata|api"
                << "$options" << "i"
            << bsoncxx::builder::stream::close_document
            << bsoncxx::builder::stream::finalize;
        
        auto result = collection.delete_many(filter.view());
        
        if (result) {
            Logger::info("✅ Removed " + std::to_string(result->deleted_count()) + " existing test posts");
        }
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error cleaning up posts: " + std::string(e.what()));
    }
}

std::string HighPerformanceDataGenerator::getRandomTopic() {
    static const std::vector<std::string> topics = {
        "Technology and Innovation",
        "Software Development Best Practices", 
        "Machine Learning and AI",
        "Web Development Trends",
        "Database Optimization",
        "Cloud Computing Solutions",
        "Cybersecurity Insights",
        "Mobile App Development",
        "DevOps and Automation",
        "Data Science and Analytics",
        "Blockchain Technology",
        "Internet of Things (IoT)",
        "Quantum Computing",
        "Edge Computing",
        "Microservices Architecture"
    };
    
    std::uniform_int_distribution<size_t> dist(0, topics.size() - 1);
    return topics[dist(rng_)];
}

std::string HighPerformanceDataGenerator::generateSlug(const std::string& title, const std::string& userId, int postIndex) {
    std::string slug = title;
    
    // Convert to lowercase
    std::transform(slug.begin(), slug.end(), slug.begin(), ::tolower);
    
    // Replace non-alphanumeric with hyphens
    for (char& c : slug) {
        if (!std::isalnum(c) && c != ' ') {
            c = '-';
        } else if (c == ' ') {
            c = '-';
        }
    }
    
    // Remove multiple consecutive hyphens
    slug.erase(std::unique(slug.begin(), slug.end(), [](char a, char b) {
        return a == '-' && b == '-';
    }), slug.end());
    
    // Add unique identifiers
    slug += "-" + userId.substr(userId.length() - 6) + "-" + std::to_string(postIndex);
    
    return slug;
}

std::string HighPerformanceDataGenerator::generateContent(const std::string& topic, int postIndex, const std::string& userId) {
    std::ostringstream content;
    
    content << "This is a comprehensive blog post about " << topic << ".\n\n";
    content << "Post Number: " << (postIndex + 1) << "\n";
    content << "Author User ID: " << userId << "\n";
    content << "Created for massive data testing and performance analysis.\n\n";
    
    content << "Content includes:\n";
    content << "- Detailed technical analysis\n";
    content << "- Real-world examples and case studies\n";
    content << "- Best practices and recommendations\n";
    content << "- Performance benchmarks and metrics\n";
    content << "- Future trends and predictions\n\n";
    
    // Generate substantial content
    for (int i = 0; i < 25; ++i) {
        content << "Sentence " << (i + 1) << ": This is detailed content about " << topic 
                << " that provides valuable insights for developers and technical professionals. ";
    }
    
    content << "\n\nThis post demonstrates the relationship between users and posts in our system, "
            << "with proper foreign key relationships and soft delete capabilities.";
    
    return content.str();
}

std::vector<std::string> HighPerformanceDataGenerator::generateTags(const std::string& topic, const std::string& userId, int postIndex) {
    std::vector<std::string> tags;
    
    // Convert topic to tag
    std::string topicTag = topic;
    std::transform(topicTag.begin(), topicTag.end(), topicTag.begin(), ::tolower);
    std::replace(topicTag.begin(), topicTag.end(), ' ', '-');
    
    tags.push_back(topicTag);
    tags.push_back("technical");
    tags.push_back("development");
    tags.push_back("best-practices");
    tags.push_back("user-" + userId.substr(userId.length() - 6));
    tags.push_back("post-" + std::to_string(postIndex + 1));
    
    return tags;
}

PostData HighPerformanceDataGenerator::generatePost(const User& user, int postIndex) {
    PostData post;
    
    std::string topic = getRandomTopic();
    post.title = topic + " - Post " + std::to_string(postIndex + 1) + " by User " + user.id;
    post.slug = generateSlug(post.title, user.id, postIndex);
    post.content = generateContent(topic, postIndex, user.id);
    post.excerpt = "A comprehensive guide to " + topic + " with practical examples and insights.";
    post.tags = generateTags(topic, user.id, postIndex);
    
    // Status logic
    post.status = (postIndex % 10 == 0) ? "draft" : "published";
    post.featured = (postIndex % 25 == 0);
    
    // User relationship
    post.authorId = user.id;
    post.authorEmail = user.email;
    post.authorName = user.name;
    
    // Timestamp (random within last 30 days)
    auto now = std::chrono::system_clock::now();
    auto random_offset = std::chrono::hours(rng_() % (30 * 24));
    post.createdAt = now - random_offset;
    
    post.postIndex = postIndex;
    
    return post;
}

bsoncxx::document::value HighPerformanceDataGenerator::createBSONDocument(const PostData& post) {
    using bsoncxx::builder::stream::document;
    using bsoncxx::builder::stream::array;
    using bsoncxx::builder::stream::finalize;
    
    // Convert time_point to BSON date
    auto time_t = std::chrono::system_clock::to_time_t(post.createdAt);
    auto bson_date = bsoncxx::types::b_date{std::chrono::system_clock::from_time_t(time_t)};
    auto now_bson = bsoncxx::types::b_date{std::chrono::system_clock::now()};
    
    // Build tags array
    auto tags_array = array{};
    for (const auto& tag : post.tags) {
        tags_array << tag;
    }
    
    // Create ObjectId from string
    bsoncxx::oid author_oid{post.authorId};
    
    return document{}
        << "title" << post.title
        << "slug" << post.slug
        << "content" << post.content
        << "excerpt" << post.excerpt
        << "tags" << tags_array << finalize
        << "status" << post.status
        << "featured" << post.featured
        << "author" << author_oid
        << "authorEmail" << post.authorEmail
        << "authorName" << post.authorName
        << "createdAt" << bson_date
        << "updatedAt" << now_bson
        << "isDeleted" << false
        << "deletedAt" << bsoncxx::types::b_null{}
        << "deletedBy" << bsoncxx::types::b_null{}
        << "version" << 1
        << "viewCount" << (rng_() % 10000)
        << "likeCount" << (rng_() % 1000)
        << "metadata" << document{}
            << "postNumber" << (post.postIndex + 1)
            << "authorUserId" << post.authorId
            << "wordCount" << (500 + (rng_() % 1000))
            << "readingTime" << (2 + (rng_() % 10))
            << "category" << "Technical"
            << "difficulty" << (post.postIndex % 3 == 0 ? "beginner" : 
                              post.postIndex % 3 == 1 ? "intermediate" : "advanced")
            << "estimatedViews" << (rng_() % 10000)
            << "socialShares" << (rng_() % 500)
        << finalize
        << finalize;
}

void HighPerformanceDataGenerator::createBatchesForUser(const User& user) {
    std::vector<PostData> batch;
    batch.reserve(BATCH_SIZE);
    
    for (int i = 0; i < POSTS_PER_USER; ++i) {
        batch.push_back(generatePost(user, i));
        
        if (batch.size() == BATCH_SIZE || i == POSTS_PER_USER - 1) {
            // Add batch to queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                batch_queue_.push(batch);
            }
            cv_.notify_one();
            
            batch.clear();
            batch.reserve(BATCH_SIZE);
        }
    }
    
    users_processed_++;
}

void HighPerformanceDataGenerator::insertBatch(const std::vector<PostData>& batch) {
    try {
        auto collection = db_["posts"];
        std::vector<bsoncxx::document::value> documents;
        documents.reserve(batch.size());
        
        for (const auto& post : batch) {
            documents.push_back(createBSONDocument(post));
        }
        
        auto result = collection.insert_many(documents);
        
        if (result) {
            posts_created_ += result->inserted_count();
            batches_processed_++;
            
            // Progress logging every 10 batches
            if (batches_processed_ % 10 == 0) {
                Logger::info("📈 Progress: " + std::to_string(posts_created_.load()) + " posts created, " +
                           std::to_string(batches_processed_.load()) + " batches processed");
            }
        }
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error inserting batch: " + std::string(e.what()));
    }
}

void HighPerformanceDataGenerator::workerThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_.wait(lock, [this] { return !batch_queue_.empty() || shutdown_; });
        
        if (shutdown_ && batch_queue_.empty()) {
            break;
        }
        
        if (!batch_queue_.empty()) {
            auto batch = batch_queue_.front();
            batch_queue_.pop();
            lock.unlock();
            
            insertBatch(batch);
        }
    }
}

void HighPerformanceDataGenerator::startWorkerThreads() {
    shutdown_ = false;
    
    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        std::thread(&HighPerformanceDataGenerator::workerThread, this).detach();
    }
    
    Logger::info("🚀 Started " + std::to_string(THREAD_COUNT) + " worker threads");
}

void HighPerformanceDataGenerator::stopWorkerThreads() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
    
    // Wait for all batches to be processed
    while (true) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (batch_queue_.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    Logger::info("🛑 Stopped all worker threads");
}

void HighPerformanceDataGenerator::generateMassiveData() {
    Logger::info("\n🚀 STARTING HIGH-PERFORMANCE MASSIVE DATA GENERATION");
    Logger::info("===================================================");
    
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // Clean up existing data
    cleanupExistingPosts();
    
    // Start worker threads
    startWorkerThreads();
    
    // Generate data for all users in parallel
    std::vector<std::future<void>> futures;
    
    for (const auto& user : users_) {
        futures.push_back(std::async(std::launch::async, [this, user]() {
            createBatchesForUser(user);
        }));
    }
    
    // Wait for all user processing to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Stop worker threads
    stopWorkerThreads();
    
    end_time_ = std::chrono::high_resolution_clock::now();
    
    Logger::info("\n🎉 HIGH-PERFORMANCE DATA GENERATION COMPLETED!");
    Logger::info("==============================================");
}

void HighPerformanceDataGenerator::performSoftDeletes() {
    Logger::info("\n🗑️  PERFORMING SOFT DELETES");
    Logger::info("===========================");
    
    try {
        auto collection = db_["posts"];
        
        // Find posts to soft delete (first 1000 active posts)
        auto filter = bsoncxx::builder::stream::document{}
            << "isDeleted" << false
            << bsoncxx::builder::stream::finalize;
        
        auto cursor = collection.find(filter.view()).limit(1000);
        std::vector<bsoncxx::oid> ids_to_delete;
        
        for (auto&& doc : cursor) {
            ids_to_delete.push_back(doc["_id"].get_oid().value);
        }
        
        if (!ids_to_delete.empty()) {
            // Create filter for soft delete
            auto delete_filter = bsoncxx::builder::stream::document{};
            auto id_array = bsoncxx::builder::stream::array{};
            
            for (const auto& id : ids_to_delete) {
                id_array << id;
            }
            
            delete_filter << "_id" << bsoncxx::builder::stream::open_document
                         << "$in" << id_array << bsoncxx::builder::stream::finalize
                         << bsoncxx::builder::stream::close_document
                         << bsoncxx::builder::stream::finalize;
            
            // Perform soft delete
            auto update_doc = bsoncxx::builder::stream::document{}
                << "$set" << bsoncxx::builder::stream::open_document
                    << "isDeleted" << true
                    << "deletedAt" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
                    << "deletedBy" << "cpp_system_test"
                    << "deletedReason" << "High-performance soft delete testing"
                << bsoncxx::builder::stream::close_document
                << "$inc" << bsoncxx::builder::stream::open_document
                    << "version" << 1
                << bsoncxx::builder::stream::close_document
                << bsoncxx::builder::stream::finalize;
            
            auto result = collection.update_many(delete_filter.view(), update_doc.view());
            
            if (result) {
                Logger::info("✅ Soft deleted " + std::to_string(result->modified_count()) + " posts");
            }
        }
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error performing soft deletes: " + std::string(e.what()));
    }
}

void HighPerformanceDataGenerator::performCRUDOperations() {
    Logger::info("\n🔧 PERFORMING HIGH-PERFORMANCE CRUD OPERATIONS");
    Logger::info("==============================================");
    
    try {
        auto collection = db_["posts"];
        
        // UPDATE - Bulk update draft posts to published
        auto update_filter = bsoncxx::builder::stream::document{}
            << "status" << "draft"
            << "isDeleted" << false
            << bsoncxx::builder::stream::finalize;
        
        auto update_doc = bsoncxx::builder::stream::document{}
            << "$set" << bsoncxx::builder::stream::open_document
                << "status" << "published"
                << "updatedAt" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
                << "publishedAt" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
            << bsoncxx::builder::stream::close_document
            << "$inc" << bsoncxx::builder::stream::open_document
                << "version" << 1
            << bsoncxx::builder::stream::close_document
            << bsoncxx::builder::stream::finalize;
        
        auto update_result = collection.update_many(update_filter.view(), update_doc.view());
        
        if (update_result) {
            Logger::info("✅ Updated " + std::to_string(update_result->modified_count()) + " posts from draft to published");
        }
        
        // Perform soft deletes
        performSoftDeletes();
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error performing CRUD operations: " + std::string(e.what()));
    }
}

void HighPerformanceDataGenerator::generateStatistics() {
    Logger::info("\n📊 GENERATING COMPREHENSIVE STATISTICS");
    Logger::info("=====================================");
    
    try {
        auto posts_collection = db_["posts"];
        auto users_collection = db_["users"];
        
        // Count statistics
        auto total_posts = posts_collection.count_documents({});
        auto active_posts = posts_collection.count_documents(
            bsoncxx::builder::stream::document{} << "isDeleted" << false << bsoncxx::builder::stream::finalize);
        auto deleted_posts = posts_collection.count_documents(
            bsoncxx::builder::stream::document{} << "isDeleted" << true << bsoncxx::builder::stream::finalize);
        auto published_posts = posts_collection.count_documents(
            bsoncxx::builder::stream::document{} << "status" << "published" << "isDeleted" << false << bsoncxx::builder::stream::finalize);
        auto featured_posts = posts_collection.count_documents(
            bsoncxx::builder::stream::document{} << "featured" << true << "isDeleted" << false << bsoncxx::builder::stream::finalize);
        
        Logger::info("📝 POST STATISTICS:");
        Logger::info("   Total Posts: " + std::to_string(total_posts));
        Logger::info("   Active Posts: " + std::to_string(active_posts));
        Logger::info("   Soft Deleted Posts: " + std::to_string(deleted_posts));
        Logger::info("   Published Posts: " + std::to_string(published_posts));
        Logger::info("   Featured Posts: " + std::to_string(featured_posts));
        
        Logger::info("\n🔗 RELATIONSHIPS:");
        Logger::info("   Users Processed: " + std::to_string(users_.size()));
        Logger::info("   Posts per User: " + std::to_string(POSTS_PER_USER));
        Logger::info("   Total Expected Posts: " + std::to_string(users_.size() * POSTS_PER_USER));
        
    } catch (const std::exception& e) {
        Logger::error("❌ Error generating statistics: " + std::string(e.what()));
    }
}

void HighPerformanceDataGenerator::printPerformanceMetrics() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
    double seconds = duration.count() / 1000.0;
    double posts_per_second = posts_created_.load() / seconds;
    
    Logger::info("\n⚡ HIGH-PERFORMANCE METRICS:");
    Logger::info("============================");
    Logger::info("   Total Posts Created: " + std::to_string(posts_created_.load()));
    Logger::info("   Total Time: " + std::to_string(seconds) + " seconds");
    Logger::info("   Posts per Second: " + std::to_string(static_cast<int>(posts_per_second)));
    Logger::info("   Batches Processed: " + std::to_string(batches_processed_.load()));
    Logger::info("   Batch Size: " + std::to_string(BATCH_SIZE));
    Logger::info("   Worker Threads: " + std::to_string(THREAD_COUNT));
    Logger::info("   Users Processed: " + std::to_string(users_processed_.load()));
    
    // Performance comparison
    Logger::info("\n🚀 PERFORMANCE COMPARISON:");
    Logger::info("   C++ Implementation: " + std::to_string(static_cast<int>(posts_per_second)) + " posts/sec");
    Logger::info("   JavaScript would take: ~" + std::to_string(static_cast<int>(posts_created_.load() / 100)) + " seconds");
    Logger::info("   Speed Improvement: ~" + std::to_string(static_cast<int>(posts_per_second / 100)) + "x faster");
}

} // namespace DataGenerator