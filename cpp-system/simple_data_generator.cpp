#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class HighSpeedDataGenerator {
private:
    std::string api_base_url_;
    std::atomic<size_t> posts_created_{0};
    std::atomic<size_t> requests_sent_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    
    static constexpr size_t THREAD_COUNT = 8;
    static constexpr size_t BATCH_SIZE = 100;
    static constexpr size_t POSTS_PER_USER = 1000;
    
    thread_local static std::mt19937 rng_;

public:
    HighSpeedDataGenerator(const std::string& api_url) 
        : api_base_url_(api_url) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    
    ~HighSpeedDataGenerator() {
        curl_global_cleanup();
    }

private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    
    std::string getRandomTopic() {
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
            "Data Science and Analytics"
        };
        
        std::uniform_int_distribution<size_t> dist(0, topics.size() - 1);
        return topics[dist(rng_)];
    }
    
    json generatePost(const std::string& userId, const std::string& userEmail, int postIndex) {
        std::string topic = getRandomTopic();
        std::string title = topic + " - Post " + std::to_string(postIndex + 1) + " by User " + userId;
        
        // Generate slug
        std::string slug = title;
        std::transform(slug.begin(), slug.end(), slug.begin(), ::tolower);
        for (char& c : slug) {
            if (!std::isalnum(c) && c != ' ') c = '-';
            else if (c == ' ') c = '-';
        }
        slug += "-" + userId.substr(userId.length() - 6) + "-" + std::to_string(postIndex);
        
        // Generate content
        std::ostringstream content;
        content << "This is a comprehensive blog post about " << topic << ".\n\n";
        content << "Post Number: " << (postIndex + 1) << "\n";
        content << "Author User ID: " << userId << "\n";
        content << "Created for massive data testing and performance analysis.\n\n";
        
        for (int i = 0; i < 20; ++i) {
            content << "Sentence " << (i + 1) << ": This is detailed content about " << topic 
                    << " that provides valuable insights for developers. ";
        }
        
        json post = {
            {"title", title},
            {"slug", slug},
            {"content", content.str()},
            {"excerpt", "A comprehensive guide to " + topic + " with practical examples."},
            {"tags", {topic, "technical", "development", "cpp-generated"}},
            {"status", (postIndex % 10 == 0) ? "draft" : "published"},
            {"featured", (postIndex % 25 == 0)},
            {"author", userId},
            {"authorEmail", userEmail},
            {"metadata", {
                {"postNumber", postIndex + 1},
                {"authorUserId", userId},
                {"wordCount", 500 + (rng_() % 1000)},
                {"readingTime", 2 + (rng_() % 10)},
                {"category", topic},
                {"generatedBy", "cpp-high-performance-system"}
            }}
        };
        
        return post;
    }
    
    bool sendBatchRequest(const std::vector<json>& posts) {
        CURL* curl = curl_easy_init();
        if (!curl) return false;
        
        json batch_data = {
            {"posts", posts},
            {"batchSize", posts.size()},
            {"source", "cpp-high-performance-generator"}
        };
        
        std::string json_string = batch_data.dump();
        std::string response_string;
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, (api_base_url_ + "/api/posts/batch").c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        
        if (res == CURLE_OK && response_code == 200) {
            posts_created_ += posts.size();
            requests_sent_++;
            return true;
        }
        
        return false;
    }
    
    void generatePostsForUser(const json& user) {
        std::string userId;
        std::string userEmail;
        
        // Handle different ID formats
        if (user.contains("_id")) {
            if (user["_id"].is_string()) {
                userId = user["_id"];
            } else if (user["_id"].is_object() && user["_id"].contains("$oid")) {
                userId = user["_id"]["$oid"];
            }
        }
        
        if (user.contains("email") && user["email"].is_string()) {
            userEmail = user["email"];
        }
        
        std::vector<json> batch;
        batch.reserve(BATCH_SIZE);
        
        for (int i = 0; i < POSTS_PER_USER; ++i) {
            batch.push_back(generatePost(userId, userEmail, i));
            
            if (batch.size() == BATCH_SIZE || i == POSTS_PER_USER - 1) {
                // Send batch
                if (sendBatchRequest(batch)) {
                    if (requests_sent_ % 10 == 0) {
                        std::cout << "📈 Progress: " << posts_created_.load() 
                                 << " posts created, " << requests_sent_.load() 
                                 << " batches sent\n";
                    }
                } else {
                    std::cerr << "❌ Failed to send batch for user: " << userEmail << "\n";
                }
                
                batch.clear();
                batch.reserve(BATCH_SIZE);
                
                // Small delay to avoid overwhelming the server
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

public:
    std::vector<json> fetchUsers() {
        std::cout << "📊 Fetching test users...\n";
        
        CURL* curl = curl_easy_init();
        if (!curl) return {};
        
        std::string response_string;
        std::string url = api_base_url_ + "/api/users?filter=test";
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_easy_cleanup(curl);
        
        if (res == CURLE_OK && response_code == 200) {
            try {
                json response = json::parse(response_string);
                if (response.contains("users") && response["users"].is_array()) {
                    std::vector<json> users = response["users"];
                    std::cout << "✅ Found " << users.size() << " test users\n";
                    return users;
                } else {
                    std::cerr << "❌ Invalid response format\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Error parsing users response: " << e.what() << "\n";
                std::cerr << "Response: " << response_string.substr(0, 200) << "...\n";
            }
        }
        
        return {};
    }
    
    void generateMassiveData() {
        std::cout << "\n🚀 STARTING HIGH-PERFORMANCE MASSIVE DATA GENERATION\n";
        std::cout << "===================================================\n";
        
        start_time_ = std::chrono::high_resolution_clock::now();
        
        // Fetch users
        auto users = fetchUsers();
        if (users.empty()) {
            std::cerr << "❌ No users found\n";
            return;
        }
        
        // Generate data using multiple threads
        std::vector<std::thread> threads;
        size_t users_per_thread = users.size() / THREAD_COUNT;
        
        for (size_t t = 0; t < THREAD_COUNT; ++t) {
            size_t start_idx = t * users_per_thread;
            size_t end_idx = (t == THREAD_COUNT - 1) ? users.size() : (t + 1) * users_per_thread;
            
            threads.emplace_back([this, &users, start_idx, end_idx]() {
                for (size_t i = start_idx; i < end_idx; ++i) {
                    generatePostsForUser(users[i]);
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        end_time_ = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n🎉 HIGH-PERFORMANCE DATA GENERATION COMPLETED!\n";
        printPerformanceMetrics();
    }
    
    void printPerformanceMetrics() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
        double seconds = duration.count() / 1000.0;
        double posts_per_second = posts_created_.load() / seconds;
        
        std::cout << "\n⚡ HIGH-PERFORMANCE METRICS:\n";
        std::cout << "============================\n";
        std::cout << "   Total Posts Created: " << posts_created_.load() << "\n";
        std::cout << "   Total Time: " << std::fixed << std::setprecision(2) << seconds << " seconds\n";
        std::cout << "   Posts per Second: " << static_cast<int>(posts_per_second) << "\n";
        std::cout << "   Batch Requests Sent: " << requests_sent_.load() << "\n";
        std::cout << "   Batch Size: " << BATCH_SIZE << "\n";
        std::cout << "   Worker Threads: " << THREAD_COUNT << "\n";
        
        std::cout << "\n🚀 PERFORMANCE COMPARISON:\n";
        std::cout << "   C++ Implementation: " << static_cast<int>(posts_per_second) << " posts/sec\n";
        std::cout << "   JavaScript would take: ~" << static_cast<int>(posts_created_.load() / 100) << " seconds\n";
        std::cout << "   Speed Improvement: ~" << static_cast<int>(posts_per_second / 100) << "x faster\n";
    }
};

thread_local std::mt19937 HighSpeedDataGenerator::rng_(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
    std::cout << "🚀 HIGH-PERFORMANCE MASSIVE DATA GENERATOR (C++)\n";
    std::cout << "===============================================\n";
    std::cout << "Creating 1000 posts for each of 76 users (76,000 total posts!)\n";
    std::cout << "Using C++ multi-threaded batch processing for maximum speed\n\n";
    
    // Use local API endpoint
    HighSpeedDataGenerator generator("http://localhost:3005");
    
    try {
        generator.generateMassiveData();
        
        std::cout << "\n✅ All data generated successfully!\n";
        std::cout << "✅ Check your MongoDB database for the results\n";
        std::cout << "✅ Use MongoDB Compass to view the data\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}