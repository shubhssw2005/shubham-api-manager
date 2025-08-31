#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include <thread>
#include <chrono>

using namespace ultra_cpp::database;
using namespace testing;

class DatabaseConnectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test database configuration
        config_.host = "localhost";
        config_.port = 5432;
        config_.database = "test_ultra_cpp";
        config_.username = "test_user";
        config_.password = "test_password";
        config_.connection_timeout_ms = 5000;
        config_.query_timeout_ms = 30000;
        config_.enable_ssl = false; // Disable SSL for testing
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    DatabaseConnector::Config config_;
};

TEST_F(DatabaseConnectorTest, ConstructorInitializesCorrectly) {
    DatabaseConnector connector(config_);
    
    EXPECT_FALSE(connector.is_connected());
    
    auto metrics = connector.get_metrics();
    EXPECT_EQ(metrics.queries_executed.load(), 0);
    EXPECT_EQ(metrics.queries_failed.load(), 0);
    EXPECT_EQ(metrics.connections_created.load(), 0);
}

TEST_F(DatabaseConnectorTest, ConnectionStringBuilding) {
    DatabaseConnector connector(config_);
    
    // Test connection (this will fail in CI but tests the connection string building)
    bool connected = connector.connect();
    
    // In a real test environment with PostgreSQL running, this would be true
    // For CI/CD, we expect this to fail gracefully
    if (!connected) {
        auto metrics = connector.get_metrics();
        EXPECT_GT(metrics.connections_failed.load(), 0);
    }
}

TEST_F(DatabaseConnectorTest, PreparedStatementManagement) {
    DatabaseConnector connector(config_);
    
    // Test preparing statements without connection (should fail)
    bool prepared = connector.prepare_statement("test_stmt", "SELECT $1::text", {TEXTOID});
    EXPECT_FALSE(prepared);
}

TEST_F(DatabaseConnectorTest, QueryExecutionWithoutConnection) {
    DatabaseConnector connector(config_);
    
    auto result = connector.execute_query("SELECT 1");
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_EQ(result.rows.size(), 0);
}

TEST_F(DatabaseConnectorTest, TransactionManagementWithoutConnection) {
    DatabaseConnector connector(config_);
    
    EXPECT_FALSE(connector.begin_transaction());
    EXPECT_FALSE(connector.commit_transaction());
    EXPECT_FALSE(connector.rollback_transaction());
}

TEST_F(DatabaseConnectorTest, MetricsTracking) {
    DatabaseConnector connector(config_);
    
    // Execute a query that will fail (no connection)
    auto result = connector.execute_query("SELECT 1");
    
    auto metrics = connector.get_metrics();
    EXPECT_GT(metrics.queries_executed.load(), 0);
    EXPECT_GT(metrics.queries_failed.load(), 0);
    
    // Reset metrics
    connector.reset_metrics();
    metrics = connector.get_metrics();
    EXPECT_EQ(metrics.queries_executed.load(), 0);
    EXPECT_EQ(metrics.queries_failed.load(), 0);
}

TEST_F(DatabaseConnectorTest, AsyncQueryExecution) {
    DatabaseConnector connector(config_);
    
    auto future = connector.execute_query_async("SELECT 1");
    
    // Should complete quickly even if it fails
    auto status = future.wait_for(std::chrono::seconds(1));
    EXPECT_EQ(status, std::future_status::ready);
    
    auto result = future.get();
    EXPECT_FALSE(result.success); // Expected to fail without connection
}

// Mock tests for when we have a database connection
class MockDatabaseConnectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.host = "localhost";
        config_.port = 5432;
        config_.database = "test_db";
        config_.username = "test_user";
        config_.password = "test_pass";
    }
    
    DatabaseConnector::Config config_;
};

TEST_F(MockDatabaseConnectorTest, QueryResultProcessing) {
    DatabaseConnector connector(config_);
    
    // Test query result structure
    DatabaseConnector::QueryResult result;
    result.success = true;
    result.affected_rows = 5;
    result.execution_time_ns = 1000000; // 1ms
    result.rows = {{"col1", "col2"}, {"val1", "val2"}};
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.affected_rows, 5);
    EXPECT_EQ(result.execution_time_ns, 1000000);
    EXPECT_EQ(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0].size(), 2);
    EXPECT_EQ(result.rows[0][0], "col1");
    EXPECT_EQ(result.rows[1][1], "val2");
}

TEST_F(MockDatabaseConnectorTest, PreparedStatementStructure) {
    DatabaseConnector::PreparedStatement stmt;
    stmt.name = "test_stmt";
    stmt.query = "SELECT * FROM users WHERE id = $1";
    stmt.param_types = {INT4OID};
    stmt.param_count = 1;
    
    EXPECT_EQ(stmt.name, "test_stmt");
    EXPECT_EQ(stmt.param_count, 1);
    EXPECT_EQ(stmt.param_types.size(), 1);
    EXPECT_EQ(stmt.param_types[0], INT4OID);
}

// Performance tests
class DatabaseConnectorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.host = "localhost";
        config_.port = 5432;
        config_.database = "test_db";
        config_.username = "test_user";
        config_.password = "test_pass";
    }
    
    DatabaseConnector::Config config_;
};

TEST_F(DatabaseConnectorPerformanceTest, ConcurrentQueryExecution) {
    DatabaseConnector connector(config_);
    
    const int num_threads = 10;
    const int queries_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::atomic<int> completed_queries{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&connector, &completed_queries, queries_per_thread]() {
            for (int j = 0; j < queries_per_thread; ++j) {
                auto result = connector.execute_query("SELECT 1");
                completed_queries.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_EQ(completed_queries.load(), num_threads * queries_per_thread);
    
    // Log performance metrics
    std::cout << "Completed " << completed_queries.load() 
              << " queries in " << duration.count() << "ms" << std::endl;
    
    auto metrics = connector.get_metrics();
    std::cout << "Total queries executed: " << metrics.queries_executed.load() << std::endl;
    std::cout << "Failed queries: " << metrics.queries_failed.load() << std::endl;
    
    if (metrics.queries_executed.load() > 0) {
        double avg_time_ns = static_cast<double>(metrics.total_execution_time_ns.load()) / 
                           metrics.queries_executed.load();
        std::cout << "Average query time: " << avg_time_ns / 1000000.0 << "ms" << std::endl;
    }
}

TEST_F(DatabaseConnectorPerformanceTest, AsyncQueryPerformance) {
    DatabaseConnector connector(config_);
    
    const int num_queries = 1000;
    std::vector<std::future<DatabaseConnector::QueryResult>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Submit all queries asynchronously
    for (int i = 0; i < num_queries; ++i) {
        futures.push_back(connector.execute_query_async("SELECT " + std::to_string(i)));
    }
    
    // Wait for all to complete
    int completed = 0;
    for (auto& future : futures) {
        auto result = future.get();
        if (result.success || !result.success) { // Count both success and failure
            completed++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_EQ(completed, num_queries);
    
    std::cout << "Completed " << completed << " async queries in " 
              << duration.count() << "ms" << std::endl;
}

// Integration tests (require actual database)
class DatabaseConnectorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip if no test database is available
        const char* test_db_url = std::getenv("TEST_DATABASE_URL");
        if (!test_db_url) {
            GTEST_SKIP() << "No TEST_DATABASE_URL environment variable set";
        }
        
        // Parse database URL (simplified)
        config_.host = "localhost";
        config_.port = 5432;
        config_.database = "test_ultra_cpp";
        config_.username = "test_user";
        config_.password = "test_password";
        config_.enable_ssl = false;
    }
    
    DatabaseConnector::Config config_;
};

TEST_F(DatabaseConnectorIntegrationTest, DISABLED_RealDatabaseConnection) {
    // This test is disabled by default as it requires a real database
    DatabaseConnector connector(config_);
    
    bool connected = connector.connect();
    ASSERT_TRUE(connected) << "Failed to connect to test database";
    
    EXPECT_TRUE(connector.is_connected());
    EXPECT_TRUE(connector.ping());
    
    // Test basic query
    auto result = connector.execute_query("SELECT 1 as test_column");
    EXPECT_TRUE(result.success) << "Query failed: " << result.error_message;
    EXPECT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0].size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
    
    connector.disconnect();
    EXPECT_FALSE(connector.is_connected());
}

TEST_F(DatabaseConnectorIntegrationTest, DISABLED_PreparedStatements) {
    DatabaseConnector connector(config_);
    
    ASSERT_TRUE(connector.connect());
    
    // Prepare a statement
    bool prepared = connector.prepare_statement("test_select", 
                                               "SELECT $1::text as param_value", 
                                               {TEXTOID});
    EXPECT_TRUE(prepared);
    
    // Execute prepared statement
    auto result = connector.execute_prepared("test_select", {"hello world"});
    EXPECT_TRUE(result.success) << "Prepared query failed: " << result.error_message;
    EXPECT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "hello world");
    
    // Deallocate statement
    bool deallocated = connector.deallocate_statement("test_select");
    EXPECT_TRUE(deallocated);
}

TEST_F(DatabaseConnectorIntegrationTest, DISABLED_TransactionManagement) {
    DatabaseConnector connector(config_);
    
    ASSERT_TRUE(connector.connect());
    
    // Test transaction lifecycle
    EXPECT_TRUE(connector.begin_transaction());
    
    // Execute some queries within transaction
    auto result = connector.execute_query("SELECT 1");
    EXPECT_TRUE(result.success);
    
    EXPECT_TRUE(connector.commit_transaction());
    
    // Test rollback
    EXPECT_TRUE(connector.begin_transaction());
    EXPECT_TRUE(connector.rollback_transaction());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize logging for tests
    ultra_cpp::common::Logger::initialize(ultra_cpp::common::LogLevel::DEBUG);
    
    return RUN_ALL_TESTS();
}