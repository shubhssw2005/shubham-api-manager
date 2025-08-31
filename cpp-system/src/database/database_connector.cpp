#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include "common/error_handling.hpp"
#include <sstream>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace ultra_cpp {
namespace database {

DatabaseConnector::DatabaseConnector(const Config& config)
    : config_(config), connection_(nullptr), io_manager_(nullptr) {
    
    // Initialize io_uring manager for async operations
    IOUringManager::Config io_config;
    io_config.queue_depth = 128;
    io_config.worker_threads = 2;
    io_manager_ = std::make_unique<IOUringManager>(io_config);
    
    if (!io_manager_->initialize()) {
        LOG_ERROR("Failed to initialize io_uring manager");
    }
}

DatabaseConnector::~DatabaseConnector() {
    disconnect();
    if (io_manager_) {
        io_manager_->shutdown();
    }
}

bool DatabaseConnector::connect() {
    if (connected_.load()) {
        return true;
    }

    auto start_time = std::chrono::steady_clock::now();
    
    try {
        std::string conn_str = build_connection_string();
        connection_ = PQconnectdb(conn_str.c_str());
        
        if (PQstatus(connection_) != CONNECTION_OK) {
            std::string error = PQerrorMessage(connection_);
            LOG_ERROR("Database connection failed: {}", error);
            
            PQfinish(connection_);
            connection_ = nullptr;
            metrics_.connections_failed.fetch_add(1);
            return false;
        }

        // Set connection parameters for optimal performance
        PQexec(connection_, "SET synchronous_commit = off");
        PQexec(connection_, "SET wal_writer_delay = 10ms");
        PQexec(connection_, "SET checkpoint_completion_target = 0.9");
        
        connected_.store(true);
        metrics_.connections_created.fetch_add(1);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        LOG_INFO("Database connection established in {}ns", duration.count());
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during database connection: {}", e.what());
        metrics_.connections_failed.fetch_add(1);
        return false;
    }
}

void DatabaseConnector::disconnect() {
    if (connection_) {
        // Rollback any pending transaction
        if (in_transaction_.load()) {
            rollback_transaction();
        }
        
        PQfinish(connection_);
        connection_ = nullptr;
        connected_.store(false);
        in_transaction_.store(false);
        
        LOG_INFO("Database connection closed");
    }
}

bool DatabaseConnector::is_connected() const noexcept {
    return connected_.load() && connection_ && PQstatus(connection_) == CONNECTION_OK;
}

bool DatabaseConnector::ping() {
    if (!is_connected()) {
        return false;
    }
    
    PGresult* result = PQexec(connection_, "SELECT 1");
    bool success = (PQresultStatus(result) == PGRES_TUPLES_OK);
    PQclear(result);
    
    if (!success) {
        LOG_WARN("Database ping failed, connection may be stale");
        connected_.store(false);
    }
    
    return success;
}

bool DatabaseConnector::prepare_statement(const std::string& name, const std::string& query, 
                                         const std::vector<Oid>& param_types) {
    if (!is_connected()) {
        LOG_ERROR("Cannot prepare statement: not connected to database");
        return false;
    }
    
    // Check if statement already exists
    if (prepared_statements_.find(name) != prepared_statements_.end()) {
        LOG_WARN("Statement '{}' already prepared, skipping", name);
        return true;
    }
    
    PGresult* result = PQprepare(connection_, name.c_str(), query.c_str(), 
                                param_types.size(), param_types.empty() ? nullptr : param_types.data());
    
    if (PQresultStatus(result) != PGRES_COMMAND_OK) {
        std::string error = PQerrorMessage(connection_);
        LOG_ERROR("Failed to prepare statement '{}': {}", name, error);
        PQclear(result);
        return false;
    }
    
    PQclear(result);
    
    // Store prepared statement info
    PreparedStatement stmt;
    stmt.name = name;
    stmt.query = query;
    stmt.param_types = param_types;
    stmt.param_count = param_types.size();
    
    prepared_statements_[name] = std::move(stmt);
    
    LOG_DEBUG("Prepared statement '{}' with {} parameters", name, param_types.size());
    return true;
}

bool DatabaseConnector::deallocate_statement(const std::string& name) {
    if (!is_connected()) {
        return false;
    }
    
    auto it = prepared_statements_.find(name);
    if (it == prepared_statements_.end()) {
        LOG_WARN("Statement '{}' not found for deallocation", name);
        return false;
    }
    
    std::string deallocate_query = "DEALLOCATE " + name;
    PGresult* result = PQexec(connection_, deallocate_query.c_str());
    
    bool success = (PQresultStatus(result) == PGRES_COMMAND_OK);
    PQclear(result);
    
    if (success) {
        prepared_statements_.erase(it);
        LOG_DEBUG("Deallocated prepared statement '{}'", name);
    } else {
        LOG_ERROR("Failed to deallocate statement '{}'", name);
    }
    
    return success;
}

DatabaseConnector::QueryResult DatabaseConnector::execute_query(const std::string& query) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_connected()) {
        LOG_ERROR("Cannot execute query: not connected to database");
        return {false, "Not connected to database", {}, 0, 0};
    }
    
    PGresult* result = PQexec(connection_, query.c_str());
    auto query_result = process_result(result, 
        std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count());
    
    PQclear(result);
    
    metrics_.queries_executed.fetch_add(1);
    if (!query_result.success) {
        metrics_.queries_failed.fetch_add(1);
    }
    metrics_.total_execution_time_ns.fetch_add(query_result.execution_time_ns);
    
    return query_result;
}

DatabaseConnector::QueryResult DatabaseConnector::execute_prepared(const std::string& statement_name, 
                                                                  const std::vector<std::string>& params) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_connected()) {
        LOG_ERROR("Cannot execute prepared statement: not connected to database");
        return {false, "Not connected to database", {}, 0, 0};
    }
    
    auto it = prepared_statements_.find(statement_name);
    if (it == prepared_statements_.end()) {
        LOG_ERROR("Prepared statement '{}' not found", statement_name);
        return {false, "Prepared statement not found", {}, 0, 0};
    }
    
    const auto& stmt = it->second;
    if (params.size() != stmt.param_count) {
        LOG_ERROR("Parameter count mismatch for statement '{}': expected {}, got {}", 
                 statement_name, stmt.param_count, params.size());
        return {false, "Parameter count mismatch", {}, 0, 0};
    }
    
    // Convert parameters to C-style arrays
    std::vector<const char*> param_values;
    std::vector<int> param_lengths;
    std::vector<int> param_formats;
    
    param_values.reserve(params.size());
    param_lengths.reserve(params.size());
    param_formats.reserve(params.size());
    
    for (const auto& param : params) {
        param_values.push_back(param.c_str());
        param_lengths.push_back(param.length());
        param_formats.push_back(0); // Text format
    }
    
    PGresult* result = PQexecPrepared(connection_, statement_name.c_str(), params.size(),
                                     param_values.data(), param_lengths.data(), 
                                     param_formats.data(), 0);
    
    auto query_result = process_result(result, 
        std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count());
    
    PQclear(result);
    
    metrics_.queries_executed.fetch_add(1);
    if (!query_result.success) {
        metrics_.queries_failed.fetch_add(1);
    }
    metrics_.total_execution_time_ns.fetch_add(query_result.execution_time_ns);
    
    return query_result;
}

std::future<DatabaseConnector::QueryResult> DatabaseConnector::execute_query_async(const std::string& query) {
    if (!io_manager_) {
        // Fallback to synchronous execution
        std::promise<QueryResult> promise;
        auto future = promise.get_future();
        promise.set_value(execute_query(query));
        return future;
    }
    
    return io_manager_->submit_query(connection_, query);
}

std::future<DatabaseConnector::QueryResult> DatabaseConnector::execute_prepared_async(
    const std::string& statement_name, const std::vector<std::string>& params) {
    
    if (!io_manager_) {
        // Fallback to synchronous execution
        std::promise<QueryResult> promise;
        auto future = promise.get_future();
        promise.set_value(execute_prepared(statement_name, params));
        return future;
    }
    
    return io_manager_->submit_prepared_query(connection_, statement_name, params);
}

bool DatabaseConnector::begin_transaction() {
    if (!is_connected()) {
        LOG_ERROR("Cannot begin transaction: not connected to database");
        return false;
    }
    
    if (in_transaction_.load()) {
        LOG_WARN("Transaction already in progress");
        return true;
    }
    
    PGresult* result = PQexec(connection_, "BEGIN");
    bool success = (PQresultStatus(result) == PGRES_COMMAND_OK);
    PQclear(result);
    
    if (success) {
        in_transaction_.store(true);
        LOG_DEBUG("Transaction started");
    } else {
        LOG_ERROR("Failed to begin transaction: {}", PQerrorMessage(connection_));
    }
    
    return success;
}

bool DatabaseConnector::commit_transaction() {
    if (!is_connected()) {
        LOG_ERROR("Cannot commit transaction: not connected to database");
        return false;
    }
    
    if (!in_transaction_.load()) {
        LOG_WARN("No transaction in progress");
        return true;
    }
    
    PGresult* result = PQexec(connection_, "COMMIT");
    bool success = (PQresultStatus(result) == PGRES_COMMAND_OK);
    PQclear(result);
    
    if (success) {
        in_transaction_.store(false);
        LOG_DEBUG("Transaction committed");
    } else {
        LOG_ERROR("Failed to commit transaction: {}", PQerrorMessage(connection_));
    }
    
    return success;
}

bool DatabaseConnector::rollback_transaction() {
    if (!is_connected()) {
        LOG_ERROR("Cannot rollback transaction: not connected to database");
        return false;
    }
    
    if (!in_transaction_.load()) {
        LOG_WARN("No transaction in progress");
        return true;
    }
    
    PGresult* result = PQexec(connection_, "ROLLBACK");
    bool success = (PQresultStatus(result) == PGRES_COMMAND_OK);
    PQclear(result);
    
    if (success) {
        in_transaction_.store(false);
        LOG_DEBUG("Transaction rolled back");
    } else {
        LOG_ERROR("Failed to rollback transaction: {}", PQerrorMessage(connection_));
    }
    
    return success;
}

DatabaseConnector::Metrics DatabaseConnector::get_metrics() const noexcept {
    return metrics_;
}

void DatabaseConnector::reset_metrics() noexcept {
    metrics_.queries_executed.store(0);
    metrics_.queries_failed.store(0);
    metrics_.total_execution_time_ns.store(0);
    metrics_.cache_hits.store(0);
    metrics_.cache_misses.store(0);
    metrics_.connections_created.store(0);
    metrics_.connections_failed.store(0);
}

DatabaseConnector::QueryResult DatabaseConnector::process_result(PGresult* result, uint64_t start_time_ns) {
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time.time_since_epoch()).count() - start_time_ns;
    
    QueryResult query_result;
    query_result.execution_time_ns = execution_time_ns;
    
    if (!result) {
        query_result.success = false;
        query_result.error_message = "No result returned from query";
        return query_result;
    }
    
    ExecStatusType status = PQresultStatus(result);
    
    switch (status) {
        case PGRES_COMMAND_OK:
            query_result.success = true;
            query_result.affected_rows = std::stoull(PQcmdTuples(result));
            break;
            
        case PGRES_TUPLES_OK: {
            query_result.success = true;
            int rows = PQntuples(result);
            int cols = PQnfields(result);
            
            query_result.rows.reserve(rows);
            
            for (int row = 0; row < rows; ++row) {
                std::vector<std::string> row_data;
                row_data.reserve(cols);
                
                for (int col = 0; col < cols; ++col) {
                    if (PQgetisnull(result, row, col)) {
                        row_data.emplace_back("");
                    } else {
                        row_data.emplace_back(PQgetvalue(result, row, col));
                    }
                }
                
                query_result.rows.emplace_back(std::move(row_data));
            }
            break;
        }
        
        default:
            query_result.success = false;
            query_result.error_message = PQerrorMessage(connection_);
            LOG_ERROR("Query failed with status {}: {}", status, query_result.error_message);
            break;
    }
    
    return query_result;
}

std::string DatabaseConnector::build_connection_string() const {
    std::ostringstream oss;
    
    oss << "host=" << config_.host
        << " port=" << config_.port
        << " dbname=" << config_.database
        << " user=" << config_.username
        << " password=" << config_.password
        << " connect_timeout=" << (config_.connection_timeout_ms / 1000);
    
    if (config_.enable_ssl) {
        oss << " sslmode=" << config_.ssl_mode;
    } else {
        oss << " sslmode=disable";
    }
    
    // Performance optimizations
    oss << " application_name=ultra_cpp_connector"
        << " tcp_keepalives_idle=600"
        << " tcp_keepalives_interval=30"
        << " tcp_keepalives_count=3";
    
    return oss.str();
}

bool DatabaseConnector::validate_connection() {
    return is_connected() && ping();
}

void DatabaseConnector::handle_connection_error() {
    LOG_ERROR("Connection error detected: {}", PQerrorMessage(connection_));
    connected_.store(false);
    
    // Attempt to reset the connection
    PQreset(connection_);
    
    if (PQstatus(connection_) == CONNECTION_OK) {
        connected_.store(true);
        LOG_INFO("Connection successfully reset");
    } else {
        LOG_ERROR("Failed to reset connection: {}", PQerrorMessage(connection_));
    }
}

} // namespace database
} // namespace ultra_cpp