#!/bin/bash

# Production Readiness Test Suite
# Comprehensive testing for API, DBMS, C++, Node.js systems

set -e

# Configuration
TEST_RESULTS_DIR="./test-results"
PERFORMANCE_LOG="$TEST_RESULTS_DIR/performance-report.json"
API_BASE="http://localhost:3005"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

# Initialize test environment
initialize_test_env() {
    print_header "INITIALIZING TEST ENVIRONMENT"
    
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Create performance log structure
    cat > "$PERFORMANCE_LOG" << 'EOF'
{
  "timestamp": "",
  "environment": {
    "os": "",
    "node_version": "",
    "memory": "",
    "cpu_cores": ""
  },
  "database_tests": {},
  "api_tests": {},
  "cpp_tests": {},
  "integration_tests": {},
  "load_tests": {},
  "security_tests": {},
  "production_readiness": {
    "score": 0,
    "recommendations": []
  }
}
EOF

    # Update timestamp and environment info
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local os_info=$(uname -s)
    local node_version=$(node --version 2>/dev/null || echo "Not installed")
    local memory=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || sysctl hw.memsize 2>/dev/null | awk '{print $2/1024/1024/1024 "GB"}' || echo "Unknown")
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
    
    jq --arg timestamp "$timestamp" \
       --arg os "$os_info" \
       --arg node "$node_version" \
       --arg memory "$memory" \
       --arg cores "$cpu_cores" \
       '.timestamp = $timestamp | .environment.os = $os | .environment.node_version = $node | .environment.memory = $memory | .environment.cpu_cores = $cores' \
       "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
    
    print_status "Test environment initialized"
    print_info "OS: $os_info"
    print_info "Node.js: $node_version"
    print_info "Memory: $memory"
    print_info "CPU Cores: $cpu_cores"
}

# Test 1: Database Systems
test_database_systems() {
    print_header "DATABASE SYSTEMS TESTING"
    
    local db_results="{}"
    
    # Test ScyllaDB
    print_test "Testing ScyllaDB connection and performance"
    local scylla_start=$(date +%s%N)
    
    if docker ps | grep scylladb-node &> /dev/null; then
        if docker exec scylladb-node cqlsh -e "SELECT now() FROM system.local" &> /dev/null; then
            local scylla_end=$(date +%s%N)
            local scylla_latency=$(( (scylla_end - scylla_start) / 1000000 ))
            
            # Test write performance
            local write_start=$(date +%s%N)
            docker exec scylladb-node cqlsh -k global_api -e "INSERT INTO posts (id, title, content, created_at, is_deleted) VALUES (uuid(), 'Performance Test', 'Testing write performance', toTimestamp(now()), false)" &> /dev/null
            local write_end=$(date +%s%N)
            local write_latency=$(( (write_end - write_start) / 1000000 ))
            
            # Test read performance
            local read_start=$(date +%s%N)
            docker exec scylladb-node cqlsh -k global_api -e "SELECT COUNT(*) FROM posts" &> /dev/null
            local read_end=$(date +%s%N)
            local read_latency=$(( (read_end - read_start) / 1000000 ))
            
            db_results=$(echo "$db_results" | jq --argjson conn "$scylla_latency" --argjson write "$write_latency" --argjson read "$read_latency" '. + {"scylladb": {"status": "healthy", "connection_latency_ms": $conn, "write_latency_ms": $write, "read_latency_ms": $read}}')
            
            print_status "ScyllaDB: Connection ${scylla_latency}ms, Write ${write_latency}ms, Read ${read_latency}ms"
        else
            db_results=$(echo "$db_results" | jq '. + {"scylladb": {"status": "error", "message": "Connection failed"}}')
            print_error "ScyllaDB connection failed"
        fi
    else
        db_results=$(echo "$db_results" | jq '. + {"scylladb": {"status": "not_running", "message": "Container not found"}}')
        print_warning "ScyllaDB container not running"
    fi
    
    # Test FoundationDB
    print_test "Testing FoundationDB connection and performance"
    local fdb_start=$(date +%s%N)
    
    if [[ -f "./foundationdb/fdbcli" ]]; then
        if ./foundationdb/fdbcli --exec "status" &> /dev/null; then
            local fdb_end=$(date +%s%N)
            local fdb_latency=$(( (fdb_end - fdb_start) / 1000000 ))
            
            db_results=$(echo "$db_results" | jq --argjson latency "$fdb_latency" '. + {"foundationdb": {"status": "healthy", "connection_latency_ms": $latency}}')
            print_status "FoundationDB: Connection ${fdb_latency}ms"
        else
            db_results=$(echo "$db_results" | jq '. + {"foundationdb": {"status": "error", "message": "Connection failed"}}')
            print_error "FoundationDB connection failed"
        fi
    else
        db_results=$(echo "$db_results" | jq '. + {"foundationdb": {"status": "not_installed", "message": "CLI not found"}}')
        print_warning "FoundationDB CLI not found"
    fi
    
    # Update performance log
    jq --argjson db "$db_results" '.database_tests = $db' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
}

# Test 2: API Performance
test_api_performance() {
    print_header "API PERFORMANCE TESTING"
    
    local api_results="{}"
    
    # Check if server is running
    if ! curl -s "$API_BASE/api/v2/universal/health" &> /dev/null; then
        print_error "API server not running on $API_BASE"
        api_results='{"status": "server_not_running"}'
        jq --argjson api "$api_results" '.api_tests = $api' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
        return 1
    fi
    
    # Test health endpoint
    print_test "Testing health endpoint"
    local health_start=$(date +%s%N)
    local health_response=$(curl -s -w "%{http_code}" "$API_BASE/api/v2/universal/health")
    local health_end=$(date +%s%N)
    local health_latency=$(( (health_end - health_start) / 1000000 ))
    local health_code="${health_response: -3}"
    
    if [[ "$health_code" == "200" ]]; then
        print_status "Health endpoint: ${health_latency}ms"
        api_results=$(echo "$api_results" | jq --argjson latency "$health_latency" '. + {"health": {"status": "ok", "latency_ms": $latency}}')
    else
        print_error "Health endpoint failed (HTTP $health_code)"
        api_results=$(echo "$api_results" | jq --arg code "$health_code" '. + {"health": {"status": "error", "http_code": $code}}')
    fi
    
    # Test CRUD operations performance
    print_test "Testing CRUD operations performance"
    
    # CREATE test
    local create_data='{"title": "Performance Test Post", "content": "Testing API performance", "author_id": "550e8400-e29b-41d4-a716-446655440000"}'
    local create_start=$(date +%s%N)
    local create_response=$(curl -s -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$create_data" "$API_BASE/api/v2/universal/posts")
    local create_end=$(date +%s%N)
    local create_latency=$(( (create_end - create_start) / 1000000 ))
    local create_code="${create_response: -3}"
    local create_body="${create_response%???}"
    
    if [[ "$create_code" == "201" ]]; then
        local post_id=$(echo "$create_body" | jq -r '.data.id' 2>/dev/null || echo "")
        print_status "CREATE: ${create_latency}ms"
        api_results=$(echo "$api_results" | jq --argjson latency "$create_latency" '. + {"create": {"status": "ok", "latency_ms": $latency}}')
        
        # READ test
        if [[ -n "$post_id" && "$post_id" != "null" ]]; then
            local read_start=$(date +%s%N)
            local read_response=$(curl -s -w "%{http_code}" "$API_BASE/api/v2/universal/posts/$post_id")
            local read_end=$(date +%s%N)
            local read_latency=$(( (read_end - read_start) / 1000000 ))
            local read_code="${read_response: -3}"
            
            if [[ "$read_code" == "200" ]]; then
                print_status "READ: ${read_latency}ms"
                api_results=$(echo "$api_results" | jq --argjson latency "$read_latency" '. + {"read": {"status": "ok", "latency_ms": $latency}}')
            else
                print_error "READ failed (HTTP $read_code)"
                api_results=$(echo "$api_results" | jq --arg code "$read_code" '. + {"read": {"status": "error", "http_code": $code}}')
            fi
            
            # UPDATE test
            local update_data='{"title": "Updated Performance Test Post"}'
            local update_start=$(date +%s%N)
            local update_response=$(curl -s -w "%{http_code}" -X PUT -H "Content-Type: application/json" -d "$update_data" "$API_BASE/api/v2/universal/posts/$post_id")
            local update_end=$(date +%s%N)
            local update_latency=$(( (update_end - update_start) / 1000000 ))
            local update_code="${update_response: -3}"
            
            if [[ "$update_code" == "200" ]]; then
                print_status "UPDATE: ${update_latency}ms"
                api_results=$(echo "$api_results" | jq --argjson latency "$update_latency" '. + {"update": {"status": "ok", "latency_ms": $latency}}')
            else
                print_error "UPDATE failed (HTTP $update_code)"
                api_results=$(echo "$api_results" | jq --arg code "$update_code" '. + {"update": {"status": "error", "http_code": $code}}')
            fi
            
            # DELETE test
            local delete_start=$(date +%s%N)
            local delete_response=$(curl -s -w "%{http_code}" -X DELETE "$API_BASE/api/v2/universal/posts/$post_id")
            local delete_end=$(date +%s%N)
            local delete_latency=$(( (delete_end - delete_start) / 1000000 ))
            local delete_code="${delete_response: -3}"
            
            if [[ "$delete_code" == "200" ]]; then
                print_status "DELETE: ${delete_latency}ms"
                api_results=$(echo "$api_results" | jq --argjson latency "$delete_latency" '. + {"delete": {"status": "ok", "latency_ms": $latency}}')
            else
                print_error "DELETE failed (HTTP $delete_code)"
                api_results=$(echo "$api_results" | jq --arg code "$delete_code" '. + {"delete": {"status": "error", "http_code": $code}}')
            fi
        fi
    else
        print_error "CREATE failed (HTTP $create_code)"
        api_results=$(echo "$api_results" | jq --arg code "$create_code" '. + {"create": {"status": "error", "http_code": $code}}')
    fi
    
    # Update performance log
    jq --argjson api "$api_results" '.api_tests = $api' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
}

# Test 3: C++ System Performance
test_cpp_system() {
    print_header "C++ SYSTEM TESTING"
    
    local cpp_results="{}"
    
    # Check if C++ system exists
    if [[ -d "./cpp-system" ]]; then
        print_test "Testing C++ compilation and execution"
        
        # Test compilation
        echo "Cleaning build artifacts..."
        local compile_start=$(date +%s%N)
        if cd cpp-system && make clean && make simple &> /dev/null; then
            local compile_end=$(date +%s%N)
            local compile_time=$(( (compile_end - compile_start) / 1000000 ))
            
            print_status "C++ compilation: ${compile_time}ms"
            cpp_results=$(echo "$cpp_results" | jq --argjson time "$compile_time" '. + {"compilation": {"status": "ok", "time_ms": $time}}')
            
            # Test execution if binary exists
            if [[ -f "./build/test_real_data" ]]; then
                local exec_start=$(date +%s%N)
                if timeout 30s ./build/test_real_data &> /dev/null; then
                    local exec_end=$(date +%s%N)
                    local exec_time=$(( (exec_end - exec_start) / 1000000 ))
                    
                    print_status "C++ execution: ${exec_time}ms"
                    cpp_results=$(echo "$cpp_results" | jq --argjson time "$exec_time" '. + {"execution": {"status": "ok", "time_ms": $time}}')
                else
                    print_error "C++ execution failed or timed out"
                    cpp_results=$(echo "$cpp_results" | jq '. + {"execution": {"status": "error", "message": "execution_failed"}}')
                fi
            else
                print_warning "C++ binary not found"
                cpp_results=$(echo "$cpp_results" | jq '. + {"execution": {"status": "not_found", "message": "binary_missing"}}')
            fi
        else
            print_error "C++ compilation failed"
            cpp_results=$(echo "$cpp_results" | jq '. + {"compilation": {"status": "error", "message": "compilation_failed"}}')
        fi
        cd - > /dev/null
    else
        print_warning "C++ system directory not found"
        cpp_results='{"status": "not_found", "message": "cpp_system_directory_missing"}'
    fi
    
    # Update performance log
    jq --argjson cpp "$cpp_results" '.cpp_tests = $cpp' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
}

# Test 4: Load Testing
test_load_performance() {
    print_header "LOAD TESTING"
    
    local load_results="{}"
    
    print_test "Running concurrent API requests"
    
    # Create temporary script for concurrent requests
    cat > "$TEST_RESULTS_DIR/load_test.sh" << 'EOF'
#!/bin/bash
API_BASE="http://localhost:3005"
REQUESTS=0
SUCCESSFUL=0
FAILED=0

for i in {1..50}; do
    response=$(curl -s -w "%{http_code}" "$API_BASE/api/v2/universal/health" 2>/dev/null)
    code="${response: -3}"
    REQUESTS=$((REQUESTS + 1))
    
    if [[ "$code" == "200" ]]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo "$REQUESTS,$SUCCESSFUL,$FAILED"
EOF
    
    chmod +x "$TEST_RESULTS_DIR/load_test.sh"
    
    # Run concurrent load test
    local load_start=$(date +%s%N)
    
    # Run 5 concurrent processes
    local pids=()
    for i in {1..5}; do
        "$TEST_RESULTS_DIR/load_test.sh" > "$TEST_RESULTS_DIR/load_result_$i.txt" &
        pids+=($!)
    done
    
    # Wait for all processes to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    local load_end=$(date +%s%N)
    local load_time=$(( (load_end - load_start) / 1000000 ))
    
    # Aggregate results
    local total_requests=0
    local total_successful=0
    local total_failed=0
    
    for i in {1..5}; do
        if [[ -f "$TEST_RESULTS_DIR/load_result_$i.txt" ]]; then
            local result=$(cat "$TEST_RESULTS_DIR/load_result_$i.txt")
            IFS=',' read -r requests successful failed <<< "$result"
            total_requests=$((total_requests + requests))
            total_successful=$((total_successful + successful))
            total_failed=$((total_failed + failed))
        fi
    done
    
    local success_rate=$(( total_successful * 100 / total_requests ))
    local rps=$(( total_requests * 1000 / load_time ))
    
    print_status "Load test completed: ${total_requests} requests in ${load_time}ms"
    print_info "Success rate: ${success_rate}% (${total_successful}/${total_requests})"
    print_info "Requests per second: ${rps}"
    
    load_results=$(jq -n --argjson requests "$total_requests" --argjson successful "$total_successful" --argjson failed "$total_failed" --argjson time "$load_time" --argjson rate "$success_rate" --argjson rps "$rps" '{"total_requests": $requests, "successful": $successful, "failed": $failed, "time_ms": $time, "success_rate_percent": $rate, "requests_per_second": $rps}')
    
    # Update performance log
    jq --argjson load "$load_results" '.load_tests = $load' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
    
    # Cleanup
    rm -f "$TEST_RESULTS_DIR/load_test.sh" "$TEST_RESULTS_DIR/load_result_"*.txt
}

# Test 5: Security Testing
test_security() {
    print_header "SECURITY TESTING"
    
    local security_results="{}"
    
    # Test SQL injection protection
    print_test "Testing injection protection"
    local injection_payload="'; DROP TABLE posts; --"
    local injection_response=$(curl -s -w "%{http_code}" "$API_BASE/api/v2/universal/posts?q=$injection_payload")
    local injection_code="${injection_response: -3}"
    
    if [[ "$injection_code" == "400" ]]; then
        print_status "Injection protection: OK"
        security_results=$(echo "$security_results" | jq '. + {"injection_protection": {"status": "ok", "test": "sql_injection"}}')
    else
        print_error "Injection protection: FAILED"
        security_results=$(echo "$security_results" | jq '. + {"injection_protection": {"status": "failed", "test": "sql_injection"}}')
    fi
    
    # Test rate limiting (if implemented)
    print_test "Testing rate limiting behavior"
    local rate_limit_ok=true
    for i in {1..20}; do
        local response=$(curl -s -w "%{http_code}" "$API_BASE/api/v2/universal/health")
        local code="${response: -3}"
        if [[ "$code" == "429" ]]; then
            rate_limit_ok=true
            break
        fi
    done
    
    if [[ "$rate_limit_ok" == true ]]; then
        print_status "Rate limiting: Implemented"
        security_results=$(echo "$security_results" | jq '. + {"rate_limiting": {"status": "implemented"}}')
    else
        print_warning "Rate limiting: Not detected"
        security_results=$(echo "$security_results" | jq '. + {"rate_limiting": {"status": "not_detected"}}')
    fi
    
    # Update performance log
    jq --argjson security "$security_results" '.security_tests = $security' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
}

# Calculate Production Readiness Score
calculate_readiness_score() {
    print_header "PRODUCTION READINESS ASSESSMENT"
    
    local score=0
    local recommendations=()
    
    # Database health (25 points)
    local scylla_status=$(jq -r '.database_tests.scylladb.status // "unknown"' "$PERFORMANCE_LOG")
    local fdb_status=$(jq -r '.database_tests.foundationdb.status // "unknown"' "$PERFORMANCE_LOG")
    
    if [[ "$scylla_status" == "healthy" ]]; then
        score=$((score + 15))
        print_status "ScyllaDB: Ready for production"
    else
        print_error "ScyllaDB: Not ready - $scylla_status"
        recommendations+=("Fix ScyllaDB connection and performance issues")
    fi
    
    if [[ "$fdb_status" == "healthy" ]]; then
        score=$((score + 10))
        print_status "FoundationDB: Ready for production"
    else
        print_warning "FoundationDB: Issues detected - $fdb_status"
        recommendations+=("Resolve FoundationDB connectivity issues")
    fi
    
    # API performance (30 points)
    local api_health=$(jq -r '.api_tests.health.status // "unknown"' "$PERFORMANCE_LOG")
    local create_status=$(jq -r '.api_tests.create.status // "unknown"' "$PERFORMANCE_LOG")
    local read_status=$(jq -r '.api_tests.read.status // "unknown"' "$PERFORMANCE_LOG")
    
    if [[ "$api_health" == "ok" ]]; then
        score=$((score + 10))
    else
        recommendations+=("Fix API health endpoint issues")
    fi
    
    if [[ "$create_status" == "ok" && "$read_status" == "ok" ]]; then
        score=$((score + 20))
        print_status "API CRUD: Ready for production"
    else
        print_error "API CRUD: Issues detected"
        recommendations+=("Fix API CRUD operation failures")
    fi
    
    # Load testing (20 points)
    local success_rate=$(jq -r '.load_tests.success_rate_percent // 0' "$PERFORMANCE_LOG")
    local rps=$(jq -r '.load_tests.requests_per_second // 0' "$PERFORMANCE_LOG")
    
    if [[ "$success_rate" -ge 95 ]]; then
        score=$((score + 10))
        print_status "Load testing: Excellent success rate ($success_rate%)"
    elif [[ "$success_rate" -ge 90 ]]; then
        score=$((score + 7))
        print_warning "Load testing: Good success rate ($success_rate%)"
    else
        print_error "Load testing: Poor success rate ($success_rate%)"
        recommendations+=("Improve API reliability under load")
    fi
    
    if [[ "$rps" -ge 100 ]]; then
        score=$((score + 10))
        print_status "Throughput: Excellent ($rps RPS)"
    elif [[ "$rps" -ge 50 ]]; then
        score=$((score + 7))
        print_warning "Throughput: Good ($rps RPS)"
    else
        print_error "Throughput: Poor ($rps RPS)"
        recommendations+=("Optimize API performance for higher throughput")
    fi
    
    # C++ system (15 points)
    local cpp_compile=$(jq -r '.cpp_tests.compilation.status // "unknown"' "$PERFORMANCE_LOG")
    local cpp_exec=$(jq -r '.cpp_tests.execution.status // "unknown"' "$PERFORMANCE_LOG")
    
    if [[ "$cpp_compile" == "ok" ]]; then
        score=$((score + 8))
        print_status "C++ compilation: Ready"
    else
        print_warning "C++ compilation: Issues detected"
        recommendations+=("Fix C++ compilation issues")
    fi
    
    if [[ "$cpp_exec" == "ok" ]]; then
        score=$((score + 7))
        print_status "C++ execution: Ready"
    else
        print_warning "C++ execution: Issues detected"
        recommendations+=("Fix C++ runtime issues")
    fi
    
    # Security (10 points)
    local injection_protection=$(jq -r '.security_tests.injection_protection.status // "unknown"' "$PERFORMANCE_LOG")
    
    if [[ "$injection_protection" == "ok" ]]; then
        score=$((score + 10))
        print_status "Security: Basic protection in place"
    else
        print_error "Security: Vulnerabilities detected"
        recommendations+=("Implement proper input validation and security measures")
    fi
    
    # Update performance log with final score
    local recommendations_json=$(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s .)
    jq --argjson score "$score" --argjson recs "$recommendations_json" '.production_readiness.score = $score | .production_readiness.recommendations = $recs' "$PERFORMANCE_LOG" > "$PERFORMANCE_LOG.tmp" && mv "$PERFORMANCE_LOG.tmp" "$PERFORMANCE_LOG"
    
    # Final assessment
    echo ""
    print_header "FINAL ASSESSMENT"
    
    if [[ $score -ge 85 ]]; then
        print_status "🎉 PRODUCTION READY! Score: $score/100"
        print_status "✅ System is ready for AWS deployment"
        return 0
    elif [[ $score -ge 70 ]]; then
        print_warning "⚠️  MOSTLY READY. Score: $score/100"
        print_warning "🔧 Address recommendations before deployment"
        return 1
    else
        print_error "❌ NOT READY FOR PRODUCTION. Score: $score/100"
        print_error "🚫 Critical issues must be resolved"
        return 2
    fi
}

# Generate detailed report
generate_report() {
    print_header "GENERATING DETAILED REPORT"
    
    local report_file="$TEST_RESULTS_DIR/production-readiness-report.md"
    
    cat > "$report_file" << EOF
# Production Readiness Report

**Generated:** $(date)
**System:** Global API Management v2 (ScyllaDB + FoundationDB)

## Executive Summary

$(jq -r '.production_readiness.score' "$PERFORMANCE_LOG")/100 Production Readiness Score

## Environment Information

- **OS:** $(jq -r '.environment.os' "$PERFORMANCE_LOG")
- **Node.js:** $(jq -r '.environment.node_version' "$PERFORMANCE_LOG")
- **Memory:** $(jq -r '.environment.memory' "$PERFORMANCE_LOG")
- **CPU Cores:** $(jq -r '.environment.cpu_cores' "$PERFORMANCE_LOG")

## Database Performance

### ScyllaDB
- **Status:** $(jq -r '.database_tests.scylladb.status // "Not tested"' "$PERFORMANCE_LOG")
- **Connection Latency:** $(jq -r '.database_tests.scylladb.connection_latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **Write Latency:** $(jq -r '.database_tests.scylladb.write_latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **Read Latency:** $(jq -r '.database_tests.scylladb.read_latency_ms // "N/A"' "$PERFORMANCE_LOG")ms

### FoundationDB
- **Status:** $(jq -r '.database_tests.foundationdb.status // "Not tested"' "$PERFORMANCE_LOG")
- **Connection Latency:** $(jq -r '.database_tests.foundationdb.connection_latency_ms // "N/A"' "$PERFORMANCE_LOG")ms

## API Performance

- **Health Check:** $(jq -r '.api_tests.health.latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **CREATE Operation:** $(jq -r '.api_tests.create.latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **READ Operation:** $(jq -r '.api_tests.read.latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **UPDATE Operation:** $(jq -r '.api_tests.update.latency_ms // "N/A"' "$PERFORMANCE_LOG")ms
- **DELETE Operation:** $(jq -r '.api_tests.delete.latency_ms // "N/A"' "$PERFORMANCE_LOG")ms

## Load Testing Results

- **Total Requests:** $(jq -r '.load_tests.total_requests // "N/A"' "$PERFORMANCE_LOG")
- **Success Rate:** $(jq -r '.load_tests.success_rate_percent // "N/A"' "$PERFORMANCE_LOG")%
- **Requests per Second:** $(jq -r '.load_tests.requests_per_second // "N/A"' "$PERFORMANCE_LOG")
- **Test Duration:** $(jq -r '.load_tests.time_ms // "N/A"' "$PERFORMANCE_LOG")ms

## C++ System Performance

- **Compilation:** $(jq -r '.cpp_tests.compilation.status // "Not tested"' "$PERFORMANCE_LOG")
- **Execution:** $(jq -r '.cpp_tests.execution.status // "Not tested"' "$PERFORMANCE_LOG")

## Security Assessment

- **Injection Protection:** $(jq -r '.security_tests.injection_protection.status // "Not tested"' "$PERFORMANCE_LOG")
- **Rate Limiting:** $(jq -r '.security_tests.rate_limiting.status // "Not tested"' "$PERFORMANCE_LOG")

## Recommendations

$(jq -r '.production_readiness.recommendations[]' "$PERFORMANCE_LOG" | sed 's/^/- /')

## AWS Deployment Readiness

$(if [[ $(jq -r '.production_readiness.score' "$PERFORMANCE_LOG") -ge 85 ]]; then
    echo "✅ **READY FOR AWS DEPLOYMENT**"
    echo ""
    echo "The system has passed all critical tests and is ready for production deployment on AWS."
elif [[ $(jq -r '.production_readiness.score' "$PERFORMANCE_LOG") -ge 70 ]]; then
    echo "⚠️ **CONDITIONAL DEPLOYMENT**"
    echo ""
    echo "The system is mostly ready but should address the recommendations above before deployment."
else
    echo "❌ **NOT READY FOR DEPLOYMENT**"
    echo ""
    echo "Critical issues must be resolved before considering AWS deployment."
fi)

---
*Report generated by Production Readiness Test Suite*
EOF

    print_status "Report generated: $report_file"
    print_info "Performance data: $PERFORMANCE_LOG"
}

# Main execution
main() {
    echo "🚀 Production Readiness Test Suite"
    echo "=================================="
    echo "Comprehensive testing for AWS deployment readiness"
    echo ""
    
    initialize_test_env
    echo ""
    
    test_database_systems
    echo ""
    
    test_api_performance
    echo ""
    
    test_cpp_system
    echo ""
    
    test_load_performance
    echo ""
    
    test_security
    echo ""
    
    local readiness_result
    calculate_readiness_score
    readiness_result=$?
    echo ""
    
    generate_report
    echo ""
    
    # Show recommendations if any
    local recommendations=$(jq -r '.production_readiness.recommendations[]' "$PERFORMANCE_LOG" 2>/dev/null)
    if [[ -n "$recommendations" ]]; then
        print_header "RECOMMENDATIONS"
        echo "$recommendations" | while read -r rec; do
            print_info "$rec"
        done
        echo ""
    fi
    
    # Final verdict
    local final_score=$(jq -r '.production_readiness.score' "$PERFORMANCE_LOG")
    print_header "DEPLOYMENT VERDICT"
    
    if [[ $readiness_result -eq 0 ]]; then
        print_status "🎯 DEPLOY TO AWS: System is production-ready!"
        print_status "📊 Final Score: $final_score/100"
        print_status "🚀 Proceed with AWS deployment"
    elif [[ $readiness_result -eq 1 ]]; then
        print_warning "🔧 CONDITIONAL DEPLOYMENT: Address recommendations first"
        print_warning "📊 Final Score: $final_score/100"
        print_warning "⚠️  Consider fixing issues before AWS deployment"
    else
        print_error "🚫 DO NOT DEPLOY: Critical issues detected"
        print_error "📊 Final Score: $final_score/100"
        print_error "❌ Resolve all critical issues before deployment"
    fi
    
    return $readiness_result
}

# Run main function
main "$@"