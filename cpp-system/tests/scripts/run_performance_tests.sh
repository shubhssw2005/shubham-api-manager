#!/bin/bash

# Ultra Low-Latency System Performance Test Runner
# This script runs comprehensive performance tests and generates reports

set -e

# Configuration
OUTPUT_DIR="./test_results/performance"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Ultra Low-Latency System Performance Test Runner${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Create output directory
mkdir -p "${REPORT_DIR}"

# Function to run a test and capture results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local log_file="${REPORT_DIR}/${test_name}.log"
    
    echo -e "${YELLOW}Running ${test_name}...${NC}"
    
    if eval "$test_command" > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        echo "  Log: $log_file"
        return 1
    fi
}

# Function to check if binary exists
check_binary() {
    local binary="$1"
    if [ ! -f "$binary" ]; then
        echo -e "${RED}Error: $binary not found. Please build the project first.${NC}"
        exit 1
    fi
}

# Check if binaries exist
echo "Checking test binaries..."
check_binary "./comprehensive_unit_tests"
check_binary "./comprehensive_benchmarks"
check_binary "./ultra_test_runner"

# Initialize results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Run unit tests with performance validation
echo ""
echo -e "${BLUE}=== Unit Tests with Performance Validation ===${NC}"
if run_test "unit_tests_performance" "./comprehensive_unit_tests --gtest_output=xml:${REPORT_DIR}/unit_tests.xml"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run comprehensive benchmarks
echo ""
echo -e "${BLUE}=== Performance Benchmarks ===${NC}"
if run_test "performance_benchmarks" "./comprehensive_benchmarks --benchmark_format=json --benchmark_out=${REPORT_DIR}/benchmarks.json --benchmark_min_time=1.0"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run latency tests
echo ""
echo -e "${BLUE}=== Latency Tests ===${NC}"
if run_test "latency_tests" "./ultra_test_runner --test-type benchmark --duration 30 --output-dir ${REPORT_DIR}/latency"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run throughput tests
echo ""
echo -e "${BLUE}=== Throughput Tests ===${NC}"
if run_test "throughput_tests" "./load_tests --test-type constant --rps 10000 --duration 60 --output ${REPORT_DIR}/throughput_report.html"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run memory performance tests
echo ""
echo -e "${BLUE}=== Memory Performance Tests ===${NC}"
if run_test "memory_performance" "./comprehensive_benchmarks --benchmark_filter=Memory --benchmark_format=json --benchmark_out=${REPORT_DIR}/memory_benchmarks.json"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run cache performance tests
echo ""
echo -e "${BLUE}=== Cache Performance Tests ===${NC}"
if run_test "cache_performance" "./comprehensive_benchmarks --benchmark_filter=Cache --benchmark_format=json --benchmark_out=${REPORT_DIR}/cache_benchmarks.json"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run lock-free performance tests
echo ""
echo -e "${BLUE}=== Lock-Free Performance Tests ===${NC}"
if run_test "lockfree_performance" "./comprehensive_benchmarks --benchmark_filter=LockFree --benchmark_format=json --benchmark_out=${REPORT_DIR}/lockfree_benchmarks.json"; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Generate performance report
echo ""
echo -e "${BLUE}=== Generating Performance Report ===${NC}"

cat > "${REPORT_DIR}/performance_summary.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Low-Latency System Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .test-result { margin: 10px 0; padding: 10px; border-radius: 3px; }
        .passed { background-color: #d4edda; color: #155724; }
        .failed { background-color: #f8d7da; color: #721c24; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Ultra Low-Latency System Performance Report</h1>
        <p>Generated on: $(date)</p>
        <p>Test Duration: Performance validation and benchmarking</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Total Tests</h3>
                <p style="font-size: 2em; margin: 0;">${TOTAL_TESTS}</p>
            </div>
            <div class="metric-card">
                <h3>Passed Tests</h3>
                <p style="font-size: 2em; margin: 0; color: green;">${PASSED_TESTS}</p>
            </div>
            <div class="metric-card">
                <h3>Failed Tests</h3>
                <p style="font-size: 2em; margin: 0; color: red;">${FAILED_TESTS}</p>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <p style="font-size: 2em; margin: 0;">$(( PASSED_TESTS * 100 / TOTAL_TESTS ))%</p>
            </div>
        </div>
    </div>
    
    <div class="results">
        <h2>Test Results</h2>
EOF

# Add test results to HTML report
for log_file in "${REPORT_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        test_name=$(basename "$log_file" .log)
        if grep -q "PASSED\|SUCCESS" "$log_file" 2>/dev/null; then
            echo "        <div class=\"test-result passed\">✓ ${test_name} - PASSED</div>" >> "${REPORT_DIR}/performance_summary.html"
        else
            echo "        <div class=\"test-result failed\">✗ ${test_name} - FAILED</div>" >> "${REPORT_DIR}/performance_summary.html"
        fi
    fi
done

cat >> "${REPORT_DIR}/performance_summary.html" << EOF
    </div>
    
    <div class="files">
        <h2>Generated Files</h2>
        <ul>
            <li><a href="benchmarks.json">Benchmark Results (JSON)</a></li>
            <li><a href="unit_tests.xml">Unit Test Results (XML)</a></li>
            <li><a href="throughput_report.html">Throughput Test Report</a></li>
            <li><a href="memory_benchmarks.json">Memory Performance Results</a></li>
            <li><a href="cache_benchmarks.json">Cache Performance Results</a></li>
            <li><a href="lockfree_benchmarks.json">Lock-Free Performance Results</a></li>
        </ul>
    </div>
</body>
</html>
EOF

# Print final summary
echo ""
echo -e "${BLUE}=== Performance Test Summary ===${NC}"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo ""
echo "Reports generated in: $REPORT_DIR"
echo "Main report: ${REPORT_DIR}/performance_summary.html"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All performance tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some performance tests failed. Check the logs for details.${NC}"
    exit 1
fi