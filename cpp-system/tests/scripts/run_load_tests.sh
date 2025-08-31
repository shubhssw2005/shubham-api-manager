#!/bin/bash

# Ultra Low-Latency System Load Test Runner
# This script runs comprehensive load tests with different traffic patterns

set -e

# Configuration
OUTPUT_DIR="./test_results/load_tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
TARGET_ENDPOINT="http://localhost:8080"
DEFAULT_DURATION=60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET_ENDPOINT="$2"
            shift 2
            ;;
        --duration)
            DEFAULT_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --target <endpoint>    Target endpoint (default: http://localhost:8080)"
            echo "  --duration <seconds>   Test duration in seconds (default: 60)"
            echo "  --output-dir <dir>     Output directory (default: ./test_results/load_tests)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Ultra Low-Latency System Load Test Runner${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Target Endpoint: $TARGET_ENDPOINT"
echo "Test Duration: $DEFAULT_DURATION seconds"
echo "Output Directory: $REPORT_DIR"
echo ""

# Create output directory
mkdir -p "${REPORT_DIR}"

# Function to run a load test
run_load_test() {
    local test_name="$1"
    local test_type="$2"
    local rps="$3"
    local peak_rps="$4"
    local duration="$5"
    local threads="$6"
    
    echo -e "${YELLOW}Running ${test_name}...${NC}"
    echo "  Type: $test_type, RPS: $rps, Peak RPS: $peak_rps, Duration: ${duration}s, Threads: $threads"
    
    local output_file="${REPORT_DIR}/${test_name}_report.html"
    local log_file="${REPORT_DIR}/${test_name}.log"
    
    if ./load_tests \
        --test-type "$test_type" \
        --target "$TARGET_ENDPOINT" \
        --rps "$rps" \
        --peak-rps "$peak_rps" \
        --duration "$duration" \
        --threads "$threads" \
        --output "$output_file" > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        echo "  Log: $log_file"
        return 1
    fi
}

# Function to check if the target endpoint is reachable
check_endpoint() {
    echo "Checking target endpoint: $TARGET_ENDPOINT"
    
    if curl -s --connect-timeout 5 "$TARGET_ENDPOINT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Target endpoint is reachable${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Target endpoint is not reachable. Tests will run but may fail.${NC}"
        echo "  Make sure your application is running at $TARGET_ENDPOINT"
        return 1
    fi
}

# Check if load test binary exists
if [ ! -f "./load_tests" ]; then
    echo -e "${RED}Error: load_tests binary not found. Please build the project first.${NC}"
    exit 1
fi

# Check endpoint reachability
check_endpoint

# Initialize results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo ""
echo -e "${BLUE}=== Starting Load Tests ===${NC}"

# Test 1: Constant Load - Low RPS
echo ""
if run_load_test "constant_low" "constant" 500 500 $DEFAULT_DURATION 2; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 2: Constant Load - Medium RPS
echo ""
if run_load_test "constant_medium" "constant" 1000 1000 $DEFAULT_DURATION 4; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 3: Constant Load - High RPS
echo ""
if run_load_test "constant_high" "constant" 2000 2000 $DEFAULT_DURATION 8; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 4: Ramp Up Test
echo ""
if run_load_test "ramp_up" "ramp" 100 2000 $DEFAULT_DURATION 4; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 5: Spike Test
echo ""
if run_load_test "spike_test" "spike" 500 5000 $DEFAULT_DURATION 8; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 6: Burst Test
echo ""
if run_load_test "burst_test" "burst" 200 2000 $DEFAULT_DURATION 4; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 7: Stress Test (if duration is long enough)
if [ $DEFAULT_DURATION -ge 120 ]; then
    echo ""
    if run_load_test "stress_test" "stress" 1000 10000 $DEFAULT_DURATION 16; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Test 8: Realistic Web Traffic (if duration is long enough)
if [ $DEFAULT_DURATION -ge 300 ]; then
    echo ""
    if run_load_test "realistic_web" "realistic" 500 2000 $DEFAULT_DURATION 8; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Generate comprehensive load test report
echo ""
echo -e "${BLUE}=== Generating Load Test Report ===${NC}"

cat > "${REPORT_DIR}/load_test_summary.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Low-Latency System Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .test-result { margin: 10px 0; padding: 10px; border-radius: 3px; }
        .passed { background-color: #d4edda; color: #155724; }
        .failed { background-color: #f8d7da; color: #721c24; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .test-links { margin: 20px 0; }
        .test-links a { display: block; margin: 5px 0; padding: 10px; background-color: #e9ecef; text-decoration: none; border-radius: 3px; }
        .test-links a:hover { background-color: #dee2e6; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Ultra Low-Latency System Load Test Report</h1>
        <p>Generated on: $(date)</p>
        <p>Target Endpoint: $TARGET_ENDPOINT</p>
        <p>Test Duration: $DEFAULT_DURATION seconds per test</p>
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
        if grep -q "SLA Passed: YES" "$log_file" 2>/dev/null; then
            echo "        <div class=\"test-result passed\">✓ ${test_name} - PASSED</div>" >> "${REPORT_DIR}/load_test_summary.html"
        else
            echo "        <div class=\"test-result failed\">✗ ${test_name} - FAILED</div>" >> "${REPORT_DIR}/load_test_summary.html"
        fi
    fi
done

cat >> "${REPORT_DIR}/load_test_summary.html" << EOF
    </div>
    
    <div class="test-links">
        <h2>Individual Test Reports</h2>
EOF

# Add links to individual test reports
for report_file in "${REPORT_DIR}"/*_report.html; do
    if [ -f "$report_file" ]; then
        report_name=$(basename "$report_file")
        test_name=$(basename "$report_file" _report.html)
        echo "        <a href=\"${report_name}\">${test_name} - Detailed Report</a>" >> "${REPORT_DIR}/load_test_summary.html"
    fi
done

cat >> "${REPORT_DIR}/load_test_summary.html" << EOF
    </div>
    
    <div class="logs">
        <h2>Test Logs</h2>
        <ul>
EOF

# Add links to log files
for log_file in "${REPORT_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        log_name=$(basename "$log_file")
        echo "            <li><a href=\"${log_name}\">${log_name}</a></li>" >> "${REPORT_DIR}/load_test_summary.html"
    fi
done

cat >> "${REPORT_DIR}/load_test_summary.html" << EOF
        </ul>
    </div>
</body>
</html>
EOF

# Print final summary
echo ""
echo -e "${BLUE}=== Load Test Summary ===${NC}"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo ""
echo "Reports generated in: $REPORT_DIR"
echo "Main report: ${REPORT_DIR}/load_test_summary.html"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All load tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some load tests failed. Check the reports for details.${NC}"
    exit 1
fi