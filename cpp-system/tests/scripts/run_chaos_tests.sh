#!/bin/bash

# Ultra Low-Latency System Chaos Test Runner
# This script runs chaos engineering tests to validate system resilience

set -e

# Configuration
OUTPUT_DIR="./test_results/chaos_tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
DEFAULT_DURATION=300
ENABLE_DESTRUCTIVE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DEFAULT_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
            shift 2
            ;;
        --enable-destructive)
            ENABLE_DESTRUCTIVE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --duration <seconds>   Test duration in seconds (default: 300)"
            echo "  --output-dir <dir>     Output directory (default: ./test_results/chaos_tests)"
            echo "  --enable-destructive   Enable destructive chaos tests (default: false)"
            echo "  --help                 Show this help message"
            echo ""
            echo "WARNING: Destructive tests can impact system performance and stability."
            echo "Only use --enable-destructive in test environments!"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Ultra Low-Latency System Chaos Test Runner${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""
echo "Test Duration: $DEFAULT_DURATION seconds"
echo "Output Directory: $REPORT_DIR"
echo "Destructive Tests: $([ "$ENABLE_DESTRUCTIVE" = true ] && echo "ENABLED" || echo "DISABLED")"
echo ""

if [ "$ENABLE_DESTRUCTIVE" = false ]; then
    echo -e "${YELLOW}WARNING: Running in simulation mode.${NC}"
    echo -e "${YELLOW}Use --enable-destructive to run actual chaos experiments.${NC}"
    echo ""
fi

# Create output directory
mkdir -p "${REPORT_DIR}"

# Function to run a chaos test
run_chaos_test() {
    local test_name="$1"
    local experiment="$2"
    local duration="$3"
    local probability="$4"
    
    echo -e "${YELLOW}Running ${test_name}...${NC}"
    echo "  Experiment: $experiment, Duration: ${duration}s, Probability: $probability"
    
    local output_file="${REPORT_DIR}/${test_name}_report.json"
    local log_file="${REPORT_DIR}/${test_name}.log"
    
    local destructive_flag=""
    if [ "$ENABLE_DESTRUCTIVE" = true ]; then
        destructive_flag="--enable-destructive"
    fi
    
    if ./chaos_tests \
        --experiment "$experiment" \
        --duration "$duration" \
        --probability "$probability" \
        --output "$output_file" \
        $destructive_flag > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        echo "  Log: $log_file"
        return 1
    fi
}

# Function to check system resources before chaos tests
check_system_resources() {
    echo "Checking system resources..."
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 1000 ]; then
        echo -e "${YELLOW}⚠ Low available memory: ${available_memory}MB${NC}"
        echo "  Consider freeing up memory before running chaos tests."
    fi
    
    # Check CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)
    local cpu_cores=$(nproc)
    local load_threshold=$(echo "$cpu_cores * 0.8" | bc -l)
    
    if (( $(echo "$cpu_load > $load_threshold" | bc -l) )); then
        echo -e "${YELLOW}⚠ High CPU load: ${cpu_load} (threshold: ${load_threshold})${NC}"
        echo "  Consider reducing system load before running chaos tests."
    fi
    
    # Check disk space
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        echo -e "${YELLOW}⚠ High disk usage: ${disk_usage}%${NC}"
        echo "  Consider freeing up disk space before running chaos tests."
    fi
    
    echo -e "${GREEN}✓ System resource check completed${NC}"
}

# Check if chaos test binary exists
if [ ! -f "./chaos_tests" ]; then
    echo -e "${RED}Error: chaos_tests binary not found. Please build the project first.${NC}"
    exit 1
fi

# Check system resources
check_system_resources

# Initialize results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo ""
echo -e "${BLUE}=== Starting Chaos Tests ===${NC}"

# Test 1: Network Latency Injection
echo ""
if run_chaos_test "network_latency" "network_latency" $DEFAULT_DURATION 0.2; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 2: Memory Pressure
echo ""
if run_chaos_test "memory_pressure" "memory_pressure" $DEFAULT_DURATION 0.15; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 3: Exception Injection
echo ""
if run_chaos_test "exception_injection" "exception_injection" $DEFAULT_DURATION 0.1; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Test 4: Comprehensive Chaos Test (all experiments)
if [ $DEFAULT_DURATION -ge 300 ]; then
    echo ""
    if run_chaos_test "comprehensive_chaos" "all" $DEFAULT_DURATION 0.1; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Additional destructive tests (only if enabled)
if [ "$ENABLE_DESTRUCTIVE" = true ]; then
    echo ""
    echo -e "${RED}=== Running Destructive Tests ===${NC}"
    
    # CPU Stress Test
    echo ""
    if run_chaos_test "cpu_stress" "cpu_stress" $(( DEFAULT_DURATION / 2 )) 0.3; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    # Disk I/O Failure Test
    echo ""
    if run_chaos_test "disk_io_failure" "disk_io_failure" $(( DEFAULT_DURATION / 2 )) 0.2; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Generate comprehensive chaos test report
echo ""
echo -e "${BLUE}=== Generating Chaos Test Report ===${NC}"

cat > "${REPORT_DIR}/chaos_test_summary.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Low-Latency System Chaos Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .warning { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .summary { margin: 20px 0; }
        .test-result { margin: 10px 0; padding: 10px; border-radius: 3px; }
        .passed { background-color: #d4edda; color: #155724; }
        .failed { background-color: #f8d7da; color: #721c24; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }
        .test-links { margin: 20px 0; }
        .test-links a { display: block; margin: 5px 0; padding: 10px; background-color: #e9ecef; text-decoration: none; border-radius: 3px; }
        .test-links a:hover { background-color: #dee2e6; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Ultra Low-Latency System Chaos Test Report</h1>
        <p>Generated on: $(date)</p>
        <p>Test Duration: $DEFAULT_DURATION seconds per test</p>
        <p>Destructive Tests: $([ "$ENABLE_DESTRUCTIVE" = true ] && echo "ENABLED" || echo "DISABLED")</p>
    </div>
    
EOF

if [ "$ENABLE_DESTRUCTIVE" = false ]; then
    cat >> "${REPORT_DIR}/chaos_test_summary.html" << EOF
    <div class="warning">
        <h3>⚠ Simulation Mode</h3>
        <p>These tests were run in simulation mode. To run actual chaos experiments that can impact system performance, use the --enable-destructive flag.</p>
    </div>
EOF
fi

cat >> "${REPORT_DIR}/chaos_test_summary.html" << EOF
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
                <h3>Resilience Score</h3>
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
        if grep -q "Passed: YES" "$log_file" 2>/dev/null; then
            echo "        <div class=\"test-result passed\">✓ ${test_name} - PASSED</div>" >> "${REPORT_DIR}/chaos_test_summary.html"
        else
            echo "        <div class=\"test-result failed\">✗ ${test_name} - FAILED</div>" >> "${REPORT_DIR}/chaos_test_summary.html"
        fi
    fi
done

cat >> "${REPORT_DIR}/chaos_test_summary.html" << EOF
    </div>
    
    <div class="test-links">
        <h2>Individual Test Reports</h2>
EOF

# Add links to individual test reports
for report_file in "${REPORT_DIR}"/*_report.json; do
    if [ -f "$report_file" ]; then
        report_name=$(basename "$report_file")
        test_name=$(basename "$report_file" _report.json)
        echo "        <a href=\"${report_name}\">${test_name} - JSON Report</a>" >> "${REPORT_DIR}/chaos_test_summary.html"
    fi
done

cat >> "${REPORT_DIR}/chaos_test_summary.html" << EOF
    </div>
    
    <div class="logs">
        <h2>Test Logs</h2>
        <ul>
EOF

# Add links to log files
for log_file in "${REPORT_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        log_name=$(basename "$log_file")
        echo "            <li><a href=\"${log_name}\">${log_name}</a></li>" >> "${REPORT_DIR}/chaos_test_summary.html"
    fi
done

cat >> "${REPORT_DIR}/chaos_test_summary.html" << EOF
        </ul>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            <li>Review failed tests to identify system weaknesses</li>
            <li>Implement circuit breakers for components that failed under chaos</li>
            <li>Add monitoring and alerting for detected failure modes</li>
            <li>Run chaos tests regularly in staging environments</li>
            <li>Consider implementing auto-recovery mechanisms</li>
        </ul>
    </div>
</body>
</html>
EOF

# Print final summary
echo ""
echo -e "${BLUE}=== Chaos Test Summary ===${NC}"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Resilience Score: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo ""
echo "Reports generated in: $REPORT_DIR"
echo "Main report: ${REPORT_DIR}/chaos_test_summary.html"

if [ "$ENABLE_DESTRUCTIVE" = false ]; then
    echo ""
    echo -e "${YELLOW}Note: Tests were run in simulation mode.${NC}"
    echo -e "${YELLOW}Use --enable-destructive for actual chaos experiments.${NC}"
fi

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All chaos tests passed! System shows good resilience.${NC}"
    exit 0
else
    echo -e "${RED}Some chaos tests failed. System resilience needs improvement.${NC}"
    exit 1
fi