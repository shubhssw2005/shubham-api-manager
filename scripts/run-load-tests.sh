#!/bin/bash

# Load Testing and Performance Validation Runner
# This script orchestrates all load testing scenarios for the AWS deployment system

set -e

# Configuration
BASE_URL="${API_BASE_URL:-http://localhost:3000}"
METRICS_URL="${METRICS_URL:-http://localhost:9090}"
CHAOS_API_URL="${CHAOS_API_URL:-http://localhost:8080}"
JWT_TOKEN="${JWT_TOKEN:-}"
TENANT_ID="${TENANT_ID:-test-tenant}"
OUTPUT_DIR="tests/performance/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if k6 is installed
    if ! command -v k6 &> /dev/null; then
        log_error "k6 is not installed. Please install k6 from https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    
    # Check if Node.js is available for capacity planning tool
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if API is accessible
    if ! curl -s "$BASE_URL/health" > /dev/null; then
        log_warning "API at $BASE_URL is not accessible. Some tests may fail."
    fi
    
    log_success "Prerequisites check completed"
}

# Run API load tests
run_api_load_tests() {
    log_info "Running API load tests..."
    
    local test_scenarios=("baseline_load" "stress_test" "spike_test" "soak_test")
    
    for scenario in "${test_scenarios[@]}"; do
        log_info "Running $scenario scenario..."
        
        k6 run \
            --env API_BASE_URL="$BASE_URL" \
            --env JWT_TOKEN="$JWT_TOKEN" \
            --env TENANT_ID="$TENANT_ID" \
            --env K6_SCENARIO_NAME="$scenario" \
            --out json="$OUTPUT_DIR/api_load_${scenario}_${TIMESTAMP}.json" \
            --out influxdb=http://localhost:8086/k6 \
            tests/load/k6-api-load-test.js
        
        if [ $? -eq 0 ]; then
            log_success "$scenario completed successfully"
        else
            log_error "$scenario failed"
        fi
        
        # Wait between scenarios to allow system recovery
        sleep 30
    done
}

# Run media upload tests
run_media_upload_tests() {
    log_info "Running media upload tests..."
    
    local test_scenarios=("media_upload_stress" "concurrent_uploads" "large_file_test")
    
    for scenario in "${test_scenarios[@]}"; do
        log_info "Running $scenario scenario..."
        
        k6 run \
            --env API_BASE_URL="$BASE_URL" \
            --env JWT_TOKEN="$JWT_TOKEN" \
            --env TENANT_ID="$TENANT_ID" \
            --env K6_SCENARIO_NAME="$scenario" \
            --out json="$OUTPUT_DIR/media_upload_${scenario}_${TIMESTAMP}.json" \
            tests/load/k6-media-upload-test.js
        
        if [ $? -eq 0 ]; then
            log_success "$scenario completed successfully"
        else
            log_error "$scenario failed"
        fi
        
        sleep 30
    done
}

# Run chaos engineering tests
run_chaos_tests() {
    log_info "Running chaos engineering tests..."
    
    local chaos_scenarios=("database_chaos" "cache_chaos" "storage_chaos" "network_chaos" "pod_chaos")
    
    for scenario in "${chaos_scenarios[@]}"; do
        log_info "Running $scenario scenario..."
        
        k6 run \
            --env API_BASE_URL="$BASE_URL" \
            --env CHAOS_API_URL="$CHAOS_API_URL" \
            --env JWT_TOKEN="$JWT_TOKEN" \
            --env TENANT_ID="$TENANT_ID" \
            --env K6_SCENARIO_NAME="$scenario" \
            --out json="$OUTPUT_DIR/chaos_${scenario}_${TIMESTAMP}.json" \
            tests/chaos/chaos-engineering-tests.js
        
        if [ $? -eq 0 ]; then
            log_success "$scenario completed successfully"
        else
            log_error "$scenario failed"
        fi
        
        # Longer wait for chaos tests to allow system recovery
        sleep 60
    done
}

# Run performance benchmarks
run_performance_benchmarks() {
    log_info "Running performance benchmarks..."
    
    local benchmark_scenarios=("baseline_benchmark" "capacity_planning" "scalability_test" "efficiency_test")
    
    for scenario in "${benchmark_scenarios[@]}"; do
        log_info "Running $scenario benchmark..."
        
        # Skip endurance test in normal runs (24 hours)
        if [ "$scenario" = "endurance_test" ] && [ "$RUN_ENDURANCE" != "true" ]; then
            log_info "Skipping endurance test (set RUN_ENDURANCE=true to include)"
            continue
        fi
        
        k6 run \
            --env API_BASE_URL="$BASE_URL" \
            --env METRICS_URL="$METRICS_URL" \
            --env JWT_TOKEN="$JWT_TOKEN" \
            --env TENANT_ID="$TENANT_ID" \
            --env K6_SCENARIO_NAME="$scenario" \
            --env TEST_START_TIME="$(date +%s)000" \
            --out json="$OUTPUT_DIR/benchmark_${scenario}_${TIMESTAMP}.json" \
            tests/performance/benchmark-suite.js
        
        if [ $? -eq 0 ]; then
            log_success "$scenario completed successfully"
        else
            log_error "$scenario failed"
        fi
        
        sleep 30
    done
}

# Run capacity planning analysis
run_capacity_planning() {
    log_info "Running capacity planning analysis..."
    
    node scripts/capacity-planning-tool.js
    
    if [ $? -eq 0 ]; then
        log_success "Capacity planning analysis completed"
    else
        log_error "Capacity planning analysis failed"
    fi
}

# Generate consolidated report
generate_report() {
    log_info "Generating consolidated performance report..."
    
    local report_file="$OUTPUT_DIR/performance_report_${TIMESTAMP}.md"
    
    cat > "$report_file" << EOF
# Performance Test Report

Generated: $(date)
Test Run ID: $TIMESTAMP

## Test Configuration

- **Base URL**: $BASE_URL
- **Tenant ID**: $TENANT_ID
- **Output Directory**: $OUTPUT_DIR

## Test Results Summary

### API Load Tests
EOF

    # Add results from each test type
    for result_file in "$OUTPUT_DIR"/api_load_*_${TIMESTAMP}.json; do
        if [ -f "$result_file" ]; then
            scenario=$(basename "$result_file" | cut -d'_' -f3)
            echo "- **$scenario**: $(jq -r '.metrics.http_req_duration.avg' "$result_file" 2>/dev/null || echo "N/A")ms avg response time" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

### Media Upload Tests
EOF

    for result_file in "$OUTPUT_DIR"/media_upload_*_${TIMESTAMP}.json; do
        if [ -f "$result_file" ]; then
            scenario=$(basename "$result_file" | cut -d'_' -f3)
            echo "- **$scenario**: $(jq -r '.metrics.upload_latency.avg' "$result_file" 2>/dev/null || echo "N/A")ms avg upload time" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

### Chaos Engineering Tests
EOF

    for result_file in "$OUTPUT_DIR"/chaos_*_${TIMESTAMP}.json; do
        if [ -f "$result_file" ]; then
            scenario=$(basename "$result_file" | cut -d'_' -f2)
            resilience_score=$(jq -r '.metrics.resilience_score.rate' "$result_file" 2>/dev/null || echo "N/A")
            echo "- **$scenario**: ${resilience_score} resilience score" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

### Performance Benchmarks
EOF

    for result_file in "$OUTPUT_DIR"/benchmark_*_${TIMESTAMP}.json; do
        if [ -f "$result_file" ]; then
            scenario=$(basename "$result_file" | cut -d'_' -f2)
            performance_score=$(jq -r '.metrics.performance_score.value' "$result_file" 2>/dev/null || echo "N/A")
            echo "- **$scenario**: ${performance_score} performance score" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## Capacity Planning

See detailed capacity planning report: \`tests/performance/capacity-plan-report.md\`

## Recommendations

Based on the test results:

1. **Performance**: Review any scenarios with response times > 200ms (P95) or > 500ms (P99)
2. **Resilience**: Investigate any chaos scenarios with resilience score < 0.8
3. **Capacity**: Follow capacity planning recommendations for resource scaling
4. **Cost**: Monitor cost projections and implement optimization strategies

## Files Generated

- Test results: \`$OUTPUT_DIR/*_${TIMESTAMP}.json\`
- Capacity plan: \`tests/performance/capacity-plan.json\`
- This report: \`$report_file\`

---

*Report generated by automated performance testing suite*
EOF

    log_success "Performance report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop any running chaos experiments
    if command -v curl &> /dev/null && [ -n "$CHAOS_API_URL" ]; then
        curl -s -X POST "$CHAOS_API_URL/chaos/stop-all" || true
    fi
    
    # Archive old results (keep last 10 runs)
    find "$OUTPUT_DIR" -name "*.json" -type f | sort | head -n -50 | xargs rm -f 2>/dev/null || true
}

# Main execution
main() {
    log_info "Starting Load Testing and Performance Validation Suite"
    log_info "Timestamp: $TIMESTAMP"
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --api-only)
                RUN_API_ONLY=true
                shift
                ;;
            --media-only)
                RUN_MEDIA_ONLY=true
                shift
                ;;
            --chaos-only)
                RUN_CHAOS_ONLY=true
                shift
                ;;
            --benchmark-only)
                RUN_BENCHMARK_ONLY=true
                shift
                ;;
            --capacity-only)
                RUN_CAPACITY_ONLY=true
                shift
                ;;
            --endurance)
                RUN_ENDURANCE=true
                shift
                ;;
            --skip-report)
                SKIP_REPORT=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --api-only       Run only API load tests"
                echo "  --media-only     Run only media upload tests"
                echo "  --chaos-only     Run only chaos engineering tests"
                echo "  --benchmark-only Run only performance benchmarks"
                echo "  --capacity-only  Run only capacity planning"
                echo "  --endurance      Include 24-hour endurance test"
                echo "  --skip-report    Skip report generation"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Run selected test suites
    if [ "$RUN_API_ONLY" = "true" ]; then
        run_api_load_tests
    elif [ "$RUN_MEDIA_ONLY" = "true" ]; then
        run_media_upload_tests
    elif [ "$RUN_CHAOS_ONLY" = "true" ]; then
        run_chaos_tests
    elif [ "$RUN_BENCHMARK_ONLY" = "true" ]; then
        run_performance_benchmarks
    elif [ "$RUN_CAPACITY_ONLY" = "true" ]; then
        run_capacity_planning
    else
        # Run all tests
        run_api_load_tests
        run_media_upload_tests
        run_chaos_tests
        run_performance_benchmarks
        run_capacity_planning
    fi
    
    # Generate report unless skipped
    if [ "$SKIP_REPORT" != "true" ]; then
        generate_report
    fi
    
    log_success "Load testing and performance validation completed!"
    log_info "Results available in: $OUTPUT_DIR"
}

# Execute main function with all arguments
main "$@"