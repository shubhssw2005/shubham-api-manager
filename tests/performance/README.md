# Load Testing and Performance Validation Suite

This comprehensive testing suite validates the performance, scalability, and resilience of the AWS deployment system according to the requirements specified in the system design.

## Overview

The suite includes:

1. **API Load Testing** - Validates API performance under various load conditions
2. **Media Upload Testing** - Tests media processing pipeline performance
3. **Chaos Engineering** - Validates system resilience through failure injection
4. **Performance Benchmarking** - Measures system performance across multiple dimensions
5. **Capacity Planning** - Analyzes current usage and projects future resource needs

## Requirements Validation

This test suite validates the following requirements from the AWS deployment system spec:

- **Requirement 9.1**: Sustain 100k+ QPS with P99 latency within SLOs
- **Requirement 9.2**: Validate presigned URL flows and multipart uploads
- **Requirement 9.3**: Validate regional failover scenarios
- **Requirement 9.4**: Inject failures and validate recovery procedures
- **Requirement 9.5**: Run sustained load for 24-72 hours without degradation

## Prerequisites

### Software Requirements

1. **k6** - Load testing tool
   ```bash
   # macOS
   brew install k6
   
   # Linux
   sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
   echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
   sudo apt-get update
   sudo apt-get install k6
   ```

2. **Node.js** - For capacity planning tool
   ```bash
   # Ensure Node.js 16+ is installed
   node --version
   ```

3. **jq** - For JSON processing in reports
   ```bash
   # macOS
   brew install jq
   
   # Linux
   sudo apt-get install jq
   ```

### Infrastructure Requirements

1. **Target System** - The API system should be running and accessible
2. **Monitoring Stack** - Prometheus/Grafana for metrics collection (optional)
3. **Chaos Engineering Tools** - Chaos Mesh or Litmus for failure injection (optional)

## Quick Start

### 1. Basic Load Test

Run all test scenarios:

```bash
./scripts/run-load-tests.sh
```

### 2. Specific Test Types

Run only API load tests:
```bash
./scripts/run-load-tests.sh --api-only
```

Run only media upload tests:
```bash
./scripts/run-load-tests.sh --media-only
```

Run only chaos engineering tests:
```bash
./scripts/run-load-tests.sh --chaos-only
```

Run only performance benchmarks:
```bash
./scripts/run-load-tests.sh --benchmark-only
```

Run only capacity planning:
```bash
./scripts/run-load-tests.sh --capacity-only
```

### 3. Configuration

Set environment variables:

```bash
export API_BASE_URL="https://your-api.example.com"
export JWT_TOKEN="your-jwt-token"
export TENANT_ID="your-tenant-id"
export METRICS_URL="http://your-prometheus:9090"
export CHAOS_API_URL="http://your-chaos-mesh:8080"
```

## Test Scenarios

### API Load Testing

#### Baseline Load Test
- **Purpose**: Establish performance baseline
- **Load**: 50 concurrent users for 5 minutes
- **Validates**: Normal operation performance

#### Stress Test
- **Purpose**: Find system breaking point
- **Load**: Gradual increase from 100 to 5,000 users
- **Duration**: 47 minutes total
- **Validates**: System behavior under increasing load

#### Spike Test
- **Purpose**: Test sudden load increases
- **Load**: Sudden spike from 100 to 2,000 users
- **Duration**: 5 minutes total
- **Validates**: Auto-scaling and load balancing

#### Soak Test
- **Purpose**: Test sustained load performance
- **Load**: 200 concurrent users for 1 hour
- **Validates**: Memory leaks and performance degradation

### Media Upload Testing

#### Concurrent Upload Test
- **Purpose**: Test parallel upload handling
- **Load**: 50 concurrent uploads for 10 minutes
- **Validates**: S3 presigned URL performance

#### Large File Test
- **Purpose**: Test multipart upload handling
- **Load**: 5 concurrent large file uploads (50-200MB)
- **Duration**: 15 minutes
- **Validates**: Multipart upload reliability

#### Upload Stress Test
- **Purpose**: Test upload pipeline under stress
- **Load**: Gradual increase to 200 concurrent uploads
- **Validates**: Media processing pipeline scalability

### Chaos Engineering Tests

#### Database Chaos
- **Failure**: Database pod failure
- **Duration**: 2 minutes
- **Validates**: Database failover and recovery

#### Cache Chaos
- **Failure**: Redis network partition
- **Duration**: 1.5 minutes
- **Validates**: Cache failure graceful degradation

#### Storage Chaos
- **Failure**: S3 network latency injection
- **Duration**: 3 minutes
- **Validates**: Storage latency handling

#### Network Chaos
- **Failure**: API service network partition
- **Duration**: 2.5 minutes
- **Validates**: Network failure recovery

#### Pod Chaos
- **Failure**: Random API pod termination
- **Duration**: 1.67 minutes
- **Validates**: Pod failure and load balancing

### Performance Benchmarks

#### Baseline Benchmark
- **Purpose**: Measure baseline performance metrics
- **Load**: 50 users for 5 minutes
- **Metrics**: Response time, throughput, resource efficiency

#### Capacity Planning Benchmark
- **Purpose**: Determine maximum system capacity
- **Load**: Gradual increase to 10,000 users
- **Metrics**: Capacity utilization, scaling behavior

#### Scalability Benchmark
- **Purpose**: Test exponential load scaling
- **Load**: Exponential growth pattern
- **Metrics**: Scalability index, performance degradation

#### Efficiency Benchmark
- **Purpose**: Measure resource efficiency
- **Load**: 1,000 users for 10 minutes
- **Metrics**: CPU/memory/network efficiency

#### Endurance Benchmark (Optional)
- **Purpose**: Long-term stability testing
- **Load**: 500 users for 24 hours
- **Metrics**: Performance degradation over time

## Configuration

### Test Configuration

Edit `tests/performance/load-test-config.json` to customize:

- **SLO Targets**: Performance thresholds
- **Load Patterns**: User counts and durations
- **Chaos Experiments**: Failure types and durations
- **Resource Limits**: Capacity planning parameters

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | Target API base URL | `http://localhost:3000` |
| `JWT_TOKEN` | Authentication token | Empty (will attempt login) |
| `TENANT_ID` | Tenant identifier for multi-tenant testing | `test-tenant` |
| `METRICS_URL` | Prometheus metrics endpoint | `http://localhost:9090` |
| `CHAOS_API_URL` | Chaos engineering API endpoint | `http://localhost:8080` |
| `RUN_ENDURANCE` | Include 24-hour endurance test | `false` |

## Results and Reporting

### Output Files

All results are saved to `tests/performance/results/`:

- **JSON Results**: Raw k6 output in JSON format
- **Performance Report**: Consolidated markdown report
- **Capacity Plan**: Detailed capacity planning analysis

### Metrics Collected

#### Performance Metrics
- Request rate (RPS)
- Response time percentiles (P50, P95, P99)
- Error rate
- Throughput
- Resource utilization

#### Resilience Metrics
- Recovery time
- Resilience score
- Error budget consumption
- Failure injection success rate

#### Capacity Metrics
- Performance score
- Capacity utilization
- Resource efficiency
- Scalability index

### SLO Validation

The tests validate against these SLO targets:

- **Availability**: 99.99%
- **P95 Latency**: < 200ms
- **P99 Latency**: < 500ms
- **Error Rate**: < 1%
- **Throughput**: 100,000+ RPS

## Capacity Planning

The capacity planning tool analyzes:

1. **Current Performance**: Real-time metrics collection
2. **Historical Trends**: Performance trend analysis
3. **Growth Projections**: Future load predictions
4. **Resource Requirements**: Infrastructure sizing
5. **Cost Projections**: Budget planning
6. **Scaling Recommendations**: Action items

### Running Capacity Planning

```bash
# Run standalone capacity planning
node scripts/capacity-planning-tool.js

# Or as part of full test suite
./scripts/run-load-tests.sh --capacity-only
```

### Capacity Planning Output

- `tests/performance/capacity-plan.json` - Machine-readable plan
- `tests/performance/capacity-plan-report.md` - Human-readable report

## Troubleshooting

### Common Issues

#### k6 Installation Issues
```bash
# Verify k6 installation
k6 version

# Test k6 with simple script
k6 run --vus 1 --duration 10s -e K6_NO_SETUP=true tests/load/k6-api-load-test.js
```

#### API Connection Issues
```bash
# Test API connectivity
curl -v $API_BASE_URL/health

# Check authentication
curl -H "Authorization: Bearer $JWT_TOKEN" $API_BASE_URL/api/auth/me
```

#### Missing Dependencies
```bash
# Install all dependencies
npm install

# Verify Node.js version
node --version  # Should be 16+
```

### Performance Issues

#### High Response Times
1. Check system resource utilization
2. Verify database performance
3. Check network latency
4. Review application logs

#### Test Failures
1. Verify SLO thresholds are realistic
2. Check system capacity
3. Review error logs
4. Validate test configuration

## Integration with CI/CD

### GitHub Actions Integration

Add to `.github/workflows/performance-tests.yml`:

```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install k6
        run: |
          sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
      
      - name: Run Performance Tests
        env:
          API_BASE_URL: ${{ secrets.API_BASE_URL }}
          JWT_TOKEN: ${{ secrets.JWT_TOKEN }}
        run: ./scripts/run-load-tests.sh --api-only
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: tests/performance/results/
```

### Monitoring Integration

#### Prometheus Metrics
The tests can push metrics to Prometheus:

```bash
k6 run --out influxdb=http://localhost:8086/k6 tests/load/k6-api-load-test.js
```

#### Grafana Dashboards
Import the provided Grafana dashboards for visualization:

- Performance Dashboard
- Chaos Engineering Dashboard
- Capacity Planning Dashboard

## Best Practices

### Test Design
1. **Realistic Load Patterns**: Model actual user behavior
2. **Gradual Load Increase**: Avoid sudden load spikes in stress tests
3. **Proper Think Time**: Include realistic delays between requests
4. **Data Variation**: Use varied test data to avoid caching artifacts

### Execution
1. **Baseline First**: Always establish baseline before stress testing
2. **System Recovery**: Allow time between tests for system recovery
3. **Monitoring**: Monitor system resources during tests
4. **Repeatability**: Ensure tests are repeatable and deterministic

### Analysis
1. **SLO Focus**: Validate against defined SLOs
2. **Trend Analysis**: Look for performance trends over time
3. **Bottleneck Identification**: Identify and address performance bottlenecks
4. **Capacity Planning**: Use results for future capacity planning

## Contributing

### Adding New Tests

1. Create test script in appropriate directory
2. Update configuration in `load-test-config.json`
3. Add scenario to `run-load-tests.sh`
4. Update documentation

### Test Script Structure

```javascript
// Standard k6 test structure
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export let options = {
  // Test configuration
};

export function setup() {
  // Test setup
}

export default function() {
  // Test execution
}

export function teardown() {
  // Test cleanup
}
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review test logs in `tests/performance/results/`
3. Validate configuration in `load-test-config.json`
4. Check system resources and connectivity

## License

This testing suite is part of the AWS deployment system project and follows the same license terms.