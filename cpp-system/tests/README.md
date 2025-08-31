# Ultra Low-Latency System Testing Framework

This comprehensive testing framework provides unit tests, performance benchmarks, load testing, and chaos engineering capabilities for the ultra low-latency C++ system.

## Overview

The testing framework consists of four main components:

1. **Unit Tests** - Comprehensive test suite using Google Test
2. **Performance Benchmarks** - Detailed performance analysis using Google Benchmark
3. **Load Testing** - Configurable traffic pattern testing with real HTTP requests
4. **Chaos Testing** - Failure injection and resilience validation

## Quick Start

### Building the Tests

```bash
# From the cpp-system directory
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd tests
```

### Running All Tests

```bash
# Run comprehensive test suite
./ultra_test_runner --test-type all --duration 60

# Run specific test types
./ultra_test_runner --test-type unit
./ultra_test_runner --test-type benchmark
./ultra_test_runner --test-type load --target http://localhost:8080
./ultra_test_runner --test-type chaos --enable-destructive
```

### Using Test Scripts

```bash
# Performance tests
./scripts/run_performance_tests.sh

# Load tests
./scripts/run_load_tests.sh --target http://localhost:8080 --duration 120

# Chaos tests (simulation mode)
./scripts/run_chaos_tests.sh --duration 300

# Chaos tests (destructive mode - use with caution!)
./scripts/run_chaos_tests.sh --duration 300 --enable-destructive
```

## Test Components

### 1. Unit Tests (`comprehensive_unit_tests`)

Validates core functionality with performance requirements:

- **Logger Performance**: Sub-microsecond logging latency
- **Error Handling**: Exception handling under 500ns
- **NUMA Allocator**: 100ns allocation latency for small objects
- **Atomic Reference Counting**: 50ns increment/decrement operations
- **Ultra Cache**: Sub-100ns get operations, 200ns put operations
- **Memory Leak Detection**: Automatic leak detection in TearDown
- **Stress Testing**: Multi-threaded concurrent operations
- **Edge Case Handling**: Boundary conditions and error scenarios

#### Key Features:
- Automatic memory leak detection
- Performance regression testing
- Concurrent operation validation
- Hardware-specific optimizations testing

### 2. Performance Benchmarks (`comprehensive_benchmarks`)

Detailed performance analysis across multiple dimensions:

#### Cache Benchmarks:
- Get/Put operations across different cache sizes
- Mixed read/write workloads (80/20 ratio)
- Concurrent access patterns
- Cache eviction performance

#### Memory Benchmarks:
- Small allocation performance (64B - 4KB)
- Aligned allocation performance (8B - 256B alignment)
- NUMA-aware allocation strategies

#### Lock-Free Benchmarks:
- Atomic reference counting operations
- Mixed increment/decrement patterns
- Contention scenarios

#### System Benchmarks:
- Logging performance across log levels
- Memory bandwidth measurements
- CPU cache effects analysis
- Latency distribution analysis (P50, P95, P99, P99.9)

### 3. Load Testing (`load_tests`)

Configurable HTTP load testing with multiple traffic patterns:

#### Traffic Patterns:
- **Constant**: Steady request rate
- **Ramp Up**: Gradually increasing load
- **Ramp Down**: Gradually decreasing load
- **Spike**: Sudden traffic spikes
- **Burst**: Periodic traffic bursts
- **Random**: Random request intervals
- **Realistic Web**: Simulated web traffic patterns

#### Features:
- Real HTTP requests using libcurl
- Configurable worker threads
- Response validation
- SLA compliance checking
- Real-time progress monitoring
- Comprehensive HTML reports

#### Example Usage:
```bash
# Constant load test
./load_tests --test-type constant --rps 1000 --duration 60

# Spike test
./load_tests --test-type spike --rps 500 --peak-rps 5000 --duration 120

# Stress test
./load_tests --test-type stress --target http://localhost:8080
```

### 4. Chaos Testing (`chaos_tests`)

Failure injection and resilience validation:

#### Chaos Experiments:
- **Network Chaos**: Latency injection, packet loss, bandwidth limits
- **Resource Chaos**: Memory pressure, CPU stress, disk I/O failures
- **Application Chaos**: Exception injection, timeout injection, memory leaks
- **Infrastructure Chaos**: Process kills, thread starvation

#### Safety Features:
- Configurable safety limits
- Automatic recovery mechanisms
- System resource monitoring
- Non-destructive simulation mode

#### Example Usage:
```bash
# Safe simulation mode
./chaos_tests --experiment network_latency --duration 300

# Destructive mode (test environments only!)
./chaos_tests --experiment all --duration 600 --enable-destructive
```

## Test Framework Architecture

### Core Components

#### `TestFramework` (`framework/test_framework.hpp`)
- Base testing utilities and fixtures
- Performance measurement tools
- Memory tracking and leak detection
- Chaos injection utilities

#### `LoadTestSuite` (`load/load_test_suite.hpp`)
- HTTP client implementation
- Traffic pattern generators
- Load test execution engine
- Report generation

#### `ChaosTestingFramework` (`chaos/chaos_testing_framework.hpp`)
- Network chaos injection
- Resource exhaustion simulation
- Application failure injection
- System monitoring and recovery

### Test Fixtures

#### `UltraTestFixture`
Base fixture for unit tests with:
- Automatic memory leak detection
- Performance assertion helpers
- Concurrent test execution utilities
- Chaos injection integration

#### `UltraBenchmarkFixture`
Base fixture for benchmarks with:
- Memory pool setup
- Test data generation
- Resource cleanup
- Performance measurement helpers

## Performance Requirements

The testing framework validates these performance targets:

| Component | Operation | Target Latency | Target Throughput |
|-----------|-----------|----------------|-------------------|
| Logger | Info/Debug | < 1μs | > 100K ops/sec |
| Cache | Get | < 100ns | > 1M ops/sec |
| Cache | Put | < 200ns | > 500K ops/sec |
| Memory | Small Alloc | < 100ns | > 10M ops/sec |
| RefCount | Inc/Dec | < 50ns | > 10M ops/sec |
| API Gateway | GET Request | < 500μs | > 1M QPS |

## Report Generation

### HTML Reports
- Interactive performance dashboards
- Latency distribution charts
- Throughput over time graphs
- Error rate analysis

### JSON Reports
- Machine-readable test results
- Integration with CI/CD pipelines
- Historical trend analysis
- Automated alerting

### CSV Reports
- Raw performance data
- Statistical analysis
- Custom visualization support

## Integration with CI/CD

### GitHub Actions Integration
```yaml
- name: Run Performance Tests
  run: |
    cd cpp-system/build/tests
    ./scripts/run_performance_tests.sh
    
- name: Run Load Tests
  run: |
    cd cpp-system/build/tests
    ./scripts/run_load_tests.sh --duration 30
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: cpp-system/build/tests/test_results/
```

### Performance Regression Detection
The framework automatically detects performance regressions by comparing current results against historical baselines.

## Configuration

### Environment Variables
```bash
export ULTRA_TEST_TARGET="http://localhost:8080"
export ULTRA_TEST_DURATION="60"
export ULTRA_TEST_THREADS="4"
export ULTRA_ENABLE_CHAOS="false"
```

### CMake Options
```bash
cmake .. -DENABLE_TESTING=ON \
         -DENABLE_BENCHMARKS=ON \
         -DENABLE_LOAD_TESTS=ON \
         -DENABLE_CHAOS_TESTS=ON
```

## Safety Considerations

### Chaos Testing Safety
- **Never run destructive chaos tests in production**
- Use simulation mode for development
- Set appropriate safety limits
- Monitor system resources
- Have recovery procedures ready

### Resource Limits
```cpp
// Example safety configuration
std::map<std::string, double> safety_limits = {
    {"cpu_usage", 90.0},        // Max 90% CPU usage
    {"memory_usage", 8192.0},   // Max 8GB memory usage
    {"error_rate", 0.5}         // Max 50% error rate
};
```

## Troubleshooting

### Common Issues

#### Build Errors
```bash
# Missing dependencies
sudo apt-get install libgtest-dev libbenchmark-dev libcurl4-openssl-dev

# CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
```

#### Test Failures
```bash
# Check system resources
free -h
top
df -h

# Verify target endpoint
curl -I http://localhost:8080

# Run with verbose output
./ultra_test_runner --verbose --test-type unit
```

#### Performance Issues
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set performance mode
sudo cpupower frequency-set --governor performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Contributing

### Adding New Tests
1. Create test files in appropriate subdirectories
2. Follow naming conventions (`test_*.cpp`, `benchmark_*.cpp`)
3. Use provided fixtures and utilities
4. Update CMakeLists.txt
5. Add documentation

### Performance Test Guidelines
- Use realistic workloads
- Measure multiple metrics (latency, throughput, resource usage)
- Include warm-up periods
- Test under various load conditions
- Validate against requirements

### Chaos Test Guidelines
- Start with simulation mode
- Implement proper recovery mechanisms
- Monitor system impact
- Document expected behaviors
- Test incrementally

## License

This testing framework is part of the Ultra Low-Latency System project and follows the same licensing terms.