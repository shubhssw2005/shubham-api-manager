# Performance Troubleshooting Runbook

## Overview
This runbook provides step-by-step procedures for diagnosing and resolving performance issues in the ultra-low-latency C++ system.

## Performance Targets
- **P99 Latency**: < 1ms for cached requests
- **P50 Latency**: < 500μs for cached requests
- **Throughput**: > 100k QPS per pod
- **Error Rate**: < 0.1%
- **CPU Usage**: < 70% average
- **Memory Usage**: < 80% of allocated

## Symptoms and Diagnosis

### High Latency Issues

#### Symptoms
- P99 latency > 2ms consistently
- Slow response times reported by clients
- Increased timeout errors

#### Diagnosis Steps

1. **Check Current Latency Metrics**
```bash
# Get current latency distribution
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep latency

# Check health endpoint response time
time kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health > /dev/null
```

2. **Identify Bottlenecks**
```bash
# Check CPU usage
kubectl top pods -n ultra-cpp -l app=ultra-cpp-gateway

# Check memory usage and cache hit rates
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep -E "(cache_hit|memory_usage)"

# Check DPDK port statistics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/net/dev
```

3. **Analyze Application Performance**
```bash
# Check for lock contention
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep lock

# Check queue depths
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep queue_depth

# Review garbage collection metrics (if applicable)
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep gc
```

#### Resolution Steps

1. **CPU Optimization**
```bash
# Check CPU affinity settings
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- taskset -p 1

# Verify NUMA topology
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- numactl --show

# Check for CPU throttling
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /sys/fs/cgroup/cpu/cpu.stat
```

2. **Memory Optimization**
```bash
# Check memory allocation patterns
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/meminfo

# Verify hugepages usage
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/meminfo | grep -i huge

# Check for memory fragmentation
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/buddyinfo
```

3. **Cache Optimization**
```bash
# Check cache hit rates
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep cache_hit_rate

# Analyze cache eviction patterns
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep cache_evictions

# Review cache warming status
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep cache_warming
```

### Low Throughput Issues

#### Symptoms
- QPS below 100k per pod
- Request queuing and backlog
- Client connection timeouts

#### Diagnosis Steps

1. **Check Request Processing**
```bash
# Monitor request rate
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep requests_per_second

# Check connection pool status
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep connection_pool

# Analyze request distribution
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep request_distribution
```

2. **Network Performance**
```bash
# Check network interface statistics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/net/dev

# Monitor DPDK port statistics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep dpdk_port

# Check for packet drops
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- netstat -i
```

#### Resolution Steps

1. **Scale Resources**
```bash
# Increase pod replicas
kubectl scale deployment ultra-cpp-gateway-blue -n ultra-cpp --replicas=5

# Adjust resource limits
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp -p '{"spec":{"template":{"spec":{"containers":[{"name":"ultra-cpp-gateway","resources":{"limits":{"cpu":"8","memory":"16Gi"}}}]}}}}'
```

2. **Optimize Configuration**
```bash
# Update worker thread count
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[api_gateway]\nworker_threads = 8\n..."}}'

# Adjust memory pool size
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[api_gateway]\nmemory_pool_size = 4294967296\n..."}}'
```

### High Error Rate Issues

#### Symptoms
- Error rate > 0.5%
- 5xx HTTP status codes
- Client connection failures

#### Diagnosis Steps

1. **Analyze Error Patterns**
```bash
# Check error distribution by status code
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep http_requests_total | grep -E "5[0-9][0-9]"

# Review application logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --tail=100 | grep -i error

# Check error rate trends
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep error_rate
```

2. **Identify Root Causes**
```bash
# Check downstream service health
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -f http://nodejs-api:3005/health

# Verify authentication/authorization
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --tail=100 | grep -i "auth\|unauthorized\|forbidden"

# Check resource exhaustion
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.status != "healthy")'
```

## Performance Profiling

### CPU Profiling
```bash
# Generate CPU profile
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- perf record -g -p $(pgrep ultra-cpp-gateway) sleep 30

# Analyze CPU hotspots
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- perf report --stdio
```

### Memory Profiling
```bash
# Check memory allocation patterns
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- valgrind --tool=massif --massif-out-file=/tmp/massif.out ./ultra-cpp-gateway

# Analyze memory usage
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- ms_print /tmp/massif.out
```

### Network Profiling
```bash
# Monitor network traffic
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- tcpdump -i any -c 1000 -w /tmp/network.pcap

# Analyze packet patterns
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- tcpdump -r /tmp/network.pcap -nn
```

## Load Testing

### Stress Testing
```bash
# Run load test with wrk
kubectl run load-test --image=williamyeh/wrk --rm -it --restart=Never -- \
  -t 12 -c 400 -d 30s --latency http://ultra-cpp-gateway.ultra-cpp.svc.cluster.local/health

# Run sustained load test
kubectl run sustained-load --image=williamyeh/wrk --rm -it --restart=Never -- \
  -t 8 -c 200 -d 300s --latency http://ultra-cpp-gateway.ultra-cpp.svc.cluster.local/api/v1/status
```

### Benchmark Comparison
```bash
# Baseline performance test
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- /usr/local/bin/benchmark --baseline

# Compare with previous baseline
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- /usr/local/bin/benchmark --compare-baseline
```

## Performance Optimization

### Compiler Optimizations
```bash
# Check if PGO (Profile-Guided Optimization) is enabled
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- strings /usr/local/bin/ultra-cpp-gateway | grep -i pgo

# Verify optimization flags
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- objdump -s --section .comment /usr/local/bin/ultra-cpp-gateway
```

### Runtime Optimizations
```bash
# Adjust CPU governor
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cpupower frequency-set --governor performance

# Set CPU affinity for optimal NUMA placement
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- taskset -cp 0-7 $(pgrep ultra-cpp-gateway)

# Configure transparent hugepages
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

### Cache Optimizations
```bash
# Warm up cache with common requests
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/cache/warm

# Adjust cache size based on hit rates
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[cache]\ncapacity = 2000000\n..."}}'

# Enable cache preloading
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[cache]\nenable_preloading = true\n..."}}'
```

## Monitoring and Alerting

### Performance Dashboards
- **Latency Dashboard**: Monitor P50, P95, P99 latencies
- **Throughput Dashboard**: Track QPS and request patterns
- **Resource Dashboard**: CPU, memory, network utilization
- **Error Dashboard**: Error rates and patterns

### Key Alerts
```yaml
# High latency alert
- alert: HighLatency
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.002
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High P99 latency detected"

# Low throughput alert
- alert: LowThroughput
  expr: rate(http_requests_total[5m]) < 100000
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Throughput below target"
```

## Escalation Procedures

### Performance Degradation
1. **Immediate Actions**
   - Check system health and resource usage
   - Review recent changes and deployments
   - Consider emergency rollback if needed

2. **Investigation**
   - Collect performance profiles and metrics
   - Analyze logs for error patterns
   - Compare with baseline performance

3. **Escalation**
   - Engage performance engineering team
   - Consider service degradation if critical
   - Implement temporary workarounds

### Critical Performance Issues
1. **Emergency Response**
   - Activate incident response procedures
   - Consider traffic shifting or service shutdown
   - Engage senior engineers and management

2. **Recovery**
   - Implement immediate fixes or rollbacks
   - Monitor recovery metrics
   - Conduct post-incident review

## Prevention Strategies

### Performance Testing
- Implement continuous performance testing in CI/CD
- Establish performance baselines and regression detection
- Regular load testing and capacity planning

### Monitoring and Alerting
- Comprehensive performance monitoring
- Proactive alerting on performance degradation
- Regular review of performance trends

### Optimization
- Regular performance profiling and optimization
- Hardware and software tuning
- Capacity planning and scaling strategies