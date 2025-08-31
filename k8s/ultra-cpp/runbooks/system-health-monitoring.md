# System Health Monitoring Runbook

## Overview
This runbook covers monitoring and maintaining the health of the ultra-low-latency C++ system.

## Health Check Endpoints

### Available Endpoints
- `/health/live` - Liveness probe (basic process health)
- `/health/ready` - Readiness probe (service ready to accept traffic)
- `/health/startup` - Startup probe (initialization complete)
- `/health` - Detailed health information (JSON)
- `/metrics` - Performance metrics (JSON)

### Checking Health Status

#### Quick Health Check
```bash
# Check if pods are running
kubectl get pods -n ultra-cpp -l app=ultra-cpp-gateway

# Check service health
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health/ready
```

#### Detailed Health Information
```bash
# Get detailed health report
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq .

# Check specific component health
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="memory")'
```

## Component Health Status

### Memory Component
**Healthy**: Memory usage < 80%
**Degraded**: Memory usage 80-90%
**Unhealthy**: Memory usage > 90%

```bash
# Check memory usage
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="memory") | .details'
```

### CPU Component
**Healthy**: Load average < 70% of CPU count
**Degraded**: Load average 70-90% of CPU count
**Unhealthy**: Load average > 90% of CPU count

```bash
# Check CPU load
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="cpu") | .details'
```

### Disk Component
**Healthy**: Disk usage < 85%
**Degraded**: Disk usage 85-95%
**Unhealthy**: Disk usage > 95%

```bash
# Check disk space
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="disk") | .details'
```

### Network Component
**Healthy**: All network interfaces up
**Degraded**: Some interfaces down
**Unhealthy**: No interfaces up

```bash
# Check network status
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="network") | .details'
```

### DPDK Component
**Healthy**: Hugepages available
**Degraded**: Limited hugepages
**Unhealthy**: No hugepages configured

```bash
# Check DPDK status
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="dpdk") | .details'

# Check hugepages directly
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/meminfo | grep -i huge
```

### GPU Component
**Healthy**: NVIDIA driver available
**Degraded**: Driver issues or limited resources
**Unhealthy**: No GPU driver

```bash
# Check GPU status
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="gpu") | .details'

# Check NVIDIA driver directly
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- nvidia-smi || echo "No NVIDIA GPU available"
```

## Performance Metrics

### Key Performance Indicators

#### Latency Metrics
```bash
# Get current latency metrics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep -E "(latency|response_time)"
```

#### Throughput Metrics
```bash
# Get throughput metrics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:9090/metrics | grep -E "(requests_per_second|qps)"
```

#### Resource Utilization
```bash
# Get resource utilization
kubectl top pods -n ultra-cpp -l app=ultra-cpp-gateway
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/metrics
```

## Monitoring Dashboard

### Prometheus Queries
```promql
# P99 Latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Request Rate
rate(http_requests_total[5m])

# Error Rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Memory Usage
(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100

# CPU Usage
rate(container_cpu_usage_seconds_total[5m]) * 100
```

### Grafana Dashboard
Access the Grafana dashboard at: `https://grafana.your-domain.com/d/ultra-cpp-system`

Key panels:
- Request latency (P50, P95, P99)
- Request rate and error rate
- Resource utilization (CPU, Memory)
- DPDK port status
- GPU utilization

## Alerting Rules

### Critical Alerts
- P99 latency > 5ms for 2 minutes
- Error rate > 1% for 1 minute
- Memory usage > 90% for 5 minutes
- All pods down for 30 seconds

### Warning Alerts
- P99 latency > 2ms for 5 minutes
- Error rate > 0.5% for 2 minutes
- Memory usage > 80% for 10 minutes
- CPU usage > 80% for 10 minutes

## Troubleshooting Common Issues

### High Latency
1. Check CPU and memory usage
2. Verify DPDK port status
3. Check for network congestion
4. Review application logs for errors

### High Error Rate
1. Check application logs for specific errors
2. Verify downstream service health
3. Check authentication/authorization issues
4. Review recent deployments

### Resource Exhaustion
1. Check for memory leaks in application logs
2. Verify resource limits and requests
3. Consider horizontal scaling
4. Review cache hit rates

### DPDK Issues
1. Verify hugepages allocation
2. Check DPDK port binding
3. Review network interface configuration
4. Check for hardware issues

## Maintenance Tasks

### Daily Checks
- Review health status of all components
- Check performance metrics trends
- Verify backup and monitoring systems
- Review error logs for patterns

### Weekly Checks
- Analyze performance trends
- Review capacity planning metrics
- Update runbooks based on incidents
- Test alerting and escalation procedures

### Monthly Checks
- Performance baseline review
- Capacity planning assessment
- Security patch review
- Disaster recovery testing

## Escalation Procedures

### Level 1 - Self-Service
- Check this runbook
- Review monitoring dashboards
- Attempt standard remediation

### Level 2 - On-Call Engineer
- Page on-call engineer via PagerDuty
- Provide incident details and attempted remediation
- Follow incident response procedures

### Level 3 - Platform Team
- Escalate to platform team lead
- Engage additional engineers as needed
- Consider emergency procedures

### Level 4 - Emergency Response
- Engage senior engineers and management
- Consider service degradation or shutdown
- Activate disaster recovery procedures

## Contact Information

- **PagerDuty**: [Your PagerDuty service key]
- **Slack**: #ultra-cpp-alerts, #platform-team
- **Documentation**: [Link to internal docs]
- **Monitoring**: [Link to monitoring dashboard]