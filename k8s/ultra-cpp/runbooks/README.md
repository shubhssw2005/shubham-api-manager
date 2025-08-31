# Ultra C++ System Operational Runbooks

This directory contains operational runbooks for troubleshooting and maintaining the ultra-low-latency C++ system.

## Available Runbooks

1. **[System Health Monitoring](./system-health-monitoring.md)** - Monitor system health and performance metrics
2. **[Performance Troubleshooting](./performance-troubleshooting.md)** - Diagnose and resolve performance issues
3. **[DPDK Network Issues](./dpdk-network-issues.md)** - Troubleshoot DPDK networking problems
4. **[Memory Management](./memory-management.md)** - Handle memory-related issues and optimization
5. **[GPU Compute Issues](./gpu-compute-issues.md)** - Resolve GPU acceleration problems
6. **[Deployment Issues](./deployment-issues.md)** - Handle deployment and rollback scenarios
7. **[Emergency Procedures](./emergency-procedures.md)** - Critical incident response procedures
8. **[Maintenance Tasks](./maintenance-tasks.md)** - Regular maintenance and optimization tasks

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: Check PagerDuty rotation
- **Platform Team**: #platform-team Slack channel
- **Infrastructure Team**: #infrastructure Slack channel

### Critical Commands

```bash
# Check system health
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl http://localhost:8081/health

# Emergency rollback
./k8s/ultra-cpp/scripts/rollback.sh emergency

# View logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --tail=100 -f

# Check performance metrics
kubectl port-forward -n ultra-cpp svc/ultra-cpp-metrics 9090:9090
# Then visit http://localhost:9090/metrics
```

### Escalation Path

1. **Level 1**: Check runbooks and attempt standard remediation
2. **Level 2**: Engage on-call engineer via PagerDuty
3. **Level 3**: Escalate to platform team lead
4. **Level 4**: Engage infrastructure team and senior engineers

## Monitoring and Alerting

### Key Metrics to Monitor
- **Latency**: P99 response time < 1ms
- **Throughput**: QPS > 100k per pod
- **Error Rate**: < 0.1%
- **Memory Usage**: < 80% of allocated
- **CPU Usage**: < 70% average
- **DPDK Port Status**: All ports UP
- **GPU Utilization**: Available when needed

### Alert Thresholds
- **Critical**: P99 latency > 5ms, Error rate > 1%, Memory > 90%
- **Warning**: P99 latency > 2ms, Error rate > 0.5%, Memory > 80%

## Documentation Standards

Each runbook should include:
1. **Problem Description**: Clear description of the issue
2. **Symptoms**: How to identify the problem
3. **Diagnosis Steps**: Step-by-step troubleshooting
4. **Resolution**: How to fix the issue
5. **Prevention**: How to prevent recurrence
6. **Escalation**: When and how to escalate

## Contributing

When adding new runbooks:
1. Follow the template in `runbook-template.md`
2. Test all commands and procedures
3. Include relevant screenshots or diagrams
4. Update this README with the new runbook
5. Review with the platform team