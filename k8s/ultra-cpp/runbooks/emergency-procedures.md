# Emergency Procedures Runbook

## Overview
This runbook contains critical emergency procedures for the ultra-low-latency C++ system. These procedures should be followed during severe incidents that threaten system availability or data integrity.

## Emergency Contacts

### Immediate Response Team
- **On-Call Engineer**: PagerDuty rotation
- **Platform Team Lead**: [Phone/Slack]
- **Infrastructure Team**: #infrastructure-emergency
- **Security Team**: #security-incidents (for security-related issues)

### Escalation Chain
1. **Level 1**: On-call engineer
2. **Level 2**: Platform team lead + Senior engineers
3. **Level 3**: Engineering management + Infrastructure team
4. **Level 4**: CTO + Executive team

## Critical System Failure

### Symptoms
- All pods in CrashLoopBackOff state
- Complete service unavailability
- P99 latency > 10ms for extended period
- Error rate > 10%

### Immediate Actions (< 5 minutes)

1. **Assess Impact**
```bash
# Check pod status
kubectl get pods -n ultra-cpp -l app=ultra-cpp-gateway

# Check service availability
kubectl get svc ultra-cpp-gateway -n ultra-cpp
curl -f http://ultra-cpp-gateway.ultra-cpp.svc.cluster.local/health/live || echo "Service DOWN"

# Check recent events
kubectl get events -n ultra-cpp --sort-by='.lastTimestamp' | tail -20
```

2. **Emergency Rollback**
```bash
# Immediate rollback to previous deployment
./k8s/ultra-cpp/scripts/rollback.sh emergency

# Verify rollback success
kubectl get pods -n ultra-cpp -l app=ultra-cpp-gateway
curl -f http://ultra-cpp-gateway.ultra-cpp.svc.cluster.local/health/ready
```

3. **Traffic Diversion**
```bash
# Divert traffic to Node.js fallback if rollback fails
kubectl patch service ultra-cpp-gateway -n ultra-cpp -p '{"spec":{"selector":{"app":"nodejs-api"}}}'

# Verify traffic diversion
kubectl get svc ultra-cpp-gateway -n ultra-cpp -o yaml | grep selector
```

### Investigation (5-15 minutes)

1. **Collect Diagnostics**
```bash
# Collect pod logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --previous --tail=500 > /tmp/ultra-cpp-crash-logs.txt

# Collect system events
kubectl get events -n ultra-cpp --sort-by='.lastTimestamp' > /tmp/ultra-cpp-events.txt

# Collect resource usage
kubectl top pods -n ultra-cpp > /tmp/ultra-cpp-resources.txt
```

2. **Check Infrastructure**
```bash
# Check node health
kubectl get nodes -o wide
kubectl describe nodes | grep -A 5 -B 5 "Conditions\|Taints"

# Check cluster resources
kubectl top nodes
kubectl get pods --all-namespaces | grep -v Running
```

3. **Analyze Root Cause**
```bash
# Check for OOM kills
kubectl describe pods -n ultra-cpp -l app=ultra-cpp-gateway | grep -A 10 -B 10 "OOMKilled\|Killed"

# Check for resource exhaustion
kubectl describe nodes | grep -A 10 "Allocated resources"

# Check for configuration issues
kubectl get configmap ultra-cpp-config -n ultra-cpp -o yaml
```

## Memory Exhaustion Emergency

### Symptoms
- Pods being OOMKilled repeatedly
- Memory usage > 95%
- Swap usage increasing rapidly

### Immediate Actions

1. **Emergency Memory Relief**
```bash
# Scale down to reduce memory pressure
kubectl scale deployment ultra-cpp-gateway-blue -n ultra-cpp --replicas=1
kubectl scale deployment ultra-cpp-gateway-green -n ultra-cpp --replicas=1

# Clear cache if possible
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/cache/clear || true
```

2. **Increase Memory Limits**
```bash
# Emergency memory limit increase
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp -p '{"spec":{"template":{"spec":{"containers":[{"name":"ultra-cpp-gateway","resources":{"limits":{"memory":"16Gi"}}}]}}}}'

# Wait for rollout
kubectl rollout status deployment/ultra-cpp-gateway-blue -n ultra-cpp --timeout=300s
```

3. **Monitor Recovery**
```bash
# Monitor memory usage
watch kubectl top pods -n ultra-cpp -l app=ultra-cpp-gateway

# Check for memory leaks
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.components[] | select(.name=="memory")'
```

## DPDK Network Failure

### Symptoms
- DPDK ports showing as DOWN
- Network connectivity issues
- Hugepages allocation failures

### Immediate Actions

1. **Check DPDK Status**
```bash
# Check hugepages
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- cat /proc/meminfo | grep -i huge

# Check DPDK port binding
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- ls -la /dev/uio*

# Check network interfaces
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- ip link show
```

2. **Emergency Network Recovery**
```bash
# Restart networking (if safe)
kubectl delete pods -n ultra-cpp -l app=ultra-cpp-gateway --force --grace-period=0

# Fallback to kernel networking
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[api_gateway]\nenable_dpdk = false\n..."}}'

# Restart deployments
kubectl rollout restart deployment/ultra-cpp-gateway-blue -n ultra-cpp
```

3. **Verify Recovery**
```bash
# Check network connectivity
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -f http://nodejs-api:3005/health

# Test external connectivity
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -f https://httpbin.org/status/200
```

## Security Incident Response

### Symptoms
- Suspicious network traffic
- Unauthorized access attempts
- Security alerts from monitoring systems

### Immediate Actions

1. **Isolate System**
```bash
# Block external traffic
kubectl patch service ultra-cpp-gateway -n ultra-cpp -p '{"spec":{"type":"ClusterIP"}}'

# Enable security monitoring
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[security]\nenable_audit_logging = true\nlog_all_requests = true\n..."}}'
```

2. **Collect Evidence**
```bash
# Collect security logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --tail=1000 | grep -i "security\|auth\|unauthorized" > /tmp/security-logs.txt

# Collect network traffic
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- tcpdump -i any -w /tmp/security-traffic.pcap -c 10000
```

3. **Engage Security Team**
```bash
# Notify security team
echo "Security incident detected in ultra-cpp system. Logs collected at /tmp/security-logs.txt" | slack-notify #security-incidents

# Preserve evidence
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- tar -czf /tmp/incident-evidence.tar.gz /var/log/ultra-cpp/ /tmp/security-*
```

## Data Corruption Emergency

### Symptoms
- Cache returning incorrect data
- Data integrity check failures
- Inconsistent responses

### Immediate Actions

1. **Stop Processing**
```bash
# Scale down to prevent further corruption
kubectl scale deployment ultra-cpp-gateway-blue -n ultra-cpp --replicas=0
kubectl scale deployment ultra-cpp-gateway-green -n ultra-cpp --replicas=0

# Redirect to fallback system
kubectl patch service ultra-cpp-gateway -n ultra-cpp -p '{"spec":{"selector":{"app":"nodejs-api"}}}'
```

2. **Assess Damage**
```bash
# Check cache integrity
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/cache/verify || true

# Check data consistency
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/data/verify || true
```

3. **Recovery Actions**
```bash
# Clear corrupted cache
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/cache/clear

# Restore from backup if available
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -X POST http://localhost:8080/admin/cache/restore

# Restart with clean state
kubectl rollout restart deployment/ultra-cpp-gateway-blue -n ultra-cpp
```

## Communication Procedures

### Incident Declaration

1. **Severity Levels**
   - **SEV1**: Complete service outage, data loss risk
   - **SEV2**: Significant performance degradation, partial outage
   - **SEV3**: Minor issues, workarounds available

2. **Communication Channels**
```bash
# Slack notifications
slack-notify #ultra-cpp-alerts "SEV1: Ultra C++ system complete outage. Emergency procedures activated."
slack-notify #engineering-all "Ultra C++ system incident declared. Updates in #ultra-cpp-alerts"

# PagerDuty escalation
pagerduty-trigger --service-key=ULTRA_CPP_SERVICE --incident-key=emergency-$(date +%s) --description="Ultra C++ Emergency"
```

3. **Status Page Updates**
```bash
# Update status page
curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -d '{"incident":{"name":"Ultra C++ System Issues","status":"investigating","impact":"major"}}'
```

### Regular Updates

1. **Every 15 minutes during SEV1**
2. **Every 30 minutes during SEV2**
3. **Hourly during SEV3**

### Resolution Communication
```bash
# Resolution notification
slack-notify #ultra-cpp-alerts "RESOLVED: Ultra C++ system restored. Root cause: [DESCRIPTION]. Post-incident review scheduled."

# Close PagerDuty incident
pagerduty-resolve --incident-key=emergency-TIMESTAMP
```

## Post-Incident Procedures

### Immediate Post-Resolution (< 1 hour)

1. **Verify Full Recovery**
```bash
# Comprehensive health check
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl -s http://localhost:8081/health | jq '.overall_status'

# Performance validation
./k8s/ultra-cpp/scripts/performance-test.sh --quick-validation

# Monitor for 30 minutes
watch kubectl top pods -n ultra-cpp -l app=ultra-cpp-gateway
```

2. **Preserve Evidence**
```bash
# Archive incident logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --since=2h > /tmp/incident-logs-$(date +%s).txt

# Save configuration state
kubectl get all,configmaps,secrets -n ultra-cpp -o yaml > /tmp/incident-config-$(date +%s).yaml
```

### Post-Incident Review (< 24 hours)

1. **Schedule PIR Meeting**
2. **Prepare Timeline**
3. **Identify Root Cause**
4. **Document Lessons Learned**
5. **Create Action Items**

### Follow-up Actions (< 1 week)

1. **Implement Preventive Measures**
2. **Update Runbooks**
3. **Improve Monitoring/Alerting**
4. **Conduct Training if Needed**

## Emergency Contacts Reference

### Internal Teams
- **Platform Team**: #platform-team
- **Infrastructure**: #infrastructure
- **Security**: #security-incidents
- **Engineering Management**: #eng-leadership

### External Vendors
- **Cloud Provider**: [Support phone/portal]
- **Monitoring Vendor**: [Support contact]
- **Network Provider**: [Emergency contact]

### Escalation Matrix
| Time | Action | Contact |
|------|--------|---------|
| 0-5 min | Initial response | On-call engineer |
| 5-15 min | Technical escalation | Platform team lead |
| 15-30 min | Management notification | Engineering manager |
| 30+ min | Executive escalation | CTO/VP Engineering |

Remember: **Safety first** - If in doubt, fail safe and escalate quickly.