# Ultra C++ System Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying and operating the ultra-low-latency C++ system in Kubernetes with DPDK support.

## Prerequisites

### Cluster Requirements
- Kubernetes 1.24+
- Nodes with DPDK-capable network interfaces
- Hugepages support (2Mi pages)
- NUMA topology awareness
- Privileged container support

### Node Preparation
```bash
# Label nodes for DPDK workloads
kubectl label nodes <node-name> ultra-cpp.io/dpdk-enabled=true

# Taint nodes for dedicated workloads (optional)
kubectl taint nodes <node-name> ultra-cpp.io/dedicated=true:NoSchedule

# Verify hugepages configuration
kubectl describe node <node-name> | grep hugepages
```

### Required Resources
- **CPU**: 4-8 cores per pod (dedicated)
- **Memory**: 4-8Gi per pod
- **Hugepages**: 2-4Gi per pod
- **Network**: DPDK-compatible NICs
- **Storage**: Fast local storage for logs

## Deployment Steps

### 1. Create Namespace and Resources
```bash
# Apply namespace and resource quotas
kubectl apply -f k8s/ultra-cpp/namespace.yaml

# Verify namespace creation
kubectl get namespace ultra-cpp
kubectl get resourcequota -n ultra-cpp
```

### 2. Configure System Settings
```bash
# Apply configuration
kubectl apply -f k8s/ultra-cpp/configmap.yaml

# Verify configuration
kubectl get configmap ultra-cpp-config -n ultra-cpp -o yaml
```

### 3. Deploy Services
```bash
# Apply services
kubectl apply -f k8s/ultra-cpp/service.yaml

# Verify services
kubectl get svc -n ultra-cpp
```

### 4. Deploy Application (Blue-Green)
```bash
# Make deployment script executable
chmod +x k8s/ultra-cpp/scripts/deploy.sh

# Deploy with specific image tag
./k8s/ultra-cpp/scripts/deploy.sh --image-tag v1.0.0 --registry your-registry.com/ultra-cpp

# Monitor deployment
kubectl get pods -n ultra-cpp -w
```

### 5. Verify Deployment
```bash
# Check pod status
kubectl get pods -n ultra-cpp -l app=ultra-cpp-gateway

# Test health endpoints
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl http://localhost:8081/health/ready

# Check performance metrics
kubectl exec -n ultra-cpp deployment/ultra-cpp-gateway-blue -- curl http://localhost:9090/metrics
```

## Configuration Management

### Environment Variables
Key environment variables for the C++ application:

```yaml
env:
- name: DPDK_ENABLED
  value: "true"
- name: WORKER_THREADS
  value: "4"
- name: MEMORY_POOL_SIZE
  value: "2147483648"  # 2GB
- name: FALLBACK_UPSTREAM
  value: "http://nodejs-api:3005"
```

### ConfigMap Updates
```bash
# Update configuration
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch '{"data":{"ultra-cpp.conf":"[api_gateway]\nport = 8080\n..."}}'

# Restart pods to pick up changes
kubectl rollout restart deployment/ultra-cpp-gateway-blue -n ultra-cpp
```

## Monitoring and Observability

### Health Checks
The system provides multiple health check endpoints:

- `/health/live` - Liveness probe
- `/health/ready` - Readiness probe  
- `/health/startup` - Startup probe
- `/health` - Detailed health information
- `/metrics` - Performance metrics

### Prometheus Integration
```bash
# Port forward to access metrics
kubectl port-forward -n ultra-cpp svc/ultra-cpp-metrics 9090:9090

# Access metrics at http://localhost:9090/metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboards
Import the provided Grafana dashboard for comprehensive monitoring:
- Latency metrics (P50, P95, P99)
- Throughput and error rates
- Resource utilization
- DPDK port status

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale deployment
kubectl scale deployment ultra-cpp-gateway-blue -n ultra-cpp --replicas=5

# Auto-scaling with HPA
kubectl autoscale deployment ultra-cpp-gateway-blue -n ultra-cpp --cpu-percent=70 --min=3 --max=10
```

### Vertical Scaling
```bash
# Increase resource limits
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp -p '{"spec":{"template":{"spec":{"containers":[{"name":"ultra-cpp-gateway","resources":{"limits":{"cpu":"8","memory":"16Gi"}}}]}}}}'
```

### Performance Tuning
```bash
# CPU affinity optimization
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp --patch '{"spec":{"template":{"spec":{"containers":[{"name":"ultra-cpp-gateway","env":[{"name":"CPU_AFFINITY","value":"0-7"}]}]}}}}'

# NUMA optimization
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp --patch '{"spec":{"template":{"spec":{"containers":[{"name":"ultra-cpp-gateway","env":[{"name":"NUMA_NODE","value":"0"}]}]}}}}'
```

## Blue-Green Deployment Process

### Automated Deployment
```bash
# Deploy new version
./k8s/ultra-cpp/scripts/deploy.sh --image-tag v1.1.0

# The script will:
# 1. Create new deployment (green if blue is active)
# 2. Wait for health checks to pass
# 3. Switch traffic to new deployment
# 4. Clean up old deployment
```

### Manual Deployment Control
```bash
# Check current deployment status
./k8s/ultra-cpp/scripts/rollback.sh status

# Manual rollback if needed
./k8s/ultra-cpp/scripts/rollback.sh emergency

# Rollback to specific deployment
./k8s/ultra-cpp/scripts/rollback.sh to ultra-cpp-gateway-blue
```

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod events
kubectl describe pod <pod-name> -n ultra-cpp

# Check logs
kubectl logs <pod-name> -n ultra-cpp

# Common causes:
# - Insufficient hugepages
# - DPDK port binding issues
# - Resource constraints
```

#### DPDK Issues
```bash
# Check hugepages allocation
kubectl exec -n ultra-cpp <pod-name> -- cat /proc/meminfo | grep -i huge

# Check DPDK device binding
kubectl exec -n ultra-cpp <pod-name> -- ls -la /dev/uio*

# Verify network interfaces
kubectl exec -n ultra-cpp <pod-name> -- ip link show
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n ultra-cpp

# Check application metrics
kubectl exec -n ultra-cpp <pod-name> -- curl http://localhost:9090/metrics

# Review performance runbook
cat k8s/ultra-cpp/runbooks/performance-troubleshooting.md
```

### Log Analysis
```bash
# Application logs
kubectl logs -n ultra-cpp -l app=ultra-cpp-gateway --tail=100 -f

# System events
kubectl get events -n ultra-cpp --sort-by='.lastTimestamp'

# Performance logs
kubectl exec -n ultra-cpp <pod-name> -- tail -f /var/log/ultra-cpp/performance.log
```

## Security Considerations

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ultra-cpp-network-policy
  namespace: ultra-cpp
spec:
  podSelector:
    matchLabels:
      app: ultra-cpp-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

### Security Context
The pods run with privileged access for DPDK:
```yaml
securityContext:
  privileged: true
  capabilities:
    add:
    - IPC_LOCK
    - SYS_NICE
    - NET_ADMIN
    - NET_RAW
```

### Secrets Management
```bash
# Create JWT secret
kubectl create secret generic ultra-cpp-jwt-secret \
  --from-literal=jwt-secret="your-jwt-secret" \
  -n ultra-cpp

# Mount secret in deployment
kubectl patch deployment ultra-cpp-gateway-blue -n ultra-cpp --patch '{"spec":{"template":{"spec":{"volumes":[{"name":"jwt-secret","secret":{"secretName":"ultra-cpp-jwt-secret"}}]}}}}'
```

## Backup and Recovery

### Configuration Backup
```bash
# Backup all configurations
kubectl get all,configmaps,secrets -n ultra-cpp -o yaml > ultra-cpp-backup-$(date +%Y%m%d).yaml

# Restore from backup
kubectl apply -f ultra-cpp-backup-YYYYMMDD.yaml
```

### Disaster Recovery
```bash
# Emergency procedures
cat k8s/ultra-cpp/runbooks/emergency-procedures.md

# Complete system recovery
kubectl delete namespace ultra-cpp
kubectl apply -f k8s/ultra-cpp/namespace.yaml
kubectl apply -f k8s/ultra-cpp/
./k8s/ultra-cpp/scripts/deploy.sh --image-tag latest
```

## Maintenance

### Regular Tasks
- Monitor system health and performance
- Review and rotate logs
- Update configurations as needed
- Test backup and recovery procedures

### Updates and Patches
```bash
# Update system configuration
kubectl patch configmap ultra-cpp-config -n ultra-cpp --patch-file config-update.yaml

# Deploy security patches
./k8s/ultra-cpp/scripts/deploy.sh --image-tag v1.0.1-security

# Update Kubernetes resources
kubectl apply -f k8s/ultra-cpp/
```

### Capacity Planning
- Monitor resource usage trends
- Plan for traffic growth
- Evaluate hardware requirements
- Test scaling procedures

## Support and Documentation

### Runbooks
- [System Health Monitoring](./runbooks/system-health-monitoring.md)
- [Performance Troubleshooting](./runbooks/performance-troubleshooting.md)
- [Emergency Procedures](./runbooks/emergency-procedures.md)

### Monitoring Dashboards
- Grafana: Ultra C++ System Overview
- Prometheus: Metrics and Alerting
- Kubernetes Dashboard: Resource Management

### Contact Information
- **Platform Team**: #platform-team
- **On-Call**: PagerDuty rotation
- **Documentation**: [Internal wiki link]
- **Support**: [Ticket system link]