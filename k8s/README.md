# Kubernetes Deployment Manifests

This directory contains Helm charts and Kubernetes manifests for deploying the API platform with advanced media management capabilities on AWS EKS with Istio service mesh.

## Architecture Overview

The deployment consists of three main services:

1. **API Service** - Node.js/Express API server
2. **Media Service** - Go-based media processing service
3. **Worker Service** - Background processing workers

All services are deployed with:
- HorizontalPodAutoscaler (HPA) for automatic scaling
- PodDisruptionBudgets (PDB) for high availability
- Resource limits and requests
- Istio service mesh for mTLS and traffic management
- Prometheus metrics collection

## Prerequisites

- Kubernetes cluster (EKS recommended)
- Helm 3.x
- kubectl
- istioctl
- AWS CLI configured

## Directory Structure

```
k8s/
├── helm/
│   ├── api-service/          # API service Helm chart
│   ├── media-service/        # Media service Helm chart
│   ├── worker-service/       # Worker service Helm chart
│   └── common/               # Common Helm templates
├── istio/                    # Istio service mesh configuration
├── scripts/                  # Deployment scripts
└── README.md                 # This file
```

## Quick Start

1. **Deploy all services:**
   ```bash
   ./k8s/scripts/deploy.sh
   ```

2. **Deploy specific environment:**
   ```bash
   ENVIRONMENT=staging ./k8s/scripts/deploy.sh
   ```

## Manual Deployment

### 1. Install Istio Service Mesh

```bash
# Install Istio
istioctl install --set values.defaultRevision=default -y

# Apply Istio configurations
kubectl apply -f k8s/istio/
```

### 2. Create Namespace

```bash
kubectl create namespace production
kubectl label namespace production istio-injection=enabled
```

### 3. Deploy Services

```bash
# Deploy API service
helm upgrade --install api-service k8s/helm/api-service/ \
  --namespace production \
  --values k8s/helm/api-service/values-production.yaml

# Deploy Media service
helm upgrade --install media-service k8s/helm/media-service/ \
  --namespace production \
  --values k8s/helm/media-service/values-production.yaml

# Deploy Worker service
helm upgrade --install worker-service k8s/helm/worker-service/ \
  --namespace production \
  --values k8s/helm/worker-service/values-production.yaml
```

## Configuration

### Environment-Specific Values

Each service has environment-specific values files:
- `values-development.yaml`
- `values-staging.yaml`
- `values-production.yaml`

### Key Configuration Options

#### API Service
- **Replicas**: 3-100 (auto-scaling)
- **Resources**: 500m CPU, 512Mi memory (requests)
- **HPA**: CPU 70%, Memory 80%
- **Istio**: mTLS enabled, circuit breaker configured

#### Media Service
- **Replicas**: 2-50 (auto-scaling)
- **Resources**: 1000m CPU, 2Gi memory (requests)
- **Node Selector**: media-processing workload
- **Tolerations**: media-processing taint

#### Worker Service
- **Media Processor**: 2-100 replicas, SQS-based scaling
- **Outbox Processor**: 2-20 replicas
- **Data Export**: 1-10 replicas

## Istio Service Mesh

### Features Enabled

1. **mTLS**: Strict mutual TLS between all services
2. **Traffic Management**: Load balancing, circuit breakers, retries
3. **Security**: Authorization policies, peer authentication
4. **Observability**: Distributed tracing, metrics collection

### Gateway Configuration

- **API Gateway**: `api.company.com`
- **Media Gateway**: `media.company.com`
- **TLS Termination**: Automatic HTTPS redirect

### Traffic Policies

- **Circuit Breaker**: 5 consecutive errors trigger circuit breaker
- **Retries**: 3 attempts with exponential backoff
- **Timeouts**: 30s for API, 300s for media processing

## Monitoring and Observability

### Metrics Collection

All services expose Prometheus metrics at `/metrics`:
- Request latency (P50, P95, P99)
- Request rate and error rate
- Custom business metrics

### Health Checks

- **Liveness Probe**: `/health` endpoint
- **Readiness Probe**: `/health/ready` endpoint
- **Startup Probe**: Configured for slow-starting services

### Service Monitors

Prometheus ServiceMonitor resources are automatically created for:
- API service metrics
- Media service metrics
- Worker service metrics

## Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

All services use HPA v2 with multiple metrics:
- CPU utilization
- Memory utilization
- Custom metrics (SQS queue depth for workers)

### Scaling Behavior

- **Scale Up**: Aggressive (50% increase, max 10 pods per minute)
- **Scale Down**: Conservative (10% decrease, 5-minute stabilization)

### Vertical Pod Autoscaler (VPA)

VPA can be enabled for automatic resource recommendation:
```bash
kubectl apply -f k8s/vpa/
```

## High Availability

### Pod Disruption Budgets

- **API Service**: Minimum 2 pods available
- **Media Service**: Minimum 1 pod available
- **Workers**: Minimum 1 pod per worker type

### Anti-Affinity Rules

Services use pod anti-affinity to spread across nodes:
- Preferred scheduling on different nodes
- Topology spread constraints for even distribution

### Multi-AZ Deployment

Services are deployed across multiple availability zones:
- Node selectors for AZ distribution
- Persistent volume affinity for stateful workloads

## Security

### Pod Security Context

- **Non-root user**: UID 1000
- **Read-only root filesystem**
- **Dropped capabilities**: ALL
- **No privilege escalation**

### Network Policies

Istio authorization policies control traffic:
- Ingress traffic only from gateways
- Inter-service communication via mTLS
- Deny-by-default security model

### Secrets Management

Sensitive data stored in Kubernetes secrets:
- Database credentials
- JWT signing keys
- AWS access keys (via IRSA recommended)

## Troubleshooting

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod <pod-name> -n production
   kubectl logs <pod-name> -n production
   ```

2. **Istio Configuration Issues**
   ```bash
   istioctl proxy-status -n production
   istioctl proxy-config cluster <pod-name> -n production
   ```

3. **HPA Not Scaling**
   ```bash
   kubectl describe hpa <hpa-name> -n production
   kubectl top pods -n production
   ```

### Debug Commands

```bash
# Check service mesh status
istioctl analyze -n production

# View traffic policies
kubectl get destinationrules -n production -o yaml

# Check metrics server
kubectl top nodes
kubectl top pods -n production

# View service endpoints
kubectl get endpoints -n production
```

## Maintenance

### Rolling Updates

Services support zero-downtime rolling updates:
```bash
helm upgrade api-service k8s/helm/api-service/ \
  --namespace production \
  --set image.tag=v1.1.0
```

### Backup and Recovery

- **Configuration**: Helm values and Kubernetes manifests in Git
- **Persistent Data**: Database and Redis backups
- **Disaster Recovery**: Multi-region deployment capability

### Monitoring Deployment Health

```bash
# Check deployment status
kubectl rollout status deployment/api-service -n production

# View recent events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n production
```

## Performance Tuning

### Resource Optimization

1. **CPU Requests**: Set based on actual usage patterns
2. **Memory Limits**: Prevent OOM kills while allowing bursts
3. **JVM Tuning**: For Java-based services (if applicable)

### Network Optimization

1. **Connection Pooling**: Configured in Istio DestinationRules
2. **Keep-Alive**: HTTP/1.1 and HTTP/2 optimizations
3. **Compression**: Enabled at ingress gateway

### Storage Optimization

1. **Ephemeral Storage**: Configured for temporary processing
2. **Volume Mounts**: Optimized for container security
3. **Storage Classes**: Use appropriate storage types

## Contributing

When modifying the Kubernetes manifests:

1. Test changes in development environment first
2. Update values files for all environments
3. Validate Helm templates: `helm template <chart> --debug`
4. Check Istio configuration: `istioctl analyze`
5. Update documentation as needed

## Support

For issues with the Kubernetes deployment:
1. Check the troubleshooting section above
2. Review logs and events
3. Contact the Platform Team
4. Create an issue in the project repository