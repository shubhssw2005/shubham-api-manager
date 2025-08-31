# Observability Guide

This guide covers the comprehensive observability stack implementation including Prometheus, Grafana, OpenTelemetry, and Jaeger for monitoring, metrics, and distributed tracing.

## Overview

The observability stack provides:

- **Metrics Collection**: Prometheus for collecting and storing metrics
- **Visualization**: Grafana for dashboards and alerting
- **Distributed Tracing**: Jaeger for request tracing across services
- **SLO Monitoring**: Custom dashboards and alerts for Service Level Objectives
- **Application Instrumentation**: OpenTelemetry for consistent telemetry data

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Application   │    │   Application   │
│   (API Service) │    │ (Media Service) │    │ (Worker Service)│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ /metrics             │ /metrics             │ /metrics
          │ traces               │ traces               │ traces
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
          ┌─────────▼─────────┐    ┌─────────▼─────────┐
          │    Prometheus     │    │      Jaeger       │
          │   (Metrics)       │    │    (Traces)       │
          └─────────┬─────────┘    └───────────────────┘
                    │
          ┌─────────▼─────────┐
          │     Grafana       │
          │  (Visualization)  │
          └───────────────────┘
```

## Quick Start

### 1. Deploy the Observability Stack

```bash
# For development environment
cd k8s/scripts
./deploy-observability.sh

# For production environment
ENVIRONMENT=production ./deploy-observability.sh
```

### 2. Access the Dashboards

**Development (via port-forward):**
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686

**Production:**
- Grafana: https://observability.yourdomain.com
- Jaeger: https://jaeger.yourdomain.com

### 3. Instrument Your Application

```javascript
import { initializeTracing } from '../lib/observability/tracing.js';
import metricsCollector from '../lib/observability/metrics.js';

// Initialize tracing
const tracing = initializeTracing('my-service', '1.0.0');
metricsCollector.initialize();

// Add middleware to Express app
app.use(tracing.createExpressMiddleware());
```

## Components

### Prometheus

Prometheus collects metrics from your applications and infrastructure components.

**Configuration:**
- Retention: 30 days (development), 90 days (production)
- Storage: 100GB (development), 500GB (production)
- Scrape interval: 30 seconds

**Key Metrics Collected:**
- HTTP request metrics (rate, duration, errors)
- Database query metrics
- Cache hit/miss rates
- Media processing metrics
- Business metrics (tenant usage, API token usage)

### Grafana

Grafana provides visualization and alerting capabilities.

**Pre-configured Dashboards:**
- API Service SLO Dashboard
- Media Service SLO Dashboard
- Infrastructure Overview
- Custom business metrics

**Data Sources:**
- Prometheus (metrics)
- Jaeger (traces)

### Jaeger

Jaeger provides distributed tracing capabilities.

**Configuration:**
- Development: All-in-one deployment with in-memory storage
- Production: Distributed deployment with Elasticsearch storage

**Trace Collection:**
- HTTP requests across services
- Database queries
- Cache operations
- Media processing workflows
- Custom business operations

### OpenTelemetry

OpenTelemetry provides consistent instrumentation across all services.

**Auto-instrumentation:**
- HTTP/HTTPS requests
- Express.js applications
- MongoDB operations
- Redis operations

**Custom instrumentation:**
- Business logic tracing
- Custom metrics collection
- Span attributes for context

## Service Level Objectives (SLOs)

### API Service SLOs

1. **Availability**: 99.99%
   - Measurement: Ratio of successful requests (non-5xx) to total requests
   - Error budget: 0.01% (52.6 minutes per year)

2. **Latency**: P99 < 200ms for GET requests
   - Measurement: 99th percentile of request duration
   - Applies to: All GET endpoints

3. **Error Rate**: < 0.1%
   - Measurement: Ratio of 5xx responses to total requests
   - Excludes: Client errors (4xx)

### Media Service SLOs

1. **Upload Success Rate**: 99.9%
   - Measurement: Successful uploads vs. total upload attempts
   - Error budget: 0.1%

2. **Processing Latency**: P95 < 30 seconds
   - Measurement: Time from upload to processing completion
   - Applies to: All media types

### Alerting Rules

**Critical Alerts:**
- API availability below 99.99%
- Error budget burn rate > 14.4x (exhausts budget in 2 hours)
- Database connection pool > 80%

**Warning Alerts:**
- API P99 latency > 200ms
- Media upload success rate < 99.9%
- Processing queue depth > 1000

## Custom Metrics

### HTTP Metrics

```javascript
// Automatically collected by middleware
http_requests_total{method, route, code, tenant_id}
http_request_duration_seconds{method, route, code, tenant_id}
```

### Database Metrics

```javascript
// Record database operations
metricsCollector.recordDatabaseQuery(
  'SELECT',      // operation
  'posts',       // table
  duration,      // duration in ms
  true,          // success
  tenantId       // tenant ID
);
```

### Cache Metrics

```javascript
// Record cache operations
metricsCollector.recordCacheHit('get', cacheKey, tenantId);
metricsCollector.recordCacheMiss('get', cacheKey, tenantId);
metricsCollector.recordCacheOperation('set', duration, true, tenantId);
```

### Media Processing Metrics

```javascript
// Record media operations
metricsCollector.recordMediaUpload('image', fileSize, true, tenantId);
metricsCollector.recordMediaProcessing('video', duration, true, tenantId);
```

### Business Metrics

```javascript
// Record business events
metricsCollector.recordTenantRequest(tenantId, endpoint, method);
metricsCollector.recordApiTokenUsage(tokenId, tenantId, endpoint);
metricsCollector.recordStorageUsage(tenantId, bytes, 'media');
```

## Distributed Tracing

### Automatic Tracing

The OpenTelemetry SDK automatically traces:
- HTTP requests and responses
- Database queries
- Cache operations
- External API calls

### Custom Tracing

```javascript
// Create custom spans
const span = tracing.createSpan('business_operation', {
  attributes: {
    'tenant.id': tenantId,
    'operation.type': 'data_processing',
  },
});

try {
  // Your business logic here
  const result = await processData();
  
  span.setAttributes({
    'result.count': result.length,
    'processing.success': true,
  });
  
  return result;
} catch (error) {
  span.recordException(error);
  throw error;
} finally {
  span.end();
}
```

### Trace Function Wrapper

```javascript
// Wrap functions with automatic tracing
const tracedFunction = tracing.traceFunction(
  'function_name',
  originalFunction,
  {
    attributes: { 'custom.attribute': 'value' },
    resultAttributes: (result) => ({ 'result.size': result.length }),
  }
);
```

## Dashboard Configuration

### SLO Dashboard Queries

**API Availability:**
```promql
(
  sum(rate(http_requests_total{job="api-service", code!~"5.."}[5m])) /
  sum(rate(http_requests_total{job="api-service"}[5m]))
) * 100
```

**API Latency P99:**
```promql
histogram_quantile(0.99, 
  sum(rate(http_request_duration_seconds_bucket{job="api-service"}[5m])) by (le)
) * 1000
```

**Error Budget Remaining:**
```promql
(
  1 - (
    (
      sum(increase(http_requests_total{job="api-service", code=~"5.."}[30d])) /
      sum(increase(http_requests_total{job="api-service"}[30d]))
    ) / 0.0001
  )
) * 100
```

### Custom Dashboard Creation

1. Access Grafana dashboard
2. Create new dashboard
3. Add panels with PromQL queries
4. Configure thresholds and alerts
5. Save and share with team

## Alerting Configuration

### Alertmanager Setup

Alerts are configured in Prometheus rules and routed through Alertmanager.

**Alert Routing:**
- Critical alerts: Immediate notification
- Warning alerts: 5-minute delay
- Info alerts: Daily digest

**Notification Channels:**
- Slack integration
- Email notifications
- PagerDuty for critical alerts

### Custom Alert Rules

```yaml
- alert: CustomBusinessMetricAlert
  expr: |
    rate(custom_business_metric_total[5m]) > 100
  for: 2m
  labels:
    severity: warning
    service: business-logic
  annotations:
    summary: "Custom business metric threshold exceeded"
    description: "Business metric rate is {{ $value }} per second"
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing in Prometheus**
   - Check ServiceMonitor configuration
   - Verify application `/metrics` endpoint
   - Check Prometheus targets page

2. **Traces not appearing in Jaeger**
   - Verify Jaeger collector endpoint
   - Check OpenTelemetry configuration
   - Ensure trace sampling is enabled

3. **Dashboard not loading data**
   - Check Grafana data source configuration
   - Verify PromQL queries
   - Check time range settings

### Debug Commands

```bash
# Check Prometheus targets
kubectl port-forward -n observability svc/observability-prometheus-server 9090:80
# Visit http://localhost:9090/targets

# Check ServiceMonitor status
kubectl get servicemonitor -n observability

# Check PrometheusRule status
kubectl get prometheusrule -n observability

# View application logs
kubectl logs -f deployment/api-service

# Check Jaeger traces
kubectl port-forward -n observability svc/observability-jaeger-query 16686:16686
# Visit http://localhost:16686
```

## Performance Considerations

### Metrics Cardinality

- Limit high-cardinality labels (user IDs, request IDs)
- Use tenant ID for multi-tenant metrics
- Aggregate metrics at appropriate levels

### Trace Sampling

- Development: 100% sampling
- Production: 1-10% sampling based on traffic
- Always sample error traces

### Storage Requirements

**Prometheus:**
- Development: ~1GB per day
- Production: ~10-50GB per day (depends on metrics cardinality)

**Jaeger:**
- Development: ~100MB per day
- Production: ~1-10GB per day (depends on trace volume and sampling)

## Security

### Authentication

- Grafana: Admin credentials stored in Kubernetes secrets
- Prometheus: Internal access only
- Jaeger: Internal access only (production)

### Network Security

- All components communicate within cluster network
- External access via Ingress with TLS termination
- Service mesh (Istio) for mTLS between services

### Data Retention

- Metrics: 30-90 days retention
- Traces: 7-30 days retention
- Logs: Centralized logging with appropriate retention policies

## Maintenance

### Regular Tasks

1. **Monitor storage usage**
   - Check Prometheus disk usage
   - Monitor Jaeger storage growth
   - Clean up old data as needed

2. **Update dashboards**
   - Review and update SLO thresholds
   - Add new business metrics
   - Optimize query performance

3. **Review alerts**
   - Analyze alert frequency
   - Adjust thresholds based on SLO performance
   - Update runbooks

### Backup and Recovery

- Prometheus data: Regular snapshots
- Grafana configuration: Export dashboards and data sources
- Alert rules: Version control in Git

## Integration with CI/CD

### Deployment Validation

```yaml
# Example GitHub Actions step
- name: Validate SLO Metrics
  run: |
    # Wait for deployment to stabilize
    sleep 60
    
    # Check if metrics are being collected
    curl -f "http://prometheus:9090/api/v1/query?query=up{job=\"api-service\"}"
    
    # Verify SLO compliance
    ./scripts/check-slo-compliance.sh
```

### Automated Alerting Tests

```bash
# Test alert rules
promtool test rules tests/alert-rules-test.yml

# Validate dashboard JSON
./scripts/validate-dashboards.sh
```

This observability stack provides comprehensive monitoring, alerting, and tracing capabilities to ensure your application meets its SLO requirements and provides excellent user experience.