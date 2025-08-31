# Incident Response System

## Overview

The Incident Response System provides automated incident management, alert processing, and post-incident review capabilities for our AWS deployment system. It integrates with PagerDuty, Slack, and monitoring systems to provide comprehensive incident lifecycle management.

## Components

### 1. IncidentManager
- Creates and manages incident lifecycle
- Handles PagerDuty integration
- Manages escalation rules
- Executes automated response actions

### 2. AlertProcessor
- Processes incoming alerts from monitoring systems
- Applies alert rules and suppression logic
- Creates incidents from alerts automatically
- Maintains alert history and statistics

### 3. PIRManager
- Generates Post-Incident Review documents
- Updates knowledge base with lessons learned
- Provides PIR templates and automation
- Tracks incident patterns and trends

### 4. IncidentResponseService
- Main API service for incident management
- Webhook endpoints for alert integration
- REST API for incident operations
- Statistics and reporting endpoints

## Quick Start

### 1. Setup
```bash
# Run the setup script
node scripts/incident-response-setup.js

# Set environment variables
export PAGERDUTY_API_KEY="your-api-key"
export PAGERDUTY_SERVICE_KEY="your-service-key"
export SLACK_INCIDENT_WEBHOOK="your-slack-webhook"
```

### 2. Deploy to Kubernetes
```bash
# Create secrets
kubectl create secret generic incident-response-secrets \
  --from-literal=pagerduty-api-key="$PAGERDUTY_API_KEY" \
  --from-literal=slack-webhook-url="$SLACK_INCIDENT_WEBHOOK"

# Deploy the service
kubectl apply -f k8s/incident-response/
```

### 3. Start Locally (for development)
```bash
npm install
npm run incident-response:start
```

## API Endpoints

### Incident Management
- `POST /incidents` - Create incident
- `GET /incidents` - List incidents
- `GET /incidents/:id` - Get incident details
- `PATCH /incidents/:id` - Update incident
- `POST /incidents/:id/resolve` - Resolve incident

### Alert Webhooks
- `POST /webhooks/alerts/prometheus` - Prometheus alerts
- `POST /webhooks/alerts/cloudwatch` - CloudWatch alarms
- `POST /webhooks/alerts/pagerduty` - PagerDuty webhooks

### Post-Incident Reviews
- `POST /incidents/:id/pir` - Create PIR
- `GET /pirs` - List PIRs
- `GET /pirs/summary` - PIR summary statistics

### Statistics
- `GET /stats/alerts` - Alert statistics
- `GET /stats/incidents` - Incident statistics

## Configuration

### Environment Variables
```bash
# Required
PAGERDUTY_API_KEY=your-pagerduty-api-key
PAGERDUTY_SERVICE_KEY=your-service-key

# Optional
SLACK_INCIDENT_WEBHOOK=your-slack-webhook-url
PAGERDUTY_DATABASE_ESCALATION_KEY=database-escalation-key
PAGERDUTY_PLATFORM_ESCALATION_KEY=platform-escalation-key
PAGERDUTY_SECURITY_ESCALATION_KEY=security-escalation-key
```

### Alert Rules
Configure alert processing rules in `config/incident-response/alert-rules.json`:

```json
{
  "rules": [
    {
      "name": "database-down",
      "alertName": "DatabaseDown",
      "severity": "critical",
      "service": "database",
      "runbook": "https://runbooks.example.com/database-failover"
    }
  ]
}
```

## Integration Examples

### Prometheus AlertManager
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'incident-response'

receivers:
- name: 'incident-response'
  webhook_configs:
  - url: 'http://incident-response-service/webhooks/alerts/prometheus'
```

### CloudWatch Alarms
```bash
# Create SNS topic for incident response
aws sns create-topic --name incident-response-alerts

# Subscribe incident response service
aws sns subscribe \
  --topic-arn arn:aws:sns:region:account:incident-response-alerts \
  --protocol https \
  --notification-endpoint https://your-domain.com/webhooks/alerts/cloudwatch
```

### Manual Incident Creation
```bash
# Create incident via API
curl -X POST http://localhost:3001/incidents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database Connection Issues",
    "description": "High connection count detected",
    "severity": "warning",
    "service": "database"
  }'
```

## Runbooks

The system includes comprehensive runbooks for common scenarios:

- [Database Failover](../runbooks/database-failover.md)
- [Hot Object Mitigation](../runbooks/hot-object-mitigation.md)
- [Cache Storm Response](../runbooks/cache-storm-response.md)
- [Security Breach Response](../runbooks/security-breach-response.md)

## Monitoring

### Health Checks
```bash
# Service health
curl http://localhost:3001/health

# Alert statistics
curl http://localhost:3001/stats/alerts

# Incident statistics
curl http://localhost:3001/stats/incidents
```

### Metrics
The service exposes Prometheus metrics:
- `incidents_created_total` - Total incidents created
- `incidents_resolved_total` - Total incidents resolved
- `alerts_processed_total` - Total alerts processed
- `incident_resolution_duration_seconds` - Incident resolution time

## Testing

### Unit Tests
```bash
npm test tests/incident-response/
```

### Integration Tests
```bash
# Test alert processing
curl -X POST http://localhost:3001/webhooks/alerts/prometheus \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/prometheus-alert.json

# Test emergency response
./scripts/incident-response/emergency-response.sh database critical "Test incident"
```

## Troubleshooting

### Common Issues

1. **PagerDuty Integration Not Working**
   - Check API key and service key configuration
   - Verify network connectivity to PagerDuty API
   - Check logs for authentication errors

2. **Alerts Not Creating Incidents**
   - Verify alert rules configuration
   - Check alert format matches expected schema
   - Review suppression rules

3. **Slack Notifications Not Sent**
   - Verify webhook URL configuration
   - Check Slack webhook permissions
   - Review network connectivity

### Logs
```bash
# View service logs
kubectl logs -l app=incident-response-service

# View specific incident logs
kubectl logs -l app=incident-response-service | grep "INC-"
```

## Security Considerations

- Store sensitive credentials in Kubernetes secrets
- Use HTTPS for all webhook endpoints
- Implement proper authentication for API endpoints
- Regularly rotate PagerDuty API keys
- Monitor for unauthorized access attempts

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Test with real alerts before deploying
5. Follow security best practices