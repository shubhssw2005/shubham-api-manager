# Operational Runbooks

This directory contains step-by-step runbooks for common operational scenarios in our AWS deployment system.

## Runbook Categories

### 1. High Availability & Disaster Recovery
- [Database Failover](./database-failover.md)
- [Regional Failover](./regional-failover.md)
- [EKS Cluster Recovery](./eks-cluster-recovery.md)

### 2. Performance & Scaling
- [Hot Object Mitigation](./hot-object-mitigation.md)
- [Cache Storm Response](./cache-storm-response.md)
- [Auto-scaling Issues](./autoscaling-issues.md)

### 3. Security Incidents
- [Security Breach Response](./security-breach-response.md)
- [DDoS Attack Mitigation](./ddos-mitigation.md)
- [Certificate Expiry](./certificate-expiry.md)

### 4. Application Issues
- [API Service Degradation](./api-service-degradation.md)
- [Media Processing Failures](./media-processing-failures.md)
- [Database Connection Issues](./database-connection-issues.md)

### 5. Infrastructure Issues
- [S3 Service Degradation](./s3-service-degradation.md)
- [Network Connectivity Issues](./network-connectivity-issues.md)
- [Load Balancer Issues](./load-balancer-issues.md)

## Runbook Format

Each runbook follows a standardized format:

1. **Incident Description** - What the issue looks like
2. **Impact Assessment** - How to assess severity
3. **Immediate Actions** - First steps to take
4. **Investigation Steps** - How to diagnose the root cause
5. **Resolution Steps** - How to fix the issue
6. **Prevention** - How to prevent recurrence
7. **Escalation** - When and how to escalate

## Emergency Contacts

- **On-Call Engineer**: PagerDuty rotation
- **Platform Team Lead**: [Contact Info]
- **Security Team**: [Contact Info]
- **AWS Support**: [Support Case Process]

## Tools and Access

- [AWS Console Access](https://console.aws.amazon.com)
- [Kubernetes Dashboard](https://k8s-dashboard.example.com)
- [Grafana Dashboards](https://grafana.example.com)
- [PagerDuty Console](https://example.pagerduty.com)
- [Incident Response Slack Channel](https://example.slack.com/channels/incident-response)