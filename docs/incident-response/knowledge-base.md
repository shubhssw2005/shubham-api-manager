# Incident Response Knowledge Base

## Common Incident Patterns

### Database Connection Exhaustion
**Symptoms:** API timeouts, connection pool errors  
**Common Causes:** Traffic spikes, connection leaks, slow queries  
**Quick Fix:** Restart API services, scale read replicas  
**Runbook:** [Database Connection Issues](../runbooks/database-connection-issues.md)

### Cache Storm
**Symptoms:** High cache miss rate, database overload  
**Common Causes:** Cache invalidation, TTL expiration  
**Quick Fix:** Enable request coalescing, cache warming  
**Runbook:** [Cache Storm Response](../runbooks/cache-storm-response.md)

### S3 Hot Objects
**Symptoms:** High S3 request rates, increased latency  
**Common Causes:** Viral content, bot traffic  
**Quick Fix:** CloudFront caching, object replication  
**Runbook:** [Hot Object Mitigation](../runbooks/hot-object-mitigation.md)

## Troubleshooting Commands

### Kubernetes Diagnostics
```bash
# Check pod status
kubectl get pods -l app=api-service

# View recent events
kubectl get events --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods

# View logs
kubectl logs -l app=api-service --tail=100
```

### Database Diagnostics
```bash
# Check Aurora cluster status
aws rds describe-db-clusters --db-cluster-identifier production

# Monitor connections
kubectl exec -it deployment/api-service -- npm run db:connections

# Check slow queries
kubectl exec -it deployment/api-service -- npm run db:slow-queries
```

### Cache Diagnostics
```bash
# Redis statistics
kubectl exec -it redis-0 -- redis-cli info stats

# Cache hit rates
kubectl exec -it redis-0 -- redis-cli info keyspace

# Memory usage
kubectl exec -it redis-0 -- redis-cli info memory
```

## Escalation Matrix

| Severity | Initial Response | Escalation Time | Escalation Target |
|----------|------------------|-----------------|-------------------|
| Critical | On-call engineer | 15 minutes | Platform Team Lead |
| Warning | On-call engineer | 30 minutes | Platform Team Lead |
| Info | On-call engineer | 60 minutes | Next business day |

## Communication Templates

### Initial Incident Notification
```
🚨 INCIDENT ALERT 🚨
Severity: [CRITICAL/WARNING/INFO]
Service: [Service Name]
Impact: [Brief description]
Status: Investigating
ETA: [Estimated resolution time]
Incident ID: [INC-XXXXX]
```

### Status Update
```
📊 INCIDENT UPDATE 📊
Incident ID: [INC-XXXXX]
Status: [Investigating/Mitigating/Resolved]
Progress: [What has been done]
Next Steps: [What will be done next]
ETA: [Updated estimate]
```

### Resolution Notification
```
✅ INCIDENT RESOLVED ✅
Incident ID: [INC-XXXXX]
Duration: [X hours Y minutes]
Root Cause: [Brief explanation]
Resolution: [What was done to fix it]
PIR Scheduled: [Date/Time]
```