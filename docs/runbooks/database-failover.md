# Database Failover Runbook

## Incident Description
Aurora Postgres primary database is unresponsive or experiencing high latency (>5 seconds for queries).

## Impact Assessment
- **Severity 1**: Complete database unavailability (all writes failing)
- **Severity 2**: High latency affecting user experience (>2s response times)
- **Severity 3**: Single AZ failure with automatic failover in progress

## Immediate Actions (First 5 minutes)

### 1. Verify the Issue
```bash
# Check database connectivity
kubectl exec -it deployment/api-service -- npm run db:health-check

# Check Aurora cluster status
aws rds describe-db-clusters --db-cluster-identifier production-aurora-cluster

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBClusterIdentifier,Value=production-aurora-cluster \
  --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum
```

### 2. Check Application Health
```bash
# Check API service health
kubectl get pods -l app=api-service
kubectl logs -l app=api-service --tail=100

# Check connection pool status
kubectl exec -it deployment/api-service -- curl localhost:3000/health/database
```

### 3. Notify Stakeholders
```bash
# Send initial notification
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "DATABASE_INCIDENT_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Database failover in progress",
      "severity": "critical",
      "source": "aurora-cluster"
    }
  }'
```

## Investigation Steps

### 1. Check Aurora Cluster Status
```bash
# Get detailed cluster information
aws rds describe-db-clusters --db-cluster-identifier production-aurora-cluster \
  --query 'DBClusters[0].{Status:Status,Engine:Engine,MultiAZ:MultiAZ,BackupRetentionPeriod:BackupRetentionPeriod}'

# Check cluster members
aws rds describe-db-cluster-members --db-cluster-identifier production-aurora-cluster
```

### 2. Review Recent Changes
```bash
# Check recent deployments
kubectl get events --sort-by='.lastTimestamp' | head -20

# Check recent configuration changes
git log --oneline --since="2 hours ago" terraform/
```

### 3. Analyze Metrics
- Check Grafana dashboard: "Aurora Database Performance"
- Review CloudWatch logs for error patterns
- Check connection pool metrics in application logs

## Resolution Steps

### Option 1: Manual Failover (Recommended)
```bash
# Initiate manual failover to read replica
aws rds failover-db-cluster \
  --db-cluster-identifier production-aurora-cluster \
  --target-db-instance-identifier production-aurora-replica-1

# Monitor failover progress
watch -n 5 'aws rds describe-db-clusters --db-cluster-identifier production-aurora-cluster --query "DBClusters[0].Status"'
```

### Option 2: Scale Up Instance Class
```bash
# If performance issue, scale up temporarily
aws rds modify-db-instance \
  --db-instance-identifier production-aurora-primary \
  --db-instance-class db.r5.2xlarge \
  --apply-immediately
```

### Option 3: Connection Pool Adjustment
```bash
# Restart API services to reset connection pools
kubectl rollout restart deployment/api-service
kubectl rollout status deployment/api-service

# Update connection pool configuration if needed
kubectl patch configmap api-config -p '{"data":{"DB_POOL_SIZE":"20"}}'
```

## Verification Steps

### 1. Confirm Database Connectivity
```bash
# Test database connection
kubectl exec -it deployment/api-service -- npm run db:test-connection

# Verify read/write operations
kubectl exec -it deployment/api-service -- npm run db:test-operations
```

### 2. Check Application Performance
```bash
# Monitor API response times
curl -w "@curl-format.txt" -s -o /dev/null https://api.example.com/health

# Check error rates in logs
kubectl logs -l app=api-service --since=5m | grep -i error | wc -l
```

### 3. Validate Monitoring
- Confirm all monitoring alerts have cleared
- Verify Grafana dashboards show normal metrics
- Check that PagerDuty incident can be resolved

## Prevention Measures

### 1. Implement Connection Pooling Best Practices
```javascript
// Update database configuration
const poolConfig = {
  max: 20,
  min: 5,
  acquire: 30000,
  idle: 10000,
  evict: 1000,
  handleDisconnects: true
};
```

### 2. Set Up Proactive Monitoring
```yaml
# Add CloudWatch alarm for connection count
ConnectionCountAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: Aurora-High-Connection-Count
    MetricName: DatabaseConnections
    Namespace: AWS/RDS
    Statistic: Average
    Period: 300
    EvaluationPeriods: 2
    Threshold: 80
    ComparisonOperator: GreaterThanThreshold
```

### 3. Regular Failover Testing
```bash
# Schedule monthly failover tests
# Add to cron: 0 2 1 * * /scripts/test-failover.sh
```

## Escalation Criteria

**Escalate to Platform Team Lead if:**
- Failover takes longer than 10 minutes
- Multiple failover attempts fail
- Data corruption is suspected

**Escalate to AWS Support if:**
- Aurora service appears to have regional issues
- Failover mechanisms are not working as expected
- Need assistance with point-in-time recovery

## Post-Incident Actions

1. Update incident timeline in PagerDuty
2. Schedule post-incident review within 24 hours
3. Document any configuration changes made
4. Update runbook based on lessons learned
5. Review and update monitoring thresholds if needed

## Related Runbooks
- [Regional Failover](./regional-failover.md)
- [Database Connection Issues](./database-connection-issues.md)
- [API Service Degradation](./api-service-degradation.md)