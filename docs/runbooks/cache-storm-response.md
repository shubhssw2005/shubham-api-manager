# Cache Storm Response Runbook

## Incident Description
A cache storm occurs when a large number of requests simultaneously attempt to regenerate the same cached content, typically after cache expiration or invalidation. This can cause:
- Database overload from simultaneous cache misses
- API service degradation
- Cascading failures across dependent services

## Impact Assessment
- **Severity 1**: Database connections exhausted, API completely unavailable
- **Severity 2**: Significant API latency increase (>5s), partial service degradation
- **Severity 3**: Elevated cache miss rates detected, performance impact minimal

## Immediate Actions (First 5 minutes)

### 1. Identify Cache Storm Pattern
```bash
# Check Redis cache hit rates
kubectl exec -it redis-cluster-0 -- redis-cli info stats | grep keyspace

# Monitor API service metrics
kubectl top pods -l app=api-service

# Check database connection count
kubectl exec -it deployment/api-service -- npm run db:connection-count
```

### 2. Check Application Logs
```bash
# Look for cache miss patterns
kubectl logs -l app=api-service --since=10m | grep -i "cache miss" | wc -l

# Check for database connection errors
kubectl logs -l app=api-service --since=10m | grep -i "connection"

# Monitor error rates
kubectl logs -l app=api-service --since=5m | grep -E "(error|ERROR)" | tail -20
```

### 3. Immediate Notification
```bash
# Alert the team
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "CACHE_STORM_INCIDENT_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Cache storm detected - high cache miss rate",
      "severity": "warning",
      "source": "cache-monitoring"
    }
  }'
```

## Investigation Steps

### 1. Analyze Cache Metrics
```bash
# Get detailed Redis statistics
kubectl exec -it redis-cluster-0 -- redis-cli --latency-history -i 1

# Check cache key patterns
kubectl exec -it redis-cluster-0 -- redis-cli --scan --pattern "*" | head -20

# Monitor cache memory usage
kubectl exec -it redis-cluster-0 -- redis-cli info memory
```

### 2. Database Impact Assessment
```bash
# Check database performance metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBClusterIdentifier,Value=production-aurora-cluster \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum

# Check for slow queries
kubectl exec -it deployment/api-service -- npm run db:slow-queries
```

### 3. Identify Storm Trigger
```bash
# Check recent deployments
kubectl get events --sort-by='.lastTimestamp' | head -10

# Look for cache invalidation events
kubectl logs -l app=api-service --since=30m | grep -i "cache.*invalid"

# Check for scheduled tasks that might trigger cache refresh
kubectl get cronjobs
```

## Resolution Steps

### Option 1: Implement Request Coalescing (Immediate)
```bash
# Deploy emergency cache coalescing configuration
kubectl patch configmap api-config -p '{
  "data": {
    "CACHE_COALESCING_ENABLED": "true",
    "CACHE_COALESCING_TIMEOUT": "5000",
    "MAX_CONCURRENT_CACHE_BUILDS": "3"
  }
}'

# Restart API services to apply configuration
kubectl rollout restart deployment/api-service
kubectl rollout status deployment/api-service --timeout=300s
```

### Option 2: Enable Cache Warming
```bash
# Trigger cache warming for critical keys
kubectl create job cache-warmer-$(date +%s) --from=cronjob/cache-warmer

# Monitor cache warming progress
kubectl logs job/cache-warmer-$(date +%s) -f
```

### Option 3: Implement Circuit Breaker
```bash
# Enable circuit breaker for database calls
kubectl patch configmap api-config -p '{
  "data": {
    "CIRCUIT_BREAKER_ENABLED": "true",
    "CIRCUIT_BREAKER_THRESHOLD": "10",
    "CIRCUIT_BREAKER_TIMEOUT": "30000"
  }
}'

# Apply the configuration
kubectl rollout restart deployment/api-service
```

### Option 4: Scale Resources Temporarily
```bash
# Scale up API service replicas
kubectl scale deployment api-service --replicas=10

# Scale up database read replicas if needed
aws rds create-db-instance \
  --db-instance-identifier temp-read-replica-$(date +%s) \
  --db-instance-class db.r5.large \
  --source-db-identifier production-aurora-cluster

# Increase Redis memory if needed
kubectl patch statefulset redis-cluster -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "redis",
          "resources": {
            "requests": {"memory": "4Gi"},
            "limits": {"memory": "4Gi"}
          }
        }]
      }
    }
  }
}'
```

## Advanced Mitigation Strategies

### 1. Implement Smart Cache Refresh
```javascript
// Add to cache service
class SmartCacheManager {
  constructor() {
    this.refreshQueue = new Map();
    this.maxConcurrentRefresh = 3;
  }

  async get(key, refreshFunction) {
    // Try cache first
    let value = await this.redis.get(key);
    
    if (value) {
      return JSON.parse(value);
    }

    // Check if refresh is already in progress
    if (this.refreshQueue.has(key)) {
      // Wait for existing refresh to complete
      return await this.refreshQueue.get(key);
    }

    // Limit concurrent refreshes
    if (this.refreshQueue.size >= this.maxConcurrentRefresh) {
      // Return stale data or default value
      const staleValue = await this.redis.get(`stale:${key}`);
      if (staleValue) {
        return JSON.parse(staleValue);
      }
      throw new Error('Cache storm protection: too many concurrent refreshes');
    }

    // Start refresh process
    const refreshPromise = this.refreshWithLock(key, refreshFunction);
    this.refreshQueue.set(key, refreshPromise);

    try {
      const result = await refreshPromise;
      return result;
    } finally {
      this.refreshQueue.delete(key);
    }
  }

  async refreshWithLock(key, refreshFunction) {
    const lockKey = `lock:${key}`;
    const lockValue = Date.now().toString();
    
    // Try to acquire lock
    const acquired = await this.redis.set(lockKey, lockValue, 'PX', 30000, 'NX');
    
    if (!acquired) {
      // Another process is refreshing, wait and retry
      await new Promise(resolve => setTimeout(resolve, 100));
      return await this.get(key, refreshFunction);
    }

    try {
      const value = await refreshFunction();
      
      // Store both fresh and stale copies
      await Promise.all([
        this.redis.setex(key, 300, JSON.stringify(value)),
        this.redis.setex(`stale:${key}`, 3600, JSON.stringify(value))
      ]);
      
      return value;
    } finally {
      // Release lock
      await this.redis.del(lockKey);
    }
  }
}
```

### 2. Implement Probabilistic Cache Refresh
```javascript
// Probabilistic early refresh to prevent thundering herd
class ProbabilisticCache {
  async get(key, refreshFunction, ttl = 300) {
    const cached = await this.redis.get(key);
    
    if (cached) {
      const data = JSON.parse(cached);
      const age = Date.now() - data.timestamp;
      const refreshProbability = age / (ttl * 1000);
      
      // Probabilistically refresh before expiration
      if (Math.random() < refreshProbability) {
        // Refresh in background
        this.refreshInBackground(key, refreshFunction, ttl);
      }
      
      return data.value;
    }
    
    // Cache miss - refresh synchronously
    return await this.refreshSynchronously(key, refreshFunction, ttl);
  }
}
```

## Verification Steps

### 1. Monitor Cache Performance
```bash
# Check cache hit rates
kubectl exec -it redis-cluster-0 -- redis-cli info stats | grep -E "(hits|misses)"

# Monitor API response times
curl -w "@curl-format.txt" -s -o /dev/null https://api.example.com/health

# Check database connection count
kubectl exec -it deployment/api-service -- npm run db:connection-count
```

### 2. Verify Request Coalescing
```bash
# Check application logs for coalescing activity
kubectl logs -l app=api-service --since=5m | grep -i "coalescing"

# Monitor concurrent cache builds
kubectl exec -it deployment/api-service -- curl localhost:3000/metrics | grep cache_builds_active
```

### 3. Validate System Stability
```bash
# Check error rates
kubectl logs -l app=api-service --since=10m | grep -E "(error|ERROR)" | wc -l

# Monitor resource usage
kubectl top pods -l app=api-service

# Verify all services are healthy
kubectl get pods -l app=api-service -o wide
```

## Prevention Measures

### 1. Implement Cache Warming Strategy
```yaml
# Scheduled cache warming job
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cache-warmer
spec:
  schedule: "*/30 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cache-warmer
            image: api-service:latest
            command: ["npm", "run", "cache:warm"]
            env:
            - name: CACHE_WARM_KEYS
              value: "popular_posts,trending_media,user_preferences"
```

### 2. Configure Intelligent Cache TTLs
```javascript
// Dynamic TTL based on content popularity
const getCacheTTL = (key, accessCount) => {
  const baseTTL = 300; // 5 minutes
  const popularityMultiplier = Math.min(accessCount / 100, 10);
  return baseTTL * popularityMultiplier;
};
```

### 3. Set Up Proactive Monitoring
```yaml
# CloudWatch alarm for cache miss rate
CacheMissRateAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: High-Cache-Miss-Rate
    MetricName: CacheMissRate
    Namespace: Application/Cache
    Statistic: Average
    Period: 300
    EvaluationPeriods: 2
    Threshold: 50
    ComparisonOperator: GreaterThanThreshold
```

## Escalation Criteria

**Escalate to Platform Team Lead if:**
- Cache storm persists for more than 15 minutes
- Database connections are exhausted
- Multiple mitigation attempts fail

**Escalate to Database Team if:**
- Database performance severely degraded
- Need to implement emergency read replica scaling
- Suspected database-level issues

## Post-Incident Actions

1. Analyze cache invalidation patterns
2. Review cache key design and TTL strategies
3. Update cache warming schedules
4. Implement additional request coalescing
5. Review monitoring thresholds
6. Update cache architecture documentation

## Related Runbooks
- [Database Connection Issues](./database-connection-issues.md)
- [API Service Degradation](./api-service-degradation.md)
- [Hot Object Mitigation](./hot-object-mitigation.md)