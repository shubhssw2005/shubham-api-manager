# Hot Object Mitigation Runbook

## Incident Description
A single S3 object or small set of objects is receiving disproportionately high traffic, causing:
- Increased latency for requests to those objects
- Potential throttling from S3
- Degraded performance for other objects in the same prefix

## Impact Assessment
- **Severity 1**: >10,000 requests/second to single object, affecting overall service
- **Severity 2**: >1,000 requests/second causing noticeable latency increase
- **Severity 3**: Elevated traffic patterns detected but not yet impacting users

## Immediate Actions (First 5 minutes)

### 1. Identify Hot Objects
```bash
# Check CloudFront logs for top requested objects
aws logs start-query \
  --log-group-name /aws/cloudfront/distribution \
  --start-time $(date -d '1 hour ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, cs-uri-stem, sc-bytes, time-taken | filter sc-status = 200 | stats count() by cs-uri-stem | sort count desc | limit 10'

# Check S3 access logs if available
aws s3api select-object-content \
  --bucket access-logs-bucket \
  --key "$(date +%Y/%m/%d)/access-log" \
  --expression "SELECT object_key, COUNT(*) as request_count FROM s3object[*] WHERE time > '$(date -d '1 hour ago' --iso-8601)' GROUP BY object_key ORDER BY request_count DESC LIMIT 10" \
  --expression-type SQL \
  --input-serialization '{"JSON": {"Type": "LINES"}}' \
  --output-serialization '{"JSON": {}}'
```

### 2. Check Current Performance Impact
```bash
# Monitor CloudFront error rates
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name ErrorRate \
  --dimensions Name=DistributionId,Value=E1234567890123 \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum

# Check S3 request metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name AllRequests \
  --dimensions Name=BucketName,Value=media-bucket \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum
```

### 3. Immediate Notification
```bash
# Alert the team
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "HOT_OBJECT_INCIDENT_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Hot object detected - investigating traffic spike",
      "severity": "warning",
      "source": "s3-monitoring"
    }
  }'
```

## Investigation Steps

### 1. Analyze Traffic Patterns
```bash
# Get detailed CloudFront analytics
aws cloudfront get-distribution-config --id E1234567890123

# Check origin request patterns
aws logs get-query-results --query-id $(aws logs start-query \
  --log-group-name /aws/cloudfront/distribution \
  --start-time $(date -d '2 hours ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, cs-uri-stem, cs-referer, c-ip | filter cs-uri-stem like /hot-object-path/ | stats count() by c-ip | sort count desc' \
  --query 'queryId' --output text)
```

### 2. Identify Traffic Source
```bash
# Analyze referrer patterns
aws logs start-query \
  --log-group-name /aws/cloudfront/distribution \
  --start-time $(date -d '1 hour ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields cs-referer, cs-user-agent, c-ip | filter cs-uri-stem = "/path/to/hot/object" | stats count() by cs-referer'

# Check for bot traffic patterns
aws logs start-query \
  --log-group-name /aws/cloudfront/distribution \
  --start-time $(date -d '1 hour ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields cs-user-agent, c-ip | filter cs-uri-stem = "/path/to/hot/object" | stats count() by cs-user-agent'
```

### 3. Check Application Logs
```bash
# Look for application-level causes
kubectl logs -l app=api-service --since=1h | grep -i "hot-object-path"

# Check for any recent content changes
git log --oneline --since="4 hours ago" -- content/
```

## Resolution Steps

### Option 1: CloudFront Cache Optimization (Immediate)
```bash
# Create cache behavior for hot object with longer TTL
aws cloudfront get-distribution-config --id E1234567890123 > current-config.json

# Update cache behavior (modify current-config.json)
cat > cache-behavior-update.json << EOF
{
  "PathPattern": "/path/to/hot/object*",
  "TargetOriginId": "S3-media-bucket",
  "ViewerProtocolPolicy": "redirect-to-https",
  "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
  "Compress": true,
  "DefaultTTL": 86400,
  "MaxTTL": 31536000
}
EOF

# Apply the update
aws cloudfront update-distribution \
  --id E1234567890123 \
  --distribution-config file://updated-config.json \
  --if-match $(jq -r '.ETag' current-config.json)
```

### Option 2: Request Rate Limiting
```bash
# Implement WAF rate limiting rule
aws wafv2 create-rule-group \
  --scope CLOUDFRONT \
  --name HotObjectRateLimit \
  --capacity 100 \
  --rules '[
    {
      "Name": "RateLimitHotObject",
      "Priority": 1,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 1000,
          "AggregateKeyType": "IP",
          "ScopeDownStatement": {
            "ByteMatchStatement": {
              "SearchString": "/path/to/hot/object",
              "FieldToMatch": {"UriPath": {}},
              "TextTransformations": [{"Priority": 0, "Type": "NONE"}],
              "PositionalConstraint": "CONTAINS"
            }
          }
        }
      },
      "Action": {"Block": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "HotObjectRateLimit"
      }
    }
  ]'
```

### Option 3: Object Replication and Load Distribution
```bash
# Create multiple copies with different names
aws s3 cp s3://media-bucket/path/to/hot/object s3://media-bucket/path/to/hot/object-copy1
aws s3 cp s3://media-bucket/path/to/hot/object s3://media-bucket/path/to/hot/object-copy2
aws s3 cp s3://media-bucket/path/to/hot/object s3://media-bucket/path/to/hot/object-copy3

# Update application to use round-robin distribution
kubectl patch configmap api-config -p '{
  "data": {
    "HOT_OBJECT_COPIES": "3",
    "HOT_OBJECT_BASE_PATH": "/path/to/hot/object"
  }
}'
```

### Option 4: Move to Different S3 Prefix
```bash
# Copy object to different prefix for better distribution
aws s3 cp s3://media-bucket/hot/path/object s3://media-bucket/distributed/$(date +%s)/object

# Update application configuration
kubectl patch configmap api-config -p '{
  "data": {
    "HOT_OBJECT_NEW_PATH": "/distributed/'$(date +%s)'/object"
  }
}'
```

## Advanced Mitigation Strategies

### 1. Implement Application-Level Caching
```javascript
// Add to API service
const hotObjectCache = new Map();
const HOT_OBJECT_THRESHOLD = 100; // requests per minute

app.get('/api/media/:path', async (req, res) => {
  const objectPath = req.params.path;
  const requestCount = hotObjectCache.get(objectPath) || 0;
  
  if (requestCount > HOT_OBJECT_THRESHOLD) {
    // Serve from local cache or CDN
    return res.redirect(301, `https://cdn.example.com/cached/${objectPath}`);
  }
  
  hotObjectCache.set(objectPath, requestCount + 1);
  // Continue with normal processing
});
```

### 2. Dynamic Request Routing
```javascript
// Implement intelligent routing
const getOptimalS3Path = (objectPath, requestCount) => {
  if (requestCount > 1000) {
    // Use request prefix distribution
    const hash = crypto.createHash('md5').update(req.ip).digest('hex');
    const prefix = hash.substring(0, 2);
    return `distributed/${prefix}/${objectPath}`;
  }
  return objectPath;
};
```

## Verification Steps

### 1. Monitor Request Distribution
```bash
# Check if traffic is being distributed
aws logs start-query \
  --log-group-name /aws/cloudfront/distribution \
  --start-time $(date -d '15 minutes ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields cs-uri-stem | filter cs-uri-stem like /distributed/ | stats count() by cs-uri-stem'
```

### 2. Verify Performance Improvement
```bash
# Check CloudFront cache hit ratio
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=E1234567890123 \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# Monitor S3 request rates
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name AllRequests \
  --dimensions Name=BucketName,Value=media-bucket \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum
```

## Prevention Measures

### 1. Implement Proactive Monitoring
```yaml
# CloudWatch alarm for hot objects
HotObjectAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: S3-Hot-Object-Detection
    MetricName: AllRequests
    Namespace: AWS/S3
    Statistic: Sum
    Period: 300
    EvaluationPeriods: 2
    Threshold: 5000
    ComparisonOperator: GreaterThanThreshold
```

### 2. Automated Response System
```javascript
// Lambda function for automatic hot object detection
exports.handler = async (event) => {
  const s3Logs = await parseS3AccessLogs(event);
  const hotObjects = detectHotObjects(s3Logs);
  
  for (const hotObject of hotObjects) {
    if (hotObject.requestCount > THRESHOLD) {
      await createObjectCopies(hotObject.key);
      await updateLoadBalancerConfig(hotObject.key);
      await sendAlert(hotObject);
    }
  }
};
```

### 3. Content Distribution Strategy
```bash
# Implement predictive content distribution
# Add to deployment pipeline
aws s3 sync s3://media-bucket/popular/ s3://media-bucket/distributed/ \
  --exclude "*" \
  --include "*.jpg" \
  --include "*.png" \
  --include "*.mp4"
```

## Escalation Criteria

**Escalate to Platform Team Lead if:**
- Hot object traffic exceeds 50,000 requests/second
- Multiple mitigation attempts fail to reduce load
- S3 service limits are being approached

**Escalate to AWS Support if:**
- S3 request rate limits are being hit
- Need assistance with S3 Transfer Acceleration
- Regional S3 performance issues suspected

## Post-Incident Actions

1. Analyze root cause of traffic spike
2. Update content distribution strategy
3. Review and adjust monitoring thresholds
4. Document lessons learned
5. Update application caching strategies
6. Schedule review of content popularity patterns

## Related Runbooks
- [Cache Storm Response](./cache-storm-response.md)
- [S3 Service Degradation](./s3-service-degradation.md)
- [CDN Performance Issues](./cdn-performance-issues.md)