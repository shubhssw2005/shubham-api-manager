# Security Breach Response Runbook

## Incident Description
Suspected or confirmed security breach including unauthorized access, data exfiltration, or malicious activity.

## Impact Assessment
- **Severity 1**: Confirmed data breach with customer data exposure
- **Severity 2**: Unauthorized access to production systems
- **Severity 3**: Suspicious activity detected, investigation required

## Immediate Actions (First 10 minutes)

### 1. Contain the Threat
```bash
# Isolate affected systems
kubectl patch deployment suspicious-service -p '{"spec":{"replicas":0}}'

# Block suspicious IP addresses
aws wafv2 update-ip-set \
  --scope CLOUDFRONT \
  --id suspicious-ips \
  --addresses "192.168.1.100/32,10.0.0.50/32"

# Revoke compromised API tokens
kubectl delete secret compromised-api-tokens
```

### 2. Preserve Evidence
```bash
# Capture system state
kubectl get events --all-namespaces > security-incident-events.log
kubectl logs -l app=api-service --since=2h > security-incident-api-logs.log

# Export CloudTrail logs
aws logs create-export-task \
  --log-group-name CloudTrail/SecurityEvents \
  --from $(date -d '2 hours ago' +%s)000 \
  --to $(date +%s)000 \
  --destination s3://security-forensics-bucket
```

### 3. Notify Security Team
```bash
# Send high-priority alert
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "SECURITY_INCIDENT_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "SECURITY BREACH - Immediate response required",
      "severity": "critical",
      "source": "security-monitoring"
    }
  }'
```

## Investigation Steps

### 1. Analyze Access Logs
```bash
# Check for unusual access patterns
aws logs start-query \
  --log-group-name /aws/apigateway/access-logs \
  --start-time $(date -d '4 hours ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, sourceIPAddress, userAgent, status | filter status >= 400 | stats count() by sourceIPAddress | sort count desc'

# Review authentication logs
kubectl logs -l app=auth-service --since=4h | grep -E "(failed|unauthorized|suspicious)"
```

### 2. Check Data Access
```bash
# Review database access logs
kubectl exec -it deployment/api-service -- npm run db:audit-logs --since="4 hours ago"

# Check S3 access patterns
aws s3api get-bucket-logging --bucket media-bucket
aws logs filter-log-events \
  --log-group-name /aws/s3/access-logs \
  --start-time $(date -d '4 hours ago' +%s) \
  --filter-pattern "{ $.eventName = GetObject || $.eventName = PutObject }"
```

### 3. Validate System Integrity
```bash
# Check for unauthorized changes
git log --oneline --since="4 hours ago"
kubectl get configmaps -o yaml | grep -E "(modified|changed)"

# Scan for malware
kubectl exec -it security-scanner -- clamscan -r /app
```

## Resolution Steps

### Option 1: Credential Rotation (Immediate)
```bash
# Rotate all API keys
kubectl create job credential-rotation-$(date +%s) --from=cronjob/credential-rotator

# Update database passwords
aws secretsmanager update-secret \
  --secret-id prod/database/password \
  --generate-random-password \
  --password-length 32

# Regenerate JWT signing keys
kubectl delete secret jwt-signing-key
kubectl create secret generic jwt-signing-key --from-literal=key=$(openssl rand -base64 32)
```

### Option 2: Network Isolation
```bash
# Implement emergency network policies
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-isolation
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: trusted
EOF
```

### Option 3: System Lockdown
```bash
# Enable maintenance mode
kubectl patch configmap api-config -p '{"data":{"MAINTENANCE_MODE":"true"}}'

# Disable user registrations
kubectl patch configmap api-config -p '{"data":{"REGISTRATION_ENABLED":"false"}}'

# Enable enhanced logging
kubectl patch configmap api-config -p '{"data":{"LOG_LEVEL":"debug","AUDIT_ENABLED":"true"}}'
```

## Forensic Analysis

### 1. Timeline Reconstruction
```bash
# Correlate logs from multiple sources
aws logs start-query \
  --log-group-name /aws/lambda/security-correlator \
  --start-time $(date -d '6 hours ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, sourceIP, userAgent, action | sort @timestamp'
```

### 2. Impact Assessment
```bash
# Check affected user accounts
kubectl exec -it deployment/api-service -- npm run security:check-affected-users

# Verify data integrity
kubectl exec -it deployment/api-service -- npm run db:integrity-check

# Check for data exfiltration
aws s3api list-objects-v2 \
  --bucket audit-logs \
  --prefix "data-access/$(date +%Y/%m/%d)" \
  --query 'Contents[?Size > `10485760`]' # Files > 10MB
```

## Communication Plan

### Internal Communication
```bash
# Security team notification
echo "SECURITY INCIDENT: $(date)" | mail -s "Security Breach Response Activated" security-team@company.com

# Executive notification (for Severity 1)
echo "Critical security incident detected. Response team activated." | mail -s "URGENT: Security Incident" executives@company.com
```

### External Communication (if required)
- Legal team consultation for breach notification requirements
- Customer communication plan (if customer data affected)
- Regulatory notification (GDPR, CCPA, etc.)

## Recovery Steps

### 1. System Hardening
```bash
# Update security policies
kubectl apply -f k8s/security/enhanced-pod-security-policy.yaml

# Enable additional monitoring
kubectl apply -f k8s/security/security-monitoring.yaml

# Update WAF rules
aws wafv2 update-rule-group \
  --scope CLOUDFRONT \
  --id security-rules \
  --rules file://enhanced-security-rules.json
```

### 2. Validation Testing
```bash
# Run security scans
kubectl create job security-scan-$(date +%s) --from=cronjob/security-scanner

# Test access controls
kubectl exec -it security-tester -- npm run test:access-controls

# Validate encryption
kubectl exec -it security-tester -- npm run test:encryption
```

## Prevention Measures

### 1. Enhanced Monitoring
```yaml
# Additional security alerts
SecurityAnomalyDetection:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: Security-Anomaly-Detection
    MetricName: AnomalousActivity
    Namespace: Security/Monitoring
    Statistic: Sum
    Period: 300
    EvaluationPeriods: 1
    Threshold: 1
    ComparisonOperator: GreaterThanOrEqualToThreshold
```

### 2. Access Controls
```bash
# Implement zero-trust networking
kubectl apply -f k8s/security/zero-trust-policies.yaml

# Enable MFA for all admin access
aws iam put-user-policy \
  --user-name admin-users \
  --policy-name RequireMFA \
  --policy-document file://require-mfa-policy.json
```

### 3. Regular Security Assessments
```bash
# Schedule penetration testing
# Add to cron: 0 2 1 * * /scripts/security-assessment.sh

# Automated vulnerability scanning
kubectl create cronjob vuln-scanner \
  --image=security/vulnerability-scanner \
  --schedule="0 6 * * *" \
  --restart=OnFailure
```

## Escalation Criteria

**Escalate to CISO if:**
- Customer data confirmed compromised
- Regulatory notification required
- Media attention likely

**Escalate to Legal if:**
- Breach notification laws triggered
- Law enforcement involvement needed
- Contractual obligations affected

**Escalate to Executive Team if:**
- Business operations significantly impacted
- Reputation risk high
- Financial impact substantial

## Post-Incident Actions

1. Complete forensic analysis report
2. Update security policies and procedures
3. Conduct security training for affected teams
4. Review and update incident response procedures
5. Implement additional security controls
6. Schedule follow-up security assessment

## Related Runbooks
- [DDoS Attack Mitigation](./ddos-mitigation.md)
- [Certificate Expiry](./certificate-expiry.md)
- [API Service Degradation](./api-service-degradation.md)