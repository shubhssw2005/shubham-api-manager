# Disaster Recovery Runbooks

This document provides step-by-step procedures for disaster recovery scenarios, including automated failover processes and manual recovery procedures.

## Table of Contents

1. [Overview](#overview)
2. [RTO/RPO Objectives](#rtorpo-objectives)
3. [Automated Failover Procedures](#automated-failover-procedures)
4. [Manual Failover Procedures](#manual-failover-procedures)
5. [Point-in-Time Recovery](#point-in-time-recovery)
6. [Regional Disaster Recovery](#regional-disaster-recovery)
7. [Data Recovery Procedures](#data-recovery-procedures)
8. [Rollback Procedures](#rollback-procedures)
9. [Testing and Validation](#testing-and-validation)
10. [Emergency Contacts](#emergency-contacts)

## Overview

Our disaster recovery system is designed to meet the following objectives:
- **RTO (Recovery Time Objective)**: 15 minutes
- **RPO (Recovery Point Objective)**: 5 minutes
- **Availability Target**: 99.99%

### Architecture Components

- **Primary Region**: us-east-1
- **Secondary Region**: us-west-2
- **Tertiary Region**: eu-west-1
- **Aurora Global Database**: Cross-region replication
- **S3 Cross-Region Replication**: 15-minute replication time
- **Automated Failover**: Lambda-based automation

## RTO/RPO Objectives

| Scenario | RTO Target | RPO Target | Recovery Method |
|----------|------------|------------|-----------------|
| Database Failure | 5 minutes | 0 minutes | Aurora Multi-AZ |
| Regional Outage | 15 minutes | 5 minutes | Global Database Failover |
| Data Corruption | 30 minutes | Variable | Point-in-Time Recovery |
| Complete Disaster | 60 minutes | 15 minutes | Cross-Region Restore |

## Automated Failover Procedures

### Aurora Global Database Failover

The system includes automated failover for Aurora Global Database failures.

#### Trigger Conditions

Automated failover is triggered when:
- Primary cluster becomes unavailable for >5 minutes
- Primary cluster error rate >50% for >3 minutes
- Manual trigger via CloudWatch alarm or API call

#### Automated Process

1. **Detection**: CloudWatch alarms detect failure conditions
2. **Validation**: Lambda function validates failure state
3. **Failover**: Aurora Global Database failover initiated
4. **DNS Update**: Route53 records updated to secondary region
5. **Notification**: SNS alerts sent to operations team
6. **Monitoring**: Continuous health checks on new primary

#### Monitoring the Process

```bash
# Check failover Lambda logs
aws logs tail /aws/lambda/disaster-recovery-failover --follow

# Monitor Aurora Global Database status
aws rds describe-global-clusters --global-cluster-identifier production-global-cluster

# Check Route53 health checks
aws route53 get-health-check --health-check-id HEALTH_CHECK_ID
```

### Manual Failover Override

If automated failover fails or needs to be bypassed:

```bash
# Trigger manual failover
aws lambda invoke \
  --function-name disaster-recovery-failover \
  --payload '{"trigger_failover": true, "source": "manual"}' \
  response.json

# Check response
cat response.json
```

## Manual Failover Procedures

### Step 1: Assess the Situation

1. **Verify the Outage**
   ```bash
   # Check primary cluster status
   aws rds describe-db-clusters --db-cluster-identifier production-primary-cluster
   
   # Check application health
   curl -f https://api.yourdomain.com/health || echo "API unavailable"
   
   # Check CloudWatch metrics
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name DatabaseConnections \
     --dimensions Name=DBClusterIdentifier,Value=production-primary-cluster \
     --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 300 \
     --statistics Average
   ```

2. **Determine Scope**
   - Single service failure
   - Database cluster failure
   - Regional outage
   - Complete disaster

### Step 2: Execute Failover

#### Database Cluster Failover

```bash
# 1. Promote secondary cluster to primary
aws rds failover-global-cluster \
  --global-cluster-identifier production-global-cluster \
  --target-db-cluster-identifier production-secondary-cluster

# 2. Wait for failover completion
aws rds wait db-cluster-available \
  --db-cluster-identifier production-secondary-cluster

# 3. Update application configuration
kubectl patch configmap app-config \
  -p '{"data":{"DATABASE_HOST":"production-secondary-cluster.cluster-xyz.us-west-2.rds.amazonaws.com"}}'

# 4. Restart application pods
kubectl rollout restart deployment/api-service
kubectl rollout restart deployment/media-service
```

#### Regional Failover

```bash
# 1. Update Route53 to point to secondary region
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch file://failover-dns-change.json

# 2. Scale up secondary region infrastructure
kubectl config use-context secondary-region
kubectl scale deployment/api-service --replicas=10
kubectl scale deployment/media-service --replicas=5

# 3. Verify services are healthy
kubectl get pods -l app=api-service
kubectl get pods -l app=media-service
```

### Step 3: Validate Recovery

```bash
# 1. Test database connectivity
psql -h production-secondary-cluster.cluster-xyz.us-west-2.rds.amazonaws.com \
     -U postgres -d production -c "SELECT 1;"

# 2. Test API endpoints
curl -f https://api.yourdomain.com/health
curl -f https://api.yourdomain.com/api/posts?limit=1

# 3. Test media uploads
curl -X POST https://api.yourdomain.com/api/media/presigned-url \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.jpg","contentType":"image/jpeg","size":1024}'

# 4. Monitor error rates
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name HTTPCode_Target_5XX_Count \
  --dimensions Name=LoadBalancer,Value=app/production-alb/xyz \
  --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum
```

## Point-in-Time Recovery

### When to Use PITR

- Data corruption detected
- Accidental data deletion
- Application bug caused data issues
- Need to recover to specific timestamp

### PITR Procedure

```bash
# 1. Identify target recovery time
TARGET_TIME="2024-01-15T10:30:00.000Z"

# 2. Create new cluster from PITR
aws rds restore-db-cluster-to-point-in-time \
  --db-cluster-identifier production-pitr-recovery \
  --source-db-cluster-identifier production-primary-cluster \
  --restore-to-time $TARGET_TIME \
  --vpc-security-group-ids sg-12345678 \
  --db-subnet-group-name production-db-subnet-group

# 3. Wait for cluster to be available
aws rds wait db-cluster-available \
  --db-cluster-identifier production-pitr-recovery

# 4. Create cluster instances
aws rds create-db-instance \
  --db-instance-identifier production-pitr-recovery-1 \
  --db-cluster-identifier production-pitr-recovery \
  --db-instance-class db.r6g.large \
  --engine aurora-postgresql

# 5. Validate recovered data
psql -h production-pitr-recovery.cluster-xyz.us-east-1.rds.amazonaws.com \
     -U postgres -d production -c "SELECT COUNT(*) FROM posts WHERE created_at <= '$TARGET_TIME';"

# 6. If validation successful, promote to production
# (Follow manual failover procedures above)
```

### S3 Point-in-Time Recovery

```bash
# 1. List object versions around target time
aws s3api list-object-versions \
  --bucket production-media-bucket \
  --prefix "tenants/tenant-123/" \
  --max-items 100

# 2. Restore specific object version
aws s3api restore-object \
  --bucket production-media-bucket \
  --key "tenants/tenant-123/media/file.jpg" \
  --version-id "version-id-here" \
  --restore-request Days=7,GlacierJobParameters='{Tier=Expedited}'

# 3. Copy restored object to new location
aws s3 cp s3://production-media-bucket/tenants/tenant-123/media/file.jpg \
          s3://production-media-bucket/tenants/tenant-123/media/file-restored.jpg
```

## Regional Disaster Recovery

### Complete Regional Failure

#### Immediate Actions (0-5 minutes)

1. **Activate Incident Response**
   ```bash
   # Send alert to operations team
   aws sns publish \
     --topic-arn arn:aws:sns:us-east-1:123456789:disaster-recovery-alerts \
     --subject "CRITICAL: Regional Disaster Recovery Activated" \
     --message "Primary region (us-east-1) is experiencing complete failure. Initiating DR procedures."
   ```

2. **Assess Secondary Region Status**
   ```bash
   # Switch to secondary region
   export AWS_DEFAULT_REGION=us-west-2
   
   # Check Aurora Global Database status
   aws rds describe-global-clusters --global-cluster-identifier production-global-cluster
   
   # Check EKS cluster status
   kubectl config use-context secondary-region
   kubectl get nodes
   kubectl get pods --all-namespaces
   ```

#### Recovery Actions (5-15 minutes)

1. **Promote Secondary Database**
   ```bash
   # Detach secondary cluster from global cluster (makes it independent)
   aws rds remove-from-global-cluster \
     --global-cluster-identifier production-global-cluster \
     --db-cluster-identifier production-secondary-cluster
   
   # The cluster is now a standalone primary cluster
   ```

2. **Scale Up Secondary Region**
   ```bash
   # Scale up application services
   kubectl scale deployment/api-service --replicas=20
   kubectl scale deployment/media-service --replicas=10
   kubectl scale deployment/worker-service --replicas=15
   
   # Scale up node groups
   aws eks update-nodegroup-config \
     --cluster-name production-secondary \
     --nodegroup-name general-purpose \
     --scaling-config minSize=5,maxSize=50,desiredSize=20
   ```

3. **Update DNS and Load Balancers**
   ```bash
   # Update Route53 to point to secondary region
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z123456789 \
     --change-batch '{
       "Changes": [{
         "Action": "UPSERT",
         "ResourceRecordSet": {
           "Name": "api.yourdomain.com",
           "Type": "CNAME",
           "TTL": 60,
           "ResourceRecords": [{"Value": "production-alb-secondary.us-west-2.elb.amazonaws.com"}]
         }
       }]
     }'
   ```

#### Validation (15-30 minutes)

```bash
# 1. Test all critical endpoints
./scripts/health-check-full.sh

# 2. Verify data integrity
./scripts/data-integrity-check.sh

# 3. Monitor performance metrics
aws cloudwatch get-dashboard --dashboard-name "Production-Secondary-Region"

# 4. Test user workflows
./scripts/end-to-end-tests.sh
```

## Data Recovery Procedures

### Database Recovery

#### From Automated Backup

```bash
# 1. List available backups
aws rds describe-db-cluster-snapshots \
  --db-cluster-identifier production-primary-cluster \
  --snapshot-type automated

# 2. Restore from specific backup
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier production-restored \
  --snapshot-identifier rds:production-primary-cluster-2024-01-15-03-00

# 3. Create instances for restored cluster
aws rds create-db-instance \
  --db-instance-identifier production-restored-1 \
  --db-cluster-identifier production-restored \
  --db-instance-class db.r6g.large \
  --engine aurora-postgresql
```

#### From Manual Snapshot

```bash
# 1. Create manual snapshot (if source still available)
aws rds create-db-cluster-snapshot \
  --db-cluster-identifier production-primary-cluster \
  --db-cluster-snapshot-identifier manual-recovery-$(date +%Y%m%d-%H%M%S)

# 2. Restore from manual snapshot
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier production-manual-restored \
  --snapshot-identifier manual-recovery-20240115-103000
```

### S3 Data Recovery

#### From Cross-Region Replica

```bash
# 1. Sync from replica bucket
aws s3 sync s3://production-media-replica-bucket s3://production-media-bucket-recovered \
  --delete --exact-timestamps

# 2. Update application configuration
kubectl patch configmap app-config \
  -p '{"data":{"MEDIA_BUCKET":"production-media-bucket-recovered"}}'
```

#### From Versioned Objects

```bash
# 1. Restore all objects to specific timestamp
./scripts/s3-restore-to-timestamp.sh "2024-01-15T10:30:00Z"

# 2. Verify restored objects
aws s3 ls s3://production-media-bucket --recursive --human-readable
```

## Rollback Procedures

### Database Rollback

If the failover was unsuccessful or caused issues:

```bash
# 1. Stop application traffic
kubectl scale deployment/api-service --replicas=0

# 2. Create snapshot of current state
aws rds create-db-cluster-snapshot \
  --db-cluster-identifier production-secondary-cluster \
  --db-cluster-snapshot-identifier rollback-point-$(date +%Y%m%d-%H%M%S)

# 3. If original primary is recoverable, fail back
aws rds failover-global-cluster \
  --global-cluster-identifier production-global-cluster \
  --target-db-cluster-identifier production-primary-cluster

# 4. Restore application traffic
kubectl scale deployment/api-service --replicas=10
```

### DNS Rollback

```bash
# Revert DNS changes
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.yourdomain.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "production-alb-primary.us-east-1.elb.amazonaws.com"}]
      }
    }]
  }'
```

## Testing and Validation

### Quarterly DR Drills

#### Preparation Checklist

- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Prepare test data validation scripts
- [ ] Set up monitoring dashboards
- [ ] Brief operations team

#### Drill Procedure

```bash
# 1. Create test environment snapshot
./scripts/create-dr-test-environment.sh

# 2. Execute failover simulation
./scripts/simulate-regional-failure.sh

# 3. Run automated tests
./scripts/dr-validation-tests.sh

# 4. Document results
./scripts/generate-dr-test-report.sh

# 5. Clean up test resources
./scripts/cleanup-dr-test-environment.sh
```

#### Success Criteria

- [ ] RTO < 15 minutes
- [ ] RPO < 5 minutes
- [ ] All critical services operational
- [ ] Data integrity verified
- [ ] Performance within acceptable limits

### Automated Testing

```bash
# Daily automated DR readiness checks
crontab -e
# Add: 0 6 * * * /opt/scripts/dr-readiness-check.sh

# Weekly cross-region replication validation
# Add: 0 2 * * 0 /opt/scripts/validate-cross-region-replication.sh
```

## Emergency Contacts

### Primary Contacts

| Role | Name | Phone | Email | Escalation Time |
|------|------|-------|-------|-----------------|
| On-Call Engineer | [Name] | [Phone] | [Email] | Immediate |
| Database Admin | [Name] | [Phone] | [Email] | 15 minutes |
| Infrastructure Lead | [Name] | [Phone] | [Email] | 30 minutes |
| Engineering Manager | [Name] | [Phone] | [Email] | 1 hour |

### Vendor Contacts

| Vendor | Support Level | Phone | Case Portal | SLA |
|--------|---------------|-------|-------------|-----|
| AWS | Enterprise | 1-800-xxx-xxxx | AWS Console | 15 minutes |
| DataDog | Pro | [Phone] | [Portal] | 1 hour |

### Communication Channels

- **Slack**: #incident-response
- **PagerDuty**: [Integration Key]
- **Status Page**: status.yourdomain.com
- **War Room**: [Conference Bridge]

## Appendix

### Useful Commands Reference

```bash
# Quick health checks
alias check-primary="aws rds describe-db-clusters --db-cluster-identifier production-primary-cluster --query 'DBClusters[0].Status'"
alias check-secondary="aws rds describe-db-clusters --db-cluster-identifier production-secondary-cluster --query 'DBClusters[0].Status'"
alias check-global="aws rds describe-global-clusters --global-cluster-identifier production-global-cluster --query 'GlobalClusters[0].Status'"

# Quick failover
alias trigger-failover="aws lambda invoke --function-name disaster-recovery-failover --payload '{\"trigger_failover\": true}' response.json && cat response.json"

# Monitor replication lag
alias check-lag="aws cloudwatch get-metric-statistics --namespace AWS/RDS --metric-name AuroraGlobalDBReplicationLag --dimensions Name=SourceRegion,Value=us-east-1 Name=TargetRegion,Value=us-west-2 --start-time \$(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) --end-time \$(date -u +%Y-%m-%dT%H:%M:%S) --period 300 --statistics Average"
```

### Configuration Files

- **DNS Change Template**: `/opt/config/failover-dns-change.json`
- **Scaling Configuration**: `/opt/config/dr-scaling-config.yaml`
- **Health Check Scripts**: `/opt/scripts/health-checks/`
- **Validation Scripts**: `/opt/scripts/validation/`

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Next Review**: [Quarterly]  
**Owner**: Infrastructure Team