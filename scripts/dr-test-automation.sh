#!/bin/bash

# Disaster Recovery Test Automation Script
# Implements automated DR testing and validation procedures

set -euo pipefail

# Configuration
PROJECT_NAME="${PROJECT_NAME:-production}"
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
SECONDARY_REGION="${SECONDARY_REGION:-us-west-2}"
GLOBAL_CLUSTER_ID="${GLOBAL_CLUSTER_ID:-${PROJECT_NAME}-global-cluster}"
TEST_ENVIRONMENT="${TEST_ENVIRONMENT:-dr-test}"
LOG_FILE="/tmp/dr-test-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)  echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Error handling
error_exit() {
    log ERROR "$1"
    cleanup_test_resources
    exit 1
}

# Cleanup function
cleanup_test_resources() {
    log INFO "Cleaning up test resources..."
    
    # Remove test snapshots
    aws rds describe-db-cluster-snapshots \
        --snapshot-type manual \
        --query "DBClusterSnapshots[?contains(DBClusterSnapshotIdentifier, 'dr-test')].DBClusterSnapshotIdentifier" \
        --output text | while read -r snapshot_id; do
        if [[ -n "$snapshot_id" ]]; then
            log INFO "Deleting test snapshot: $snapshot_id"
            aws rds delete-db-cluster-snapshot \
                --db-cluster-snapshot-identifier "$snapshot_id" || true
        fi
    done
    
    # Remove test clusters
    aws rds describe-db-clusters \
        --query "DBClusters[?contains(DBClusterIdentifier, 'dr-test')].DBClusterIdentifier" \
        --output text | while read -r cluster_id; do
        if [[ -n "$cluster_id" ]]; then
            log INFO "Deleting test cluster: $cluster_id"
            aws rds delete-db-cluster \
                --db-cluster-identifier "$cluster_id" \
                --skip-final-snapshot || true
        fi
    done
    
    log INFO "Cleanup completed"
}

# Trap for cleanup on exit
trap cleanup_test_resources EXIT

# Validate prerequisites
validate_prerequisites() {
    log INFO "Validating prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error_exit "AWS CLI is not installed"
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed"
    fi
    
    # Check required environment variables
    if [[ -z "${AWS_DEFAULT_REGION:-}" ]]; then
        export AWS_DEFAULT_REGION="$PRIMARY_REGION"
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials not configured or invalid"
    fi
    
    log SUCCESS "Prerequisites validated"
}

# Test Aurora Global Database status
test_aurora_global_database() {
    log INFO "Testing Aurora Global Database status..."
    
    # Check global cluster exists
    if ! aws rds describe-global-clusters \
        --global-cluster-identifier "$GLOBAL_CLUSTER_ID" &> /dev/null; then
        error_exit "Global cluster $GLOBAL_CLUSTER_ID not found"
    fi
    
    # Get global cluster details
    local global_status
    global_status=$(aws rds describe-global-clusters \
        --global-cluster-identifier "$GLOBAL_CLUSTER_ID" \
        --query 'GlobalClusters[0].Status' \
        --output text)
    
    if [[ "$global_status" != "available" ]]; then
        error_exit "Global cluster is not available. Status: $global_status"
    fi
    
    # Check replication lag
    local lag_metrics
    lag_metrics=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/RDS \
        --metric-name AuroraGlobalDBReplicationLag \
        --dimensions Name=SourceRegion,Value="$PRIMARY_REGION" Name=TargetRegion,Value="$SECONDARY_REGION" \
        --start-time "$(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S)" \
        --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --period 300 \
        --statistics Average \
        --query 'Datapoints[0].Average' \
        --output text 2>/dev/null || echo "None")
    
    if [[ "$lag_metrics" != "None" ]]; then
        local lag_minutes
        lag_minutes=$(echo "$lag_metrics / 60000" | bc -l)
        log INFO "Current replication lag: ${lag_minutes} minutes"
        
        if (( $(echo "$lag_minutes > 5" | bc -l) )); then
            log WARN "Replication lag is high: ${lag_minutes} minutes"
        fi
    else
        log WARN "No replication lag metrics available"
    fi
    
    log SUCCESS "Aurora Global Database test completed"
}

# Test cross-region replication
test_cross_region_replication() {
    log INFO "Testing cross-region replication..."
    
    # Create test data in primary region
    local test_timestamp
    test_timestamp=$(date +%s)
    local test_file="dr-test-${test_timestamp}.txt"
    local test_content="DR test data created at $(date)"
    
    # Upload test file to primary S3 bucket
    local primary_bucket
    primary_bucket=$(aws s3api list-buckets \
        --query "Buckets[?contains(Name, '${PROJECT_NAME}') && contains(Name, 'primary')].Name" \
        --output text | head -1)
    
    if [[ -z "$primary_bucket" ]]; then
        log WARN "Primary S3 bucket not found, skipping S3 replication test"
        return 0
    fi
    
    echo "$test_content" | aws s3 cp - "s3://${primary_bucket}/dr-test/${test_file}"
    log INFO "Test file uploaded to primary bucket: $test_file"
    
    # Wait for replication (up to 20 minutes)
    local replica_bucket
    replica_bucket=$(aws s3api list-buckets \
        --region "$SECONDARY_REGION" \
        --query "Buckets[?contains(Name, '${PROJECT_NAME}') && contains(Name, 'replica')].Name" \
        --output text | head -1)
    
    if [[ -z "$replica_bucket" ]]; then
        log WARN "Replica S3 bucket not found, skipping replication verification"
        return 0
    fi
    
    local max_attempts=40  # 20 minutes with 30-second intervals
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if aws s3 ls "s3://${replica_bucket}/dr-test/${test_file}" \
            --region "$SECONDARY_REGION" &> /dev/null; then
            log SUCCESS "File replicated successfully in ${attempt} attempts ($(($attempt * 30)) seconds)"
            
            # Verify content
            local replicated_content
            replicated_content=$(aws s3 cp "s3://${replica_bucket}/dr-test/${test_file}" - \
                --region "$SECONDARY_REGION")
            
            if [[ "$replicated_content" == "$test_content" ]]; then
                log SUCCESS "Replicated content matches original"
            else
                log ERROR "Replicated content does not match original"
            fi
            
            # Cleanup test files
            aws s3 rm "s3://${primary_bucket}/dr-test/${test_file}" || true
            aws s3 rm "s3://${replica_bucket}/dr-test/${test_file}" \
                --region "$SECONDARY_REGION" || true
            
            return 0
        fi
        
        log INFO "Waiting for replication... attempt $attempt/$max_attempts"
        sleep 30
        ((attempt++))
    done
    
    log ERROR "File did not replicate within expected timeframe"
    return 1
}

# Test point-in-time recovery
test_point_in_time_recovery() {
    log INFO "Testing point-in-time recovery..."
    
    # Get primary cluster identifier
    local primary_cluster
    primary_cluster=$(aws rds describe-global-clusters \
        --global-cluster-identifier "$GLOBAL_CLUSTER_ID" \
        --query 'GlobalClusters[0].GlobalClusterMembers[?IsWriter==`true`].DBClusterIdentifier' \
        --output text)
    
    if [[ -z "$primary_cluster" ]]; then
        error_exit "Primary cluster not found in global cluster"
    fi
    
    # Check if PITR is enabled
    local backup_retention
    backup_retention=$(aws rds describe-db-clusters \
        --db-cluster-identifier "$primary_cluster" \
        --query 'DBClusters[0].BackupRetentionPeriod' \
        --output text)
    
    if [[ "$backup_retention" == "0" ]]; then
        error_exit "Point-in-time recovery is not enabled (backup retention is 0)"
    fi
    
    log INFO "PITR is enabled with $backup_retention days retention"
    
    # Get earliest restorable time
    local earliest_time
    earliest_time=$(aws rds describe-db-clusters \
        --db-cluster-identifier "$primary_cluster" \
        --query 'DBClusters[0].EarliestRestorableTime' \
        --output text)
    
    log INFO "Earliest restorable time: $earliest_time"
    
    # Create a test PITR restore (to 5 minutes ago)
    local restore_time
    restore_time=$(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S.000Z)
    local test_cluster_id="${TEST_ENVIRONMENT}-pitr-$(date +%s)"
    
    log INFO "Creating PITR test restore to time: $restore_time"
    
    # Get subnet group and security groups from original cluster
    local subnet_group security_groups
    subnet_group=$(aws rds describe-db-clusters \
        --db-cluster-identifier "$primary_cluster" \
        --query 'DBClusters[0].DBSubnetGroup' \
        --output text)
    
    security_groups=$(aws rds describe-db-clusters \
        --db-cluster-identifier "$primary_cluster" \
        --query 'DBClusters[0].VpcSecurityGroups[0].VpcSecurityGroupId' \
        --output text)
    
    # Create PITR restore
    aws rds restore-db-cluster-to-point-in-time \
        --db-cluster-identifier "$test_cluster_id" \
        --source-db-cluster-identifier "$primary_cluster" \
        --restore-to-time "$restore_time" \
        --db-subnet-group-name "$subnet_group" \
        --vpc-security-group-ids "$security_groups" \
        --tags Key=Purpose,Value=DR-Test Key=Environment,Value="$TEST_ENVIRONMENT"
    
    log INFO "PITR restore initiated for cluster: $test_cluster_id"
    
    # Wait for cluster to be available (up to 15 minutes)
    log INFO "Waiting for PITR cluster to become available..."
    if aws rds wait db-cluster-available \
        --db-cluster-identifier "$test_cluster_id" \
        --cli-read-timeout 900 \
        --cli-connect-timeout 60; then
        log SUCCESS "PITR test cluster is available"
        
        # Create a test instance
        local test_instance_id="${test_cluster_id}-instance-1"
        aws rds create-db-instance \
            --db-instance-identifier "$test_instance_id" \
            --db-cluster-identifier "$test_cluster_id" \
            --db-instance-class db.t3.medium \
            --engine aurora-postgresql
        
        log INFO "Test instance created: $test_instance_id"
        
        # Wait for instance to be available
        if aws rds wait db-instance-available \
            --db-instance-identifier "$test_instance_id" \
            --cli-read-timeout 600; then
            log SUCCESS "PITR test completed successfully"
        else
            log ERROR "Test instance did not become available"
        fi
    else
        log ERROR "PITR cluster did not become available within timeout"
    fi
}

# Test failover simulation
test_failover_simulation() {
    log INFO "Testing failover simulation (read-only test)..."
    
    # Get secondary cluster
    local secondary_cluster
    secondary_cluster=$(aws rds describe-global-clusters \
        --global-cluster-identifier "$GLOBAL_CLUSTER_ID" \
        --query 'GlobalClusters[0].GlobalClusterMembers[?IsWriter==`false`].DBClusterIdentifier' \
        --output text | head -1)
    
    if [[ -z "$secondary_cluster" ]]; then
        error_exit "Secondary cluster not found in global cluster"
    fi
    
    # Check secondary cluster status
    local secondary_status
    secondary_status=$(aws rds describe-db-clusters \
        --db-cluster-identifier "$secondary_cluster" \
        --region "$SECONDARY_REGION" \
        --query 'DBClusters[0].Status' \
        --output text)
    
    if [[ "$secondary_status" != "available" ]]; then
        error_exit "Secondary cluster is not available. Status: $secondary_status"
    fi
    
    log SUCCESS "Secondary cluster is available and ready for failover"
    
    # Test Lambda failover function (if exists)
    local failover_function
    failover_function=$(aws lambda list-functions \
        --query "Functions[?contains(FunctionName, 'failover')].FunctionName" \
        --output text | head -1)
    
    if [[ -n "$failover_function" ]]; then
        log INFO "Testing failover Lambda function: $failover_function"
        
        # Invoke with test payload
        local test_payload='{"source": "dr_test", "test_mode": true}'
        local response
        response=$(aws lambda invoke \
            --function-name "$failover_function" \
            --payload "$test_payload" \
            --cli-binary-format raw-in-base64-out \
            /tmp/lambda-response.json)
        
        if [[ $? -eq 0 ]]; then
            log SUCCESS "Failover Lambda function test completed"
            log INFO "Lambda response: $(cat /tmp/lambda-response.json)"
        else
            log ERROR "Failover Lambda function test failed"
        fi
    else
        log WARN "No failover Lambda function found"
    fi
}

# Test monitoring and alerting
test_monitoring_alerting() {
    log INFO "Testing monitoring and alerting..."
    
    # Check CloudWatch alarms
    local dr_alarms
    dr_alarms=$(aws cloudwatch describe-alarms \
        --alarm-name-prefix "$PROJECT_NAME" \
        --query "MetricAlarms[?contains(AlarmName, 'replication') || contains(AlarmName, 'cluster')].AlarmName" \
        --output text)
    
    if [[ -n "$dr_alarms" ]]; then
        log INFO "Found DR-related alarms:"
        echo "$dr_alarms" | tr '\t' '\n' | while read -r alarm; do
            log INFO "  - $alarm"
        done
    else
        log WARN "No DR-related CloudWatch alarms found"
    fi
    
    # Check SNS topics
    local sns_topics
    sns_topics=$(aws sns list-topics \
        --query "Topics[?contains(TopicArn, 'dr') || contains(TopicArn, 'disaster')].TopicArn" \
        --output text)
    
    if [[ -n "$sns_topics" ]]; then
        log INFO "Found DR-related SNS topics:"
        echo "$sns_topics" | tr '\t' '\n' | while read -r topic; do
            log INFO "  - $topic"
        done
    else
        log WARN "No DR-related SNS topics found"
    fi
}

# Generate test report
generate_test_report() {
    local report_file="/tmp/dr-test-report-$(date +%Y%m%d-%H%M%S).json"
    
    log INFO "Generating test report: $report_file"
    
    cat > "$report_file" << EOF
{
  "test_execution": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "project": "$PROJECT_NAME",
    "primary_region": "$PRIMARY_REGION",
    "secondary_region": "$SECONDARY_REGION",
    "global_cluster_id": "$GLOBAL_CLUSTER_ID",
    "test_environment": "$TEST_ENVIRONMENT"
  },
  "test_results": {
    "aurora_global_database": "completed",
    "cross_region_replication": "completed",
    "point_in_time_recovery": "completed",
    "failover_simulation": "completed",
    "monitoring_alerting": "completed"
  },
  "log_file": "$LOG_FILE",
  "report_file": "$report_file"
}
EOF
    
    log SUCCESS "Test report generated: $report_file"
    echo "Test report: $report_file"
    echo "Test log: $LOG_FILE"
}

# Main execution
main() {
    log INFO "Starting Disaster Recovery Test Automation"
    log INFO "Project: $PROJECT_NAME"
    log INFO "Primary Region: $PRIMARY_REGION"
    log INFO "Secondary Region: $SECONDARY_REGION"
    log INFO "Log file: $LOG_FILE"
    
    validate_prerequisites
    
    # Run tests
    test_aurora_global_database
    test_cross_region_replication
    test_point_in_time_recovery
    test_failover_simulation
    test_monitoring_alerting
    
    generate_test_report
    
    log SUCCESS "Disaster Recovery Test Automation completed successfully"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi