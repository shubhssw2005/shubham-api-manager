"""
Automated Failover Monitoring Lambda Function
Monitors system health and triggers alerts for potential failover scenarios
"""

import json
import boto3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
rds_client = boto3.client('rds')
cloudwatch_client = boto3.client('cloudwatch')
sns_client = boto3.client('sns')

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for automated failover monitoring
    
    Args:
        event: CloudWatch alarm event or scheduled health check
        context: Lambda context
        
    Returns:
        Response dictionary with status and details
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get environment variables
        global_cluster_id = os.environ.get('GLOBAL_CLUSTER_ID')
        primary_cluster_id = os.environ.get('PRIMARY_CLUSTER_ID')
        sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
        region = os.environ.get('REGION')
        dr_region = os.environ.get('DR_REGION')
        failover_threshold_minutes = int(os.environ.get('FAILOVER_THRESHOLD_MINUTES', '5'))
        
        # Determine event type
        event_source = event.get('source', 'unknown')
        
        if event_source == 'aws.cloudwatch':
            # CloudWatch alarm triggered
            return handle_cloudwatch_alarm(event, {
                'global_cluster_id': global_cluster_id,
                'primary_cluster_id': primary_cluster_id,
                'sns_topic_arn': sns_topic_arn,
                'region': region,
                'dr_region': dr_region,
                'failover_threshold_minutes': failover_threshold_minutes
            })
        elif event_source == 'scheduled_health_check':
            # Scheduled health check
            return perform_health_check({
                'global_cluster_id': global_cluster_id,
                'primary_cluster_id': primary_cluster_id,
                'sns_topic_arn': sns_topic_arn,
                'region': region,
                'dr_region': dr_region
            })
        else:
            # Manual trigger or other event
            return handle_manual_trigger(event, {
                'global_cluster_id': global_cluster_id,
                'primary_cluster_id': primary_cluster_id,
                'sns_topic_arn': sns_topic_arn,
                'region': region,
                'dr_region': dr_region
            })
            
    except Exception as e:
        logger.error(f"Handler failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def handle_cloudwatch_alarm(event: Dict[str, Any], config: Dict[str, str]) -> Dict[str, Any]:
    """
    Handle CloudWatch alarm events
    
    Args:
        event: CloudWatch alarm event
        config: Configuration dictionary
        
    Returns:
        Response dictionary
    """
    try:
        alarm_data = event.get('detail', {})
        alarm_name = alarm_data.get('alarmName', '')
        new_state = alarm_data.get('newState', {}).get('value', '')
        
        logger.info(f"Processing CloudWatch alarm: {alarm_name}, state: {new_state}")
        
        if new_state == 'ALARM':
            # Evaluate if this should trigger failover
            should_failover = evaluate_failover_conditions(alarm_name, config)
            
            if should_failover:
                logger.warning(f"Failover conditions met for alarm: {alarm_name}")
                
                # Send critical alert (but don't auto-failover yet - requires manual approval)
                send_failover_alert(config['sns_topic_arn'], {
                    'type': 'failover_recommended',
                    'alarm_name': alarm_name,
                    'alarm_state': new_state,
                    'timestamp': datetime.utcnow().isoformat(),
                    'recommendation': 'Manual failover recommended - review and execute if necessary'
                })
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'Failover alert sent - manual intervention required',
                        'alarm': alarm_name,
                        'recommendation': 'failover'
                    })
                }
            else:
                # Send warning but no failover needed
                send_failover_alert(config['sns_topic_arn'], {
                    'type': 'alarm_warning',
                    'alarm_name': alarm_name,
                    'alarm_state': new_state,
                    'timestamp': datetime.utcnow().isoformat(),
                    'recommendation': 'Monitor situation - no immediate action required'
                })
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'Alarm processed - monitoring situation',
                        'alarm': alarm_name,
                        'recommendation': 'monitor'
                    })
                }
        else:
            # Alarm recovered
            send_failover_alert(config['sns_topic_arn'], {
                'type': 'alarm_recovered',
                'alarm_name': alarm_name,
                'alarm_state': new_state,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Alarm recovered',
                    'alarm': alarm_name
                })
            }
            
    except Exception as e:
        logger.error(f"Failed to handle CloudWatch alarm: {e}")
        raise

def perform_health_check(config: Dict[str, str]) -> Dict[str, Any]:
    """
    Perform scheduled health check
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Response dictionary
    """
    try:
        logger.info("Performing scheduled health check")
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check Aurora Global Database status
        if config['global_cluster_id']:
            aurora_status = check_aurora_global_database(config['global_cluster_id'])
            health_status['checks']['aurora_global_db'] = aurora_status
        
        # Check primary cluster status
        if config['primary_cluster_id']:
            primary_status = check_primary_cluster(config['primary_cluster_id'])
            health_status['checks']['primary_cluster'] = primary_status
        
        # Check replication lag
        replication_lag = check_replication_lag(config['region'], config['dr_region'])
        health_status['checks']['replication_lag'] = replication_lag
        
        # Evaluate overall health
        overall_health = evaluate_overall_health(health_status['checks'])
        health_status['overall_status'] = overall_health
        
        # Send health report if there are issues
        if overall_health != 'healthy':
            send_failover_alert(config['sns_topic_arn'], {
                'type': 'health_check_warning',
                'health_status': health_status,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Health check completed',
                'health_status': health_status
            })
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise

def handle_manual_trigger(event: Dict[str, Any], config: Dict[str, str]) -> Dict[str, Any]:
    """
    Handle manual trigger events
    
    Args:
        event: Manual trigger event
        config: Configuration dictionary
        
    Returns:
        Response dictionary
    """
    try:
        logger.info("Processing manual trigger")
        
        # This is a monitoring function - it doesn't perform actual failover
        # It just validates the request and sends appropriate alerts
        
        trigger_type = event.get('trigger_type', 'unknown')
        
        send_failover_alert(config['sns_topic_arn'], {
            'type': 'manual_trigger_received',
            'trigger_type': trigger_type,
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'note': 'Manual trigger received - use disaster recovery runbooks for actual failover'
        })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Manual trigger processed - check disaster recovery runbooks',
                'trigger_type': trigger_type
            })
        }
        
    except Exception as e:
        logger.error(f"Failed to handle manual trigger: {e}")
        raise

def evaluate_failover_conditions(alarm_name: str, config: Dict[str, str]) -> bool:
    """
    Evaluate if failover conditions are met
    
    Args:
        alarm_name: Name of the triggered alarm
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if failover should be recommended
    """
    try:
        # Define critical alarms that could trigger failover recommendation
        critical_alarms = [
            'aurora-cluster-unavailable',
            'aurora-replication-lag',
            'high-error-rate'
        ]
        
        # Check if this is a critical alarm
        is_critical = any(critical in alarm_name.lower() for critical in critical_alarms)
        
        if not is_critical:
            return False
        
        # For critical alarms, check additional conditions
        if 'aurora-cluster-unavailable' in alarm_name.lower():
            # Check if cluster is actually unavailable
            cluster_status = check_primary_cluster(config['primary_cluster_id'])
            return cluster_status.get('status') != 'available'
        
        elif 'aurora-replication-lag' in alarm_name.lower():
            # Check current replication lag
            lag_status = check_replication_lag(config['region'], config['dr_region'])
            return lag_status.get('lag_minutes', 0) > int(config['failover_threshold_minutes'])
        
        elif 'high-error-rate' in alarm_name.lower():
            # Check error rate trend
            error_rate = get_current_error_rate()
            return error_rate > 50  # 50% error rate threshold
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to evaluate failover conditions: {e}")
        return False

def check_aurora_global_database(global_cluster_id: str) -> Dict[str, Any]:
    """
    Check Aurora Global Database status
    
    Args:
        global_cluster_id: Global cluster identifier
        
    Returns:
        Status dictionary
    """
    try:
        response = rds_client.describe_global_clusters(
            GlobalClusterIdentifier=global_cluster_id
        )
        
        if not response['GlobalClusters']:
            return {'status': 'not_found', 'healthy': False}
        
        global_cluster = response['GlobalClusters'][0]
        status = global_cluster['Status']
        
        return {
            'status': status,
            'healthy': status == 'available',
            'members': len(global_cluster.get('GlobalClusterMembers', [])),
            'engine': global_cluster.get('Engine'),
            'engine_version': global_cluster.get('EngineVersion')
        }
        
    except Exception as e:
        logger.error(f"Failed to check Aurora Global Database: {e}")
        return {'status': 'error', 'healthy': False, 'error': str(e)}

def check_primary_cluster(primary_cluster_id: str) -> Dict[str, Any]:
    """
    Check primary cluster status
    
    Args:
        primary_cluster_id: Primary cluster identifier
        
    Returns:
        Status dictionary
    """
    try:
        response = rds_client.describe_db_clusters(
            DBClusterIdentifier=primary_cluster_id
        )
        
        if not response['DBClusters']:
            return {'status': 'not_found', 'healthy': False}
        
        cluster = response['DBClusters'][0]
        status = cluster['Status']
        
        return {
            'status': status,
            'healthy': status == 'available',
            'endpoint': cluster.get('Endpoint'),
            'reader_endpoint': cluster.get('ReaderEndpoint'),
            'backup_retention_period': cluster.get('BackupRetentionPeriod'),
            'members': len(cluster.get('DBClusterMembers', []))
        }
        
    except Exception as e:
        logger.error(f"Failed to check primary cluster: {e}")
        return {'status': 'error', 'healthy': False, 'error': str(e)}

def check_replication_lag(source_region: str, target_region: str) -> Dict[str, Any]:
    """
    Check Aurora Global Database replication lag
    
    Args:
        source_region: Source region
        target_region: Target region
        
    Returns:
        Lag status dictionary
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=10)
        
        response = cloudwatch_client.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='AuroraGlobalDBReplicationLag',
            Dimensions=[
                {'Name': 'SourceRegion', 'Value': source_region},
                {'Name': 'TargetRegion', 'Value': target_region}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum']
        )
        
        datapoints = response.get('Datapoints', [])
        
        if not datapoints:
            return {'status': 'no_data', 'healthy': False}
        
        # Get latest datapoint
        latest = max(datapoints, key=lambda x: x['Timestamp'])
        avg_lag_ms = latest.get('Average', 0)
        max_lag_ms = latest.get('Maximum', 0)
        
        # Convert to minutes
        avg_lag_minutes = avg_lag_ms / 60000
        max_lag_minutes = max_lag_ms / 60000
        
        return {
            'status': 'available',
            'healthy': avg_lag_minutes < 5,  # 5 minute threshold
            'avg_lag_minutes': avg_lag_minutes,
            'max_lag_minutes': max_lag_minutes,
            'timestamp': latest['Timestamp'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to check replication lag: {e}")
        return {'status': 'error', 'healthy': False, 'error': str(e)}

def get_current_error_rate() -> float:
    """
    Get current application error rate
    
    Returns:
        Error rate percentage
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        # This would need to be configured with actual load balancer ARN
        # For now, return 0 as placeholder
        return 0.0
        
    except Exception as e:
        logger.error(f"Failed to get error rate: {e}")
        return 0.0

def evaluate_overall_health(checks: Dict[str, Any]) -> str:
    """
    Evaluate overall system health
    
    Args:
        checks: Dictionary of health check results
        
    Returns:
        Overall health status string
    """
    try:
        unhealthy_checks = []
        
        for check_name, check_result in checks.items():
            if not check_result.get('healthy', False):
                unhealthy_checks.append(check_name)
        
        if not unhealthy_checks:
            return 'healthy'
        elif len(unhealthy_checks) == 1:
            return 'warning'
        else:
            return 'critical'
            
    except Exception as e:
        logger.error(f"Failed to evaluate overall health: {e}")
        return 'unknown'

def send_failover_alert(sns_topic_arn: str, alert_data: Dict[str, Any]) -> None:
    """
    Send SNS alert about failover status
    
    Args:
        sns_topic_arn: SNS topic ARN
        alert_data: Alert data dictionary
    """
    try:
        alert_type = alert_data.get('type', 'unknown')
        subject = f"DR Alert: {alert_type.replace('_', ' ').title()}"
        
        # Format message
        message = {
            'alert_type': alert_type,
            'timestamp': alert_data.get('timestamp', datetime.utcnow().isoformat()),
            'details': alert_data
        }
        
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject=subject,
            Message=json.dumps(message, indent=2, default=str)
        )
        
        logger.info(f"Alert sent: {subject}")
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        # Don't raise - alert failure shouldn't fail the monitoring