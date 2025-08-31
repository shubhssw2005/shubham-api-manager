"""
Automated Failover Lambda Function
Handles Aurora Global Database failover and DNS updates
"""

import json
import boto3
import logging
import os
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
rds_client = boto3.client('rds')
route53_client = boto3.client('route53')
sns_client = boto3.client('sns')

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for automated failover
    
    Args:
        event: CloudWatch alarm event or manual trigger
        context: Lambda context
        
    Returns:
        Response dictionary with status and details
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        global_cluster_id = os.environ['GLOBAL_CLUSTER_ID']
        secondary_cluster_id = os.environ['SECONDARY_CLUSTER_ID']
        sns_topic_arn = os.environ['SNS_TOPIC_ARN']
        region = os.environ['REGION']
        
        # Determine if this is an automated failover or manual trigger
        is_manual = event.get('source') == 'manual'
        
        # Validate cluster health before failover
        if not is_manual:
            if not should_trigger_failover(event):
                logger.info("Failover conditions not met, skipping")
                return {
                    'statusCode': 200,
                    'body': json.dumps('Failover conditions not met')
                }
        
        # Perform Aurora Global Database failover
        failover_result = perform_aurora_failover(
            global_cluster_id, 
            secondary_cluster_id
        )
        
        if not failover_result['success']:
            raise Exception(f"Aurora failover failed: {failover_result['error']}")
        
        # Update DNS records to point to secondary region
        dns_result = update_dns_records(region)
        
        # Send notification
        notification_message = {
            'event': 'disaster_recovery_failover',
            'timestamp': context.aws_request_id,
            'region': region,
            'global_cluster_id': global_cluster_id,
            'secondary_cluster_id': secondary_cluster_id,
            'aurora_failover': failover_result,
            'dns_update': dns_result,
            'manual_trigger': is_manual
        }
        
        send_notification(sns_topic_arn, notification_message)
        
        logger.info("Failover completed successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Failover completed successfully',
                'details': notification_message
            })
        }
        
    except Exception as e:
        logger.error(f"Failover failed: {str(e)}")
        
        # Send failure notification
        failure_message = {
            'event': 'disaster_recovery_failover_failed',
            'timestamp': context.aws_request_id,
            'error': str(e),
            'region': region
        }
        
        try:
            send_notification(sns_topic_arn, failure_message)
        except Exception as notification_error:
            logger.error(f"Failed to send failure notification: {notification_error}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'details': failure_message
            })
        }

def should_trigger_failover(event: Dict[str, Any]) -> bool:
    """
    Determine if failover should be triggered based on the event
    
    Args:
        event: CloudWatch alarm event
        
    Returns:
        Boolean indicating if failover should proceed
    """
    try:
        # Check if this is a CloudWatch alarm
        if event.get('source') == 'aws.cloudwatch':
            alarm_data = event.get('detail', {})
            alarm_name = alarm_data.get('alarmName', '')
            new_state = alarm_data.get('newState', {}).get('value', '')
            
            # Only trigger on ALARM state for critical alarms
            if new_state == 'ALARM' and 'critical' in alarm_name.lower():
                logger.info(f"Critical alarm triggered: {alarm_name}")
                return True
        
        # Check for manual trigger
        if event.get('trigger_failover') is True:
            logger.info("Manual failover trigger detected")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error evaluating failover conditions: {e}")
        return False

def perform_aurora_failover(global_cluster_id: str, target_cluster_id: str) -> Dict[str, Any]:
    """
    Perform Aurora Global Database failover
    
    Args:
        global_cluster_id: Global cluster identifier
        target_cluster_id: Target cluster for failover
        
    Returns:
        Dictionary with success status and details
    """
    try:
        logger.info(f"Starting Aurora failover for global cluster: {global_cluster_id}")
        
        # Get current global cluster status
        response = rds_client.describe_global_clusters(
            GlobalClusterIdentifier=global_cluster_id
        )
        
        if not response['GlobalClusters']:
            raise Exception(f"Global cluster {global_cluster_id} not found")
        
        global_cluster = response['GlobalClusters'][0]
        logger.info(f"Current global cluster status: {global_cluster['Status']}")
        
        # Perform failover
        failover_response = rds_client.failover_global_cluster(
            GlobalClusterIdentifier=global_cluster_id,
            TargetDbClusterIdentifier=target_cluster_id
        )
        
        logger.info(f"Failover initiated: {failover_response}")
        
        # Wait for failover to complete (with timeout)
        waiter = rds_client.get_waiter('db_cluster_available')
        waiter.wait(
            DBClusterIdentifier=target_cluster_id,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 20  # 10 minutes timeout
            }
        )
        
        logger.info("Aurora failover completed successfully")
        
        return {
            'success': True,
            'global_cluster_id': global_cluster_id,
            'new_primary_cluster': target_cluster_id,
            'failover_time': failover_response['ResponseMetadata']['HTTPHeaders']['date']
        }
        
    except Exception as e:
        logger.error(f"Aurora failover failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def update_dns_records(region: str) -> Dict[str, Any]:
    """
    Update Route53 DNS records to point to secondary region
    
    Args:
        region: Secondary region name
        
    Returns:
        Dictionary with update status and details
    """
    try:
        logger.info(f"Updating DNS records for region: {region}")
        
        # This is a simplified example - in practice, you would:
        # 1. Get the hosted zone ID from environment or parameter store
        # 2. Update specific DNS records (API endpoints, etc.)
        # 3. Handle multiple record types and health checks
        
        # For now, we'll just log the action
        logger.info("DNS update completed (placeholder implementation)")
        
        return {
            'success': True,
            'region': region,
            'updated_records': []  # Would contain actual record updates
        }
        
    except Exception as e:
        logger.error(f"DNS update failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def send_notification(sns_topic_arn: str, message: Dict[str, Any]) -> None:
    """
    Send SNS notification about failover status
    
    Args:
        sns_topic_arn: SNS topic ARN
        message: Message dictionary to send
    """
    try:
        subject = f"Disaster Recovery: {message['event']}"
        
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject=subject,
            Message=json.dumps(message, indent=2, default=str)
        )
        
        logger.info(f"Notification sent: {subject}")
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        raise