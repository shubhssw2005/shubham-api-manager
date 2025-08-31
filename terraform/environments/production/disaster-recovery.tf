# Production Environment - Disaster Recovery Configuration

# Primary region disaster recovery module
module "disaster_recovery_primary" {
  source = "../../modules/disaster-recovery"

  project_name = var.project_name
  environment  = "production"

  # Aurora Global Database Configuration
  enable_global_database    = true
  aurora_engine_version    = "15.4"
  database_name           = "production"
  master_username         = "postgres"
  primary_instance_count  = 3
  primary_instance_class  = "db.r6g.xlarge"

  # Security Configuration
  security_group_ids    = [module.vpc.database_security_group_id]
  db_subnet_group_name = module.vpc.database_subnet_group_name

  # Backup Configuration
  backup_retention_period = 35
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  deletion_protection    = true

  # S3 Cross-Region Replication
  enable_s3_replication = true
  replica_bucket_arn    = module.disaster_recovery_secondary.replica_s3_bucket_arn
  replica_bucket_name   = module.disaster_recovery_secondary.replica_s3_bucket_id
  replica_kms_key_arn   = module.disaster_recovery_secondary.dr_secondary_kms_key_arn

  # Disaster Recovery Configuration
  dr_region   = "us-west-2"
  rto_minutes = 15
  rpo_minutes = 5

  # Monitoring and Alerting
  sns_topic_arn              = aws_sns_topic.production_alerts.arn
  alert_email_addresses      = var.alert_email_addresses
  enable_automated_failover  = true
  load_balancer_arn_suffix   = module.load_balancer.arn_suffix

  # Backup Automation
  enable_automated_backups     = true
  backup_schedule_expression   = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC

  tags = {
    Environment = "production"
    Purpose     = "disaster-recovery"
    Terraform   = "true"
  }
}

# Secondary region disaster recovery module
module "disaster_recovery_secondary" {
  source = "../../modules/disaster-recovery-secondary"

  providers = {
    aws = aws.us_west_2
  }

  project_name              = var.project_name
  environment              = "production"
  global_cluster_identifier = module.disaster_recovery_primary.global_cluster_id

  # Aurora Configuration
  enable_secondary_cluster  = true
  aurora_engine_version    = "15.4"
  secondary_instance_count = 2
  secondary_instance_class = "db.r6g.large"

  # Security Configuration (in secondary region)
  security_group_ids    = [module.vpc_secondary.database_security_group_id]
  db_subnet_group_name = module.vpc_secondary.database_subnet_group_name

  # Backup Configuration
  backup_retention_period = 35
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  deletion_protection    = true

  # Monitoring
  sns_topic_arn = aws_sns_topic.production_alerts_secondary.arn

  # Failover Automation
  enable_failover_automation = true
  rto_minutes               = 15
  rpo_minutes               = 5

  tags = {
    Environment = "production"
    Purpose     = "disaster-recovery-secondary"
    Terraform   = "true"
  }
}

# SNS topic for production alerts (primary region)
resource "aws_sns_topic" "production_alerts" {
  name = "${var.project_name}-production-alerts"

  tags = {
    Environment = "production"
    Purpose     = "alerts"
  }
}

# SNS topic for production alerts (secondary region)
resource "aws_sns_topic" "production_alerts_secondary" {
  provider = aws.us_west_2
  name     = "${var.project_name}-production-alerts-secondary"

  tags = {
    Environment = "production"
    Purpose     = "alerts-secondary"
  }
}

# SNS topic subscriptions for email alerts
resource "aws_sns_topic_subscription" "production_email_alerts" {
  count     = length(var.alert_email_addresses)
  topic_arn = aws_sns_topic.production_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}

resource "aws_sns_topic_subscription" "production_email_alerts_secondary" {
  count     = length(var.alert_email_addresses)
  provider  = aws.us_west_2
  topic_arn = aws_sns_topic.production_alerts_secondary.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}

# Route53 health checks for automated DNS failover
resource "aws_route53_health_check" "primary_region" {
  fqdn                            = "api.${var.domain_name}"
  port                           = 443
  type                           = "HTTPS"
  resource_path                  = "/health"
  failure_threshold              = 3
  request_interval               = 30
  cloudwatch_alarm_region        = var.primary_region
  cloudwatch_alarm_name          = "${var.project_name}-primary-health-check"
  insufficient_data_health_status = "Failure"

  tags = {
    Name        = "${var.project_name}-primary-health-check"
    Environment = "production"
  }
}

resource "aws_route53_health_check" "secondary_region" {
  fqdn                            = "api-secondary.${var.domain_name}"
  port                           = 443
  type                           = "HTTPS"
  resource_path                  = "/health"
  failure_threshold              = 3
  request_interval               = 30
  cloudwatch_alarm_region        = var.secondary_region
  cloudwatch_alarm_name          = "${var.project_name}-secondary-health-check"
  insufficient_data_health_status = "Failure"

  tags = {
    Name        = "${var.project_name}-secondary-health-check"
    Environment = "production"
  }
}

# Route53 DNS records with health check failover
resource "aws_route53_record" "api_primary" {
  zone_id = var.route53_zone_id
  name    = "api.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60

  set_identifier = "primary"
  
  failover_routing_policy {
    type = "PRIMARY"
  }

  health_check_id = aws_route53_health_check.primary_region.id
  records         = [module.load_balancer.dns_name]
}

resource "aws_route53_record" "api_secondary" {
  zone_id = var.route53_zone_id
  name    = "api.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60

  set_identifier = "secondary"
  
  failover_routing_policy {
    type = "SECONDARY"
  }

  health_check_id = aws_route53_health_check.secondary_region.id
  records         = [module.load_balancer_secondary.dns_name]
}

# CloudWatch dashboard for disaster recovery overview
resource "aws_cloudwatch_dashboard" "disaster_recovery_overview" {
  dashboard_name = "${var.project_name}-disaster-recovery-overview"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 24
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "AuroraGlobalDBReplicationLag", "SourceRegion", var.primary_region, "TargetRegion", var.secondary_region],
            ["AWS/RDS", "DatabaseConnections", "DBClusterIdentifier", module.disaster_recovery_primary.primary_cluster_id],
            ["AWS/S3", "ReplicationLatency", "SourceBucket", module.disaster_recovery_primary.primary_s3_bucket_id, "DestinationBucket", module.disaster_recovery_secondary.replica_s3_bucket_id]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.primary_region
          title   = "Disaster Recovery Metrics"
          period  = 300
          yAxis = {
            left = {
              min = 0
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Route53", "HealthCheckStatus", "HealthCheckId", aws_route53_health_check.primary_region.id],
            ["AWS/Route53", "HealthCheckStatus", "HealthCheckId", aws_route53_health_check.secondary_region.id]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"  # Route53 metrics are in us-east-1
          title   = "Health Check Status"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", module.load_balancer.arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", module.load_balancer.arn_suffix]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.primary_region
          title   = "Application Performance"
          period  = 300
        }
      }
    ]
  })

  tags = {
    Environment = "production"
    Purpose     = "disaster-recovery-monitoring"
  }
}

# Outputs
output "disaster_recovery_primary" {
  description = "Primary region disaster recovery configuration"
  value = {
    global_cluster_id     = module.disaster_recovery_primary.global_cluster_id
    primary_cluster_id    = module.disaster_recovery_primary.primary_cluster_id
    primary_s3_bucket_id  = module.disaster_recovery_primary.primary_s3_bucket_id
    dr_kms_key_arn       = module.disaster_recovery_primary.dr_kms_key_arn
  }
}

output "disaster_recovery_secondary" {
  description = "Secondary region disaster recovery configuration"
  value = {
    secondary_cluster_id        = module.disaster_recovery_secondary.secondary_cluster_id
    replica_s3_bucket_id       = module.disaster_recovery_secondary.replica_s3_bucket_id
    failover_lambda_function   = module.disaster_recovery_secondary.failover_lambda_function_name
    dr_secondary_kms_key_arn   = module.disaster_recovery_secondary.dr_secondary_kms_key_arn
  }
}

output "disaster_recovery_monitoring" {
  description = "Disaster recovery monitoring resources"
  value = {
    sns_topic_arn              = aws_sns_topic.production_alerts.arn
    sns_topic_secondary_arn    = aws_sns_topic.production_alerts_secondary.arn
    primary_health_check_id    = aws_route53_health_check.primary_region.id
    secondary_health_check_id  = aws_route53_health_check.secondary_region.id
    dashboard_url             = "https://console.aws.amazon.com/cloudwatch/home?region=${var.primary_region}#dashboards:name=${aws_cloudwatch_dashboard.disaster_recovery_overview.dashboard_name}"
  }
}