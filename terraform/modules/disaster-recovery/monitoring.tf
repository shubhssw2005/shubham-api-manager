# Disaster Recovery Monitoring and Alerting

# SNS topic for DR alerts
resource "aws_sns_topic" "dr_alerts" {
  name = "${var.project_name}-dr-alerts"

  tags = {
    Name        = "${var.project_name}-dr-alerts"
    Environment = var.environment
    Purpose     = "disaster-recovery-alerts"
  }
}

# SNS topic subscription for email alerts
resource "aws_sns_topic_subscription" "dr_email_alerts" {
  count     = length(var.alert_email_addresses)
  topic_arn = aws_sns_topic.dr_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}

# CloudWatch alarm for Aurora Global Database replication lag
resource "aws_cloudwatch_metric_alarm" "aurora_replication_lag" {
  count               = var.enable_global_database ? 1 : 0
  alarm_name          = "${var.project_name}-aurora-replication-lag"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "AuroraGlobalDBReplicationLag"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "300000"  # 5 minutes in milliseconds
  alarm_description   = "Aurora Global Database replication lag exceeded 5 minutes"
  alarm_actions       = [aws_sns_topic.dr_alerts.arn]
  ok_actions          = [aws_sns_topic.dr_alerts.arn]
  treat_missing_data  = "breaching"

  dimensions = {
    SourceRegion = data.aws_region.current.name
    TargetRegion = var.dr_region
  }

  tags = {
    Name        = "${var.project_name}-aurora-replication-lag"
    Environment = var.environment
    Severity    = "critical"
  }
}

# CloudWatch alarm for Aurora cluster availability
resource "aws_cloudwatch_metric_alarm" "aurora_cluster_availability" {
  count               = var.enable_global_database ? 1 : 0
  alarm_name          = "${var.project_name}-aurora-cluster-unavailable"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "60"
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "Aurora cluster appears to be unavailable"
  alarm_actions       = [aws_sns_topic.dr_alerts.arn, aws_lambda_function.automated_failover[0].arn]
  treat_missing_data  = "breaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.primary[0].id
  }

  tags = {
    Name        = "${var.project_name}-aurora-cluster-unavailable"
    Environment = var.environment
    Severity    = "critical"
  }
}

# CloudWatch alarm for high error rate (potential trigger for failover)
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "5XXError"
  namespace           = "AWS/ApplicationELB"
  period              = "300"
  statistic           = "Sum"
  threshold           = "50"
  alarm_description   = "High error rate detected - potential failover trigger"
  alarm_actions       = [aws_sns_topic.dr_alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = var.load_balancer_arn_suffix
  }

  tags = {
    Name        = "${var.project_name}-high-error-rate"
    Environment = var.environment
    Severity    = "warning"
  }
}

# CloudWatch alarm for S3 replication metrics
resource "aws_cloudwatch_metric_alarm" "s3_replication_failure" {
  count               = var.enable_s3_replication ? 1 : 0
  alarm_name          = "${var.project_name}-s3-replication-failure"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ReplicationLatency"
  namespace           = "AWS/S3"
  period              = "900"  # 15 minutes
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "S3 cross-region replication is failing or delayed"
  alarm_actions       = [aws_sns_topic.dr_alerts.arn]
  treat_missing_data  = "breaching"

  dimensions = {
    SourceBucket      = aws_s3_bucket.primary.bucket
    DestinationBucket = var.replica_bucket_name
  }

  tags = {
    Name        = "${var.project_name}-s3-replication-failure"
    Environment = var.environment
    Severity    = "warning"
  }
}

# Lambda function for automated failover
resource "aws_lambda_function" "automated_failover" {
  count            = var.enable_automated_failover ? 1 : 0
  filename         = "automated_failover.zip"
  function_name    = "${var.project_name}-automated-failover"
  role            = aws_iam_role.lambda_automated_failover[0].arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.automated_failover_zip[0].output_base64sha256
  runtime         = "python3.11"
  timeout         = 300

  environment {
    variables = {
      GLOBAL_CLUSTER_ID = var.enable_global_database ? aws_rds_global_cluster.main[0].id : ""
      PRIMARY_CLUSTER_ID = var.enable_global_database ? aws_rds_cluster.primary[0].id : ""
      SNS_TOPIC_ARN = aws_sns_topic.dr_alerts.arn
      REGION = data.aws_region.current.name
      DR_REGION = var.dr_region
      FAILOVER_THRESHOLD_MINUTES = "5"
    }
  }

  tags = {
    Name        = "${var.project_name}-automated-failover"
    Environment = var.environment
  }
}

# Lambda deployment package for automated failover
data "archive_file" "automated_failover_zip" {
  count       = var.enable_automated_failover ? 1 : 0
  type        = "zip"
  output_path = "automated_failover.zip"
  source {
    content = templatefile("${path.module}/lambda/automated_failover.py", {
      project_name = var.project_name
    })
    filename = "index.py"
  }
}

# IAM role for automated failover Lambda
resource "aws_iam_role" "lambda_automated_failover" {
  count = var.enable_automated_failover ? 1 : 0
  name  = "${var.project_name}-lambda-automated-failover-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-lambda-automated-failover-role"
    Environment = var.environment
  }
}

# IAM policy for automated failover Lambda
resource "aws_iam_role_policy" "lambda_automated_failover" {
  count = var.enable_automated_failover ? 1 : 0
  name  = "${var.project_name}-lambda-automated-failover-policy"
  role  = aws_iam_role.lambda_automated_failover[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeGlobalClusters",
          "rds:DescribeDBClusters",
          "rds:DescribeDBClusterSnapshots",
          "rds:CreateDBClusterSnapshot"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.dr_alerts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_automated_failover_basic" {
  count      = var.enable_automated_failover ? 1 : 0
  role       = aws_iam_role.lambda_automated_failover[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda permission for CloudWatch alarms to invoke the function
resource "aws_lambda_permission" "allow_cloudwatch_invoke" {
  count         = var.enable_automated_failover ? 1 : 0
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.automated_failover[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_metric_alarm.aurora_cluster_availability[0].arn
}

# CloudWatch dashboard for disaster recovery monitoring
resource "aws_cloudwatch_dashboard" "disaster_recovery" {
  dashboard_name = "${var.project_name}-disaster-recovery"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "AuroraGlobalDBReplicationLag", "SourceRegion", data.aws_region.current.name, "TargetRegion", var.dr_region],
            ["AWS/RDS", "DatabaseConnections", "DBClusterIdentifier", var.enable_global_database ? aws_rds_cluster.primary[0].id : ""],
            ["AWS/S3", "ReplicationLatency", "SourceBucket", aws_s3_bucket.primary.bucket, "DestinationBucket", var.replica_bucket_name]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Disaster Recovery Metrics"
          period  = 300
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
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", var.load_balancer_arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", var.load_balancer_arn_suffix],
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", var.load_balancer_arn_suffix]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Application Health Metrics"
          period  = 300
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-disaster-recovery-dashboard"
    Environment = var.environment
  }
}

# EventBridge rule for scheduled DR health checks
resource "aws_cloudwatch_event_rule" "dr_health_check" {
  name                = "${var.project_name}-dr-health-check"
  description         = "Scheduled disaster recovery health check"
  schedule_expression = "rate(5 minutes)"

  tags = {
    Name        = "${var.project_name}-dr-health-check"
    Environment = var.environment
  }
}

# EventBridge target for DR health check
resource "aws_cloudwatch_event_target" "dr_health_check_target" {
  count     = var.enable_automated_failover ? 1 : 0
  rule      = aws_cloudwatch_event_rule.dr_health_check.name
  target_id = "DRHealthCheckTarget"
  arn       = aws_lambda_function.automated_failover[0].arn

  input = jsonencode({
    source = "scheduled_health_check"
    check_type = "health_monitoring"
  })
}

# Lambda permission for EventBridge to invoke health check
resource "aws_lambda_permission" "allow_eventbridge_invoke" {
  count         = var.enable_automated_failover ? 1 : 0
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.automated_failover[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dr_health_check.arn
}