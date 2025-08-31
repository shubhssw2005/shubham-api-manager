# Enhanced Cost Monitoring and Alerting

# Cost and Usage Report
resource "aws_cur_report_definition" "cost_usage_report" {
  report_name                = "${var.project_name}-${var.environment}-cost-usage-report"
  time_unit                  = "DAILY"
  format                     = "Parquet"
  compression                = "GZIP"
  additional_schema_elements = ["RESOURCES"]
  s3_bucket                  = aws_s3_bucket.cost_reports.bucket
  s3_prefix                  = "cost-reports/"
  s3_region                  = data.aws_region.current.name
  additional_artifacts       = ["ATHENA", "REDSHIFT", "QUICKSIGHT"]
  refresh_closed_reports     = true
  report_versioning          = "OVERWRITE_REPORT"
}

# S3 bucket for cost reports
resource "aws_s3_bucket" "cost_reports" {
  bucket        = "${var.project_name}-${var.environment}-cost-reports-${random_id.cost_bucket_suffix.hex}"
  force_destroy = true

  tags = var.common_tags
}

resource "random_id" "cost_bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "cost_reports_versioning" {
  bucket = aws_s3_bucket.cost_reports.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cost_reports_encryption" {
  bucket = aws_s3_bucket.cost_reports.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cost_reports_lifecycle" {
  bucket = aws_s3_bucket.cost_reports.id

  rule {
    id     = "cost_reports_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Cost allocation tags
resource "aws_ce_cost_category" "service_cost_category" {
  name         = "${var.project_name}-${var.environment}-service-costs"
  rule_version = "CostCategoryExpression.v1"

  rule {
    value = "EKS"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Elastic Kubernetes Service", "Amazon Elastic Compute Cloud - Compute"]
        match_options = ["EQUALS"]
      }
    }
  }

  rule {
    value = "Storage"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Simple Storage Service"]
        match_options = ["EQUALS"]
      }
    }
  }

  rule {
    value = "Database"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Relational Database Service", "Amazon ElastiCache"]
        match_options = ["EQUALS"]
      }
    }
  }

  rule {
    value = "Networking"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon CloudFront", "Amazon Route 53", "Amazon Virtual Private Cloud"]
        match_options = ["EQUALS"]
      }
    }
  }

  tags = var.common_tags
}

# CloudWatch alarms for cost monitoring
resource "aws_cloudwatch_metric_alarm" "daily_cost_alarm" {
  alarm_name          = "${var.project_name}-${var.environment}-daily-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = var.daily_cost_threshold
  alarm_description   = "This metric monitors daily estimated charges"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency = "USD"
  }

  tags = var.common_tags
}

resource "aws_cloudwatch_metric_alarm" "s3_storage_cost_alarm" {
  alarm_name          = "${var.project_name}-${var.environment}-s3-storage-cost"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = var.s3_cost_threshold
  alarm_description   = "This metric monitors S3 storage costs"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency    = "USD"
    ServiceName = "AmazonS3"
  }

  tags = var.common_tags
}

# Cost optimization dashboard
resource "aws_cloudwatch_dashboard" "cost_optimization_dashboard" {
  dashboard_name = "${var.project_name}-${var.environment}-cost-optimization"

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
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD"],
            [".", ".", "Currency", "USD", "ServiceName", "AmazonEC2-Instance"],
            [".", ".", "Currency", "USD", "ServiceName", "AmazonS3"],
            [".", ".", "Currency", "USD", "ServiceName", "AmazonRDS"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "Daily Cost Breakdown by Service"
          period  = 86400
          yAxis = {
            left = {
              min = 0
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/S3", "BucketSizeBytes", "BucketName", "${var.project_name}-${var.environment}-media", "StorageType", "StandardStorage"],
            [".", ".", ".", ".", ".", "StandardIAStorage"],
            [".", ".", ".", ".", ".", "GlacierStorage"],
            [".", ".", ".", ".", ".", "DeepArchiveStorage"]
          ]
          view    = "timeSeries"
          stacked = true
          region  = data.aws_region.current.name
          title   = "S3 Storage Usage by Class"
          period  = 86400
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
        width  = 8
        height = 6

        properties = {
          metrics = [
            ["AWS/EKS", "cluster_node_count", "cluster_name", "${var.project_name}-${var.environment}"],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "${var.project_name}-${var.environment}-alb"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "EKS Cluster Metrics"
          period  = 300
        }
      },
      {
        type   = "log"
        x      = 8
        y      = 6
        width  = 16
        height = 6

        properties = {
          query   = "SOURCE '/aws/lambda/${var.project_name}-${var.environment}-cost-optimizer'\n| fields @timestamp, @message\n| filter @message like /recommendations/ or @message like /potential_savings/\n| sort @timestamp desc\n| limit 20"
          region  = data.aws_region.current.name
          title   = "C++ Cost Optimizer Recommendations"
          view    = "table"
        }
      }
    ]
  })
}

# EventBridge rule for weekly cost analysis (using the C++ cost optimizer)
resource "aws_cloudwatch_event_rule" "weekly_cost_analysis" {
  name                = "${var.project_name}-${var.environment}-weekly-cost-analysis"
  description         = "Trigger weekly cost analysis using C++ optimizer"
  schedule_expression = "cron(0 9 ? * MON *)" # 9 AM UTC every Monday

  tags = var.common_tags
}

resource "aws_cloudwatch_event_target" "weekly_cost_analysis_target" {
  rule      = aws_cloudwatch_event_rule.weekly_cost_analysis.name
  target_id = "WeeklyCostAnalysisTarget"
  arn       = aws_lambda_function.cost_optimizer.arn
}

resource "aws_lambda_permission" "allow_eventbridge_weekly" {
  statement_id  = "AllowExecutionFromEventBridgeWeekly"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_optimizer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_cost_analysis.arn
}

# Data sources
data "aws_region" "current" {}

# Variables for cost thresholds
variable "daily_cost_threshold" {
  description = "Daily cost threshold for alerts"
  type        = number
  default     = 100
}

variable "s3_cost_threshold" {
  description = "S3 cost threshold for alerts"
  type        = number
  default     = 50
}