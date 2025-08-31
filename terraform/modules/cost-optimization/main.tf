# Cost Optimization Module
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "notification_email" {
  description = "Email for cost alerts"
  type        = string
}

variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 10000
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {}
}

# Cost Budget with alerts
resource "aws_budgets_budget" "monthly_cost_budget" {
  name         = "${var.project_name}-${var.environment}-monthly-budget"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filters = {
    Tag = [
      "Project:${var.project_name}",
      "Environment:${var.environment}"
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 120
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }
}

# Service-specific budgets
resource "aws_budgets_budget" "eks_budget" {
  name         = "${var.project_name}-${var.environment}-eks-budget"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_limit * 0.4 # 40% of total budget
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filters = {
    Service = ["Amazon Elastic Kubernetes Service", "Amazon Elastic Compute Cloud - Compute"]
    Tag = [
      "Project:${var.project_name}",
      "Environment:${var.environment}"
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 85
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }
}

resource "aws_budgets_budget" "s3_budget" {
  name         = "${var.project_name}-${var.environment}-s3-budget"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_limit * 0.2 # 20% of total budget
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filters = {
    Service = ["Amazon Simple Storage Service"]
    Tag = [
      "Project:${var.project_name}",
      "Environment:${var.environment}"
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 85
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }
}

# Cost anomaly detection
resource "aws_ce_anomaly_detector" "service_anomaly_detector" {
  name         = "${var.project_name}-${var.environment}-service-anomaly"
  detector_type = "DIMENSIONAL"

  specification = jsonencode({
    Dimension = "SERVICE"
    MatchOptions = ["EQUALS"]
    Values = [
      "Amazon Elastic Kubernetes Service",
      "Amazon Simple Storage Service",
      "Amazon Relational Database Service",
      "Amazon ElastiCache"
    ]
  })

  tags = var.common_tags
}

resource "aws_ce_anomaly_subscription" "anomaly_subscription" {
  name      = "${var.project_name}-${var.environment}-anomaly-alerts"
  frequency = "DAILY"
  
  monitor_arn_list = [
    aws_ce_anomaly_detector.service_anomaly_detector.arn
  ]
  
  subscriber {
    type    = "EMAIL"
    address = var.notification_email
  }

  threshold_expression {
    and {
      dimension {
        key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
        values        = ["100"]
        match_options = ["GREATER_THAN_OR_EQUAL"]
      }
    }
  }

  tags = var.common_tags
}

# Cost optimization Lambda function
resource "aws_iam_role" "cost_optimizer_role" {
  name = "${var.project_name}-${var.environment}-cost-optimizer"

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

  tags = var.common_tags
}

resource "aws_iam_role_policy" "cost_optimizer_policy" {
  name = "${var.project_name}-${var.environment}-cost-optimizer-policy"
  role = aws_iam_role.cost_optimizer_role.id

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
          "ce:GetCostAndUsage",
          "ce:GetUsageReport",
          "ce:GetReservationCoverage",
          "ce:GetReservationPurchaseRecommendation",
          "ce:GetReservationUtilization",
          "ce:GetSavingsPlansUtilization",
          "ce:GetSavingsPlansUtilizationDetails",
          "ce:ListCostCategoryDefinitions",
          "ce:GetRightsizingRecommendation"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:GetBucketNotification",
          "s3:GetBucketTagging",
          "s3:GetLifecycleConfiguration",
          "s3:GetBucketAnalyticsConfiguration",
          "s3:GetIntelligentTieringConfiguration"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeReservedInstances",
          "ec2:DescribeSpotInstanceRequests"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:DescribeReservedDBInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.cost_alerts.arn
      }
    ]
  })
}

# SNS topic for cost alerts
resource "aws_sns_topic" "cost_alerts" {
  name = "${var.project_name}-${var.environment}-cost-alerts"
  
  tags = var.common_tags
}

resource "aws_sns_topic_subscription" "cost_alerts_email" {
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

# Build the cost optimizer binary
resource "null_resource" "build_cost_optimizer" {
  triggers = {
    source_hash = filemd5("${path.module}/cost_optimizer.cpp")
  }

  provisioner "local-exec" {
    command = "cd ${path.module} && chmod +x build.sh && ./build.sh"
  }
}

# Lambda function for cost optimization recommendations
resource "aws_lambda_function" "cost_optimizer" {
  filename         = "${path.module}/cost_optimizer.zip"
  function_name    = "${var.project_name}-${var.environment}-cost-optimizer"
  role            = aws_iam_role.cost_optimizer_role.arn
  handler         = "bootstrap"
  source_code_hash = filebase64sha256("${path.module}/cost_optimizer.zip")
  runtime         = "provided.al2"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      SNS_TOPIC_ARN = aws_sns_topic.cost_alerts.arn
      PROJECT_NAME  = var.project_name
      ENVIRONMENT   = var.environment
    }
  }

  depends_on = [null_resource.build_cost_optimizer]
  tags = var.common_tags
}

# EventBridge rule to run cost optimizer daily
resource "aws_cloudwatch_event_rule" "cost_optimizer_schedule" {
  name                = "${var.project_name}-${var.environment}-cost-optimizer-schedule"
  description         = "Trigger cost optimizer daily"
  schedule_expression = "cron(0 8 * * ? *)" # 8 AM UTC daily

  tags = var.common_tags
}

resource "aws_cloudwatch_event_target" "cost_optimizer_target" {
  rule      = aws_cloudwatch_event_rule.cost_optimizer_schedule.name
  target_id = "CostOptimizerTarget"
  arn       = aws_lambda_function.cost_optimizer.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_optimizer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_optimizer_schedule.arn
}

# CloudWatch dashboard for cost monitoring
resource "aws_cloudwatch_dashboard" "cost_dashboard" {
  dashboard_name = "${var.project_name}-${var.environment}-cost-monitoring"

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
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "Estimated Monthly Charges"
          period  = 86400
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
            [".", ".", ".", ".", ".", "GlacierStorage"]
          ]
          view    = "timeSeries"
          stacked = true
          region  = "us-east-1"
          title   = "S3 Storage Usage by Class"
          period  = 86400
        }
      }
    ]
  })
}

# Outputs
output "budget_arn" {
  description = "ARN of the monthly budget"
  value       = aws_budgets_budget.monthly_cost_budget.arn
}

output "cost_alerts_topic_arn" {
  description = "ARN of the cost alerts SNS topic"
  value       = aws_sns_topic.cost_alerts.arn
}

output "cost_optimizer_function_arn" {
  description = "ARN of the cost optimizer Lambda function"
  value       = aws_lambda_function.cost_optimizer.arn
}