# ECR repositories with image scanning enabled
resource "aws_ecr_repository" "api_service" {
  name                 = "${var.project_name}/${var.environment}/api-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }

  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Keep last 10 production images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["prod"]
            countType     = "imageCountMoreThan"
            countNumber   = 10
          }
          action = {
            type = "expire"
          }
        },
        {
          rulePriority = 2
          description  = "Keep last 5 staging images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["staging"]
            countType     = "imageCountMoreThan"
            countNumber   = 5
          }
          action = {
            type = "expire"
          }
        },
        {
          rulePriority = 3
          description  = "Delete untagged images older than 1 day"
          selection = {
            tagStatus   = "untagged"
            countType   = "sinceImagePushed"
            countUnit   = "days"
            countNumber = 1
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-api-service"
    Environment = var.environment
  }
}

resource "aws_ecr_repository" "media_service" {
  name                 = "${var.project_name}/${var.environment}/media-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }

  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Keep last 10 production images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["prod"]
            countType     = "imageCountMoreThan"
            countNumber   = 10
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-media-service"
    Environment = var.environment
  }
}

resource "aws_ecr_repository" "worker_service" {
  name                 = "${var.project_name}/${var.environment}/worker-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }

  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Keep last 10 production images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["prod"]
            countType     = "imageCountMoreThan"
            countNumber   = 10
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker-service"
    Environment = var.environment
  }
}

# KMS key for ECR encryption
resource "aws_kms_key" "ecr" {
  description             = "KMS key for ECR encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow ECR to use the key"
        Effect = "Allow"
        Principal = {
          Service = "ecr.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-ecr-kms"
    Environment = var.environment
  }
}

resource "aws_kms_alias" "ecr" {
  name          = "alias/${var.project_name}-${var.environment}-ecr"
  target_key_id = aws_kms_key.ecr.key_id
}

data "aws_caller_identity" "current" {}

# EventBridge rule for ECR scan results
resource "aws_cloudwatch_event_rule" "ecr_scan_results" {
  name        = "${var.project_name}-${var.environment}-ecr-scan-results"
  description = "Capture ECR image scan results"

  event_pattern = jsonencode({
    source      = ["aws.ecr"]
    detail-type = ["ECR Image Scan"]
    detail = {
      scan-status = ["COMPLETE"]
    }
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-ecr-scan-results"
    Environment = var.environment
  }
}

# Lambda function to process scan results
resource "aws_lambda_function" "process_scan_results" {
  filename         = data.archive_file.scan_processor_zip.output_path
  function_name    = "${var.project_name}-${var.environment}-process-scan-results"
  role            = aws_iam_role.scan_processor.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.scan_processor_zip.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 300

  environment {
    variables = {
      SNS_TOPIC_ARN = var.sns_topic_arn
      SLACK_WEBHOOK = var.slack_webhook_url
    }
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-process-scan-results"
    Environment = var.environment
  }
}

data "archive_file" "scan_processor_zip" {
  type        = "zip"
  output_path = "/tmp/scan_processor.zip"
  source {
    content = templatefile("${path.module}/lambda/scan_processor.js", {
      project_name = var.project_name
      environment  = var.environment
    })
    filename = "index.js"
  }
}

# IAM role for scan processor Lambda
resource "aws_iam_role" "scan_processor" {
  name = "${var.project_name}-${var.environment}-scan-processor"

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
    Name        = "${var.project_name}-${var.environment}-scan-processor"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "scan_processor" {
  name = "${var.project_name}-${var.environment}-scan-processor"
  role = aws_iam_role.scan_processor.id

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
          "ecr:DescribeImageScanFindings",
          "ecr:DescribeImages",
          "ecr:DescribeRepositories"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.sns_topic_arn
      }
    ]
  })
}

# EventBridge target
resource "aws_cloudwatch_event_target" "scan_processor" {
  rule      = aws_cloudwatch_event_rule.ecr_scan_results.name
  target_id = "ScanProcessorTarget"
  arn       = aws_lambda_function.process_scan_results.arn
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.process_scan_results.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ecr_scan_results.arn
}

# Inspector V2 for runtime vulnerability assessment
resource "aws_inspector2_enabler" "ecr" {
  account_ids    = [data.aws_caller_identity.current.account_id]
  resource_types = ["ECR"]
}

resource "aws_inspector2_enabler" "ec2" {
  account_ids    = [data.aws_caller_identity.current.account_id]
  resource_types = ["EC2"]
}

# CloudWatch dashboard for security metrics
resource "aws_cloudwatch_dashboard" "security_dashboard" {
  dashboard_name = "${var.project_name}-${var.environment}-security"

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
            ["AWS/ECR", "RepositoryPullCount", "RepositoryName", aws_ecr_repository.api_service.name],
            ["AWS/ECR", "RepositoryPullCount", "RepositoryName", aws_ecr_repository.media_service.name],
            ["AWS/ECR", "RepositoryPullCount", "RepositoryName", aws_ecr_repository.worker_service.name]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECR Repository Pull Count"
          period  = 300
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 6
        width  = 24
        height = 6

        properties = {
          query   = "SOURCE '/aws/lambda/${aws_lambda_function.process_scan_results.function_name}' | fields @timestamp, @message | filter @message like /CRITICAL|HIGH/ | sort @timestamp desc | limit 100"
          region  = var.aws_region
          title   = "Critical and High Severity Vulnerabilities"
        }
      }
    ]
  })
}

# SNS topic for security alerts (if not provided)
resource "aws_sns_topic" "security_alerts" {
  count = var.sns_topic_arn == "" ? 1 : 0
  name  = "${var.project_name}-${var.environment}-security-alerts"

  kms_master_key_id = "alias/aws/sns"

  tags = {
    Name        = "${var.project_name}-${var.environment}-security-alerts"
    Environment = var.environment
  }
}

# CloudWatch alarms for security events
resource "aws_cloudwatch_metric_alarm" "high_severity_vulnerabilities" {
  alarm_name          = "${var.project_name}-${var.environment}-high-severity-vulnerabilities"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "HighSeverityVulnerabilities"
  namespace           = "Custom/Security"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "High severity vulnerabilities detected in container images"
  alarm_actions       = [var.sns_topic_arn != "" ? var.sns_topic_arn : aws_sns_topic.security_alerts[0].arn]

  tags = {
    Name        = "${var.project_name}-${var.environment}-high-severity-vulnerabilities"
    Environment = var.environment
  }
}