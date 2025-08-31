# KMS key for Secrets Manager encryption
resource "aws_kms_key" "secrets" {
  description             = "KMS key for Secrets Manager encryption"
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
        Sid    = "Allow Secrets Manager to use the key"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:Encrypt",
          "kms:GenerateDataKey*",
          "kms:ReEncrypt*"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-secrets-kms"
    Environment = var.environment
  }
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.project_name}-${var.environment}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

data "aws_caller_identity" "current" {}

# Database master password
resource "aws_secretsmanager_secret" "db_master_password" {
  name                    = "${var.project_name}-${var.environment}-db-master-password"
  description             = "Master password for Aurora database"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 7

  replica {
    region     = var.replica_region
    kms_key_id = aws_kms_key.secrets.arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-master-password"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "db_master_password" {
  secret_id = aws_secretsmanager_secret.db_master_password.id
  secret_string = jsonencode({
    username = var.db_username
    password = var.db_password
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# JWT signing keys
resource "aws_secretsmanager_secret" "jwt_keys" {
  name                    = "${var.project_name}-${var.environment}-jwt-keys"
  description             = "JWT signing keys for authentication"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 7

  replica {
    region     = var.replica_region
    kms_key_id = aws_kms_key.secrets.arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-jwt-keys"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "jwt_keys" {
  secret_id = aws_secretsmanager_secret.jwt_keys.id
  secret_string = jsonencode({
    access_token_secret  = var.jwt_access_secret
    refresh_token_secret = var.jwt_refresh_secret
    encryption_key       = var.jwt_encryption_key
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# API keys and external service credentials
resource "aws_secretsmanager_secret" "api_keys" {
  name                    = "${var.project_name}-${var.environment}-api-keys"
  description             = "External API keys and service credentials"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 7

  replica {
    region     = var.replica_region
    kms_key_id = aws_kms_key.secrets.arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-api-keys"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    stripe_secret_key     = var.stripe_secret_key
    sendgrid_api_key      = var.sendgrid_api_key
    cloudflare_api_token  = var.cloudflare_api_token
    github_webhook_secret = var.github_webhook_secret
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# Lambda function for automatic rotation
resource "aws_lambda_function" "rotate_secrets" {
  filename         = data.archive_file.rotate_secrets_zip.output_path
  function_name    = "${var.project_name}-${var.environment}-rotate-secrets"
  role            = aws_iam_role.lambda_rotation.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.rotate_secrets_zip.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 300

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${var.aws_region}.amazonaws.com"
    }
  }

  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda_rotation.id]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-rotate-secrets"
    Environment = var.environment
  }
}

# Lambda deployment package
data "archive_file" "rotate_secrets_zip" {
  type        = "zip"
  output_path = "/tmp/rotate_secrets.zip"
  source {
    content = templatefile("${path.module}/lambda/rotate_secrets.js", {
      project_name = var.project_name
      environment  = var.environment
    })
    filename = "index.js"
  }
}

# IAM role for Lambda rotation function
resource "aws_iam_role" "lambda_rotation" {
  name = "${var.project_name}-${var.environment}-lambda-rotation"

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
    Name        = "${var.project_name}-${var.environment}-lambda-rotation"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "lambda_rotation" {
  name = "${var.project_name}-${var.environment}-lambda-rotation"
  role = aws_iam_role.lambda_rotation.id

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
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage"
        ]
        Resource = [
          aws_secretsmanager_secret.jwt_keys.arn,
          aws_secretsmanager_secret.api_keys.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.secrets.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# Security group for Lambda function
resource "aws_security_group" "lambda_rotation" {
  name_prefix = "${var.project_name}-${var.environment}-lambda-rotation"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-lambda-rotation"
    Environment = var.environment
  }
}

# Lambda permission for Secrets Manager
resource "aws_lambda_permission" "allow_secrets_manager" {
  statement_id  = "AllowExecutionFromSecretsManager"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rotate_secrets.function_name
  principal     = "secretsmanager.amazonaws.com"
}

# Automatic rotation configuration for JWT keys
resource "aws_secretsmanager_secret_rotation" "jwt_keys" {
  secret_id           = aws_secretsmanager_secret.jwt_keys.id
  lambda_arn          = aws_lambda_function.rotate_secrets.arn
  rotation_interval   = 30 # days

  depends_on = [aws_lambda_permission.allow_secrets_manager]
}

# CloudWatch alarms for secret access
resource "aws_cloudwatch_metric_alarm" "secret_access_anomaly" {
  alarm_name          = "${var.project_name}-${var.environment}-secret-access-anomaly"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "GetSecretValue"
  namespace           = "AWS/SecretsManager"
  period              = "300"
  statistic           = "Sum"
  threshold           = "100"
  alarm_description   = "Unusual secret access pattern detected"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    SecretName = aws_secretsmanager_secret.jwt_keys.name
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-secret-access-anomaly"
    Environment = var.environment
  }
}