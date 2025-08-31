# IAM Role for Tenant Isolation
resource "aws_iam_role" "tenant_role" {
  name = "${var.project_name}-${var.environment}-tenant-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Condition = {
          StringEquals = {
            "sts:ExternalId" = "$${aws:userid}"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for Tenant S3 Access
resource "aws_iam_policy" "tenant_s3_access" {
  name        = "${var.project_name}-${var.environment}-tenant-s3-policy"
  description = "Policy for tenant-specific S3 access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion",
          "s3:PutObjectAcl",
          "s3:GetObjectAcl"
        ]
        Resource = [
          for bucket_arn in var.s3_bucket_arns : "${bucket_arn}/tenants/$${aws:userid}/*"
        ]
        Condition = {
          StringEquals = {
            "s3:x-amz-metadata-tenant-id" = "$${aws:userid}"
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = var.s3_bucket_arns
        Condition = {
          StringLike = {
            "s3:prefix" = "tenants/$${aws:userid}/*"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# Attach S3 policy to tenant role
resource "aws_iam_role_policy_attachment" "tenant_s3_access" {
  role       = aws_iam_role.tenant_role.name
  policy_arn = aws_iam_policy.tenant_s3_access.arn
}

# IAM Role for EKS Service Accounts (IRSA)
resource "aws_iam_role" "eks_service_account" {
  name = "${var.project_name}-${var.environment}-eks-service-account-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Condition = {
          StringEquals = {
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:default:${var.project_name}-service-account"
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for EKS Service Account
resource "aws_iam_policy" "eks_service_account" {
  name        = "${var.project_name}-${var.environment}-eks-service-account-policy"
  description = "Policy for EKS service account"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = concat(
          var.s3_bucket_arns,
          [for bucket_arn in var.s3_bucket_arns : "${bucket_arn}/*"]
        )
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/${var.environment}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = [
              "s3.${data.aws_region.current.name}.amazonaws.com",
              "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = "arn:aws:sns:*:${data.aws_caller_identity.current.account_id}:${var.project_name}-${var.environment}-*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = "arn:aws:sqs:*:${data.aws_caller_identity.current.account_id}:${var.project_name}-${var.environment}-*"
      }
    ]
  })

  tags = var.tags
}

# Attach policy to EKS service account role
resource "aws_iam_role_policy_attachment" "eks_service_account" {
  role       = aws_iam_role.eks_service_account.name
  policy_arn = aws_iam_policy.eks_service_account.arn
}

# IAM Role for API Gateway
resource "aws_iam_role" "api_gateway" {
  name = "${var.project_name}-${var.environment}-api-gateway-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for API Gateway CloudWatch Logs
resource "aws_iam_policy" "api_gateway_logs" {
  name        = "${var.project_name}-${var.environment}-api-gateway-logs-policy"
  description = "Policy for API Gateway CloudWatch logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

# Attach logs policy to API Gateway role
resource "aws_iam_role_policy_attachment" "api_gateway_logs" {
  role       = aws_iam_role.api_gateway.name
  policy_arn = aws_iam_policy.api_gateway_logs.arn
}

# IAM Role for Lambda Functions
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-${var.environment}-lambda-execution-role"

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

  tags = var.tags
}

# IAM Policy for Lambda Functions
resource "aws_iam_policy" "lambda_execution" {
  name        = "${var.project_name}-${var.environment}-lambda-execution-policy"
  description = "Policy for Lambda function execution"

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
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          for bucket_arn in var.s3_bucket_arns : "${bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = "arn:aws:sqs:*:${data.aws_caller_identity.current.account_id}:${var.project_name}-${var.environment}-*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

# Attach policy to Lambda execution role
resource "aws_iam_role_policy_attachment" "lambda_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = aws_iam_policy.lambda_execution.arn
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_eks_cluster" "main" {
  name = replace(var.eks_cluster_arn, "/.*cluster\\/([^/]+)$/", "$1")
}