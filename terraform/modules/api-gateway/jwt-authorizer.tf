# JWT Authorizer Lambda Function
resource "aws_lambda_function" "jwt_authorizer" {
  filename         = data.archive_file.jwt_authorizer.output_path
  function_name    = "${var.project_name}-${var.environment}-jwt-authorizer"
  role            = aws_iam_role.jwt_authorizer_lambda.arn
  handler         = "index.handler"
  runtime         = "nodejs18.x"
  timeout         = 30
  memory_size     = 256

  source_code_hash = data.archive_file.jwt_authorizer.output_base64sha256

  environment {
    variables = {
      JWT_SECRET           = var.jwt_secret
      REDIS_CLUSTER_ENDPOINT = var.redis_cluster_endpoint
      RATE_LIMIT_WINDOW   = var.rate_limit_window
      LOG_LEVEL           = var.log_level
    }
  }

  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.jwt_authorizer_lambda.id]
  }

  tags = {
    Environment = var.environment
  }
}

# Lambda function code archive
data "archive_file" "jwt_authorizer" {
  type        = "zip"
  output_path = "${path.module}/jwt-authorizer.zip"
  source_dir  = "${path.module}/lambda/jwt-authorizer"
}

# IAM Role for JWT Authorizer Lambda
resource "aws_iam_role" "jwt_authorizer_lambda" {
  name = "${var.project_name}-${var.environment}-jwt-authorizer-lambda"

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
    Environment = var.environment
  }
}

# IAM Policy for JWT Authorizer Lambda
resource "aws_iam_role_policy" "jwt_authorizer_lambda" {
  name = "${var.project_name}-${var.environment}-jwt-authorizer-lambda"
  role = aws_iam_role.jwt_authorizer_lambda.id

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
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "elasticache:DescribeCacheClusters",
          "elasticache:DescribeReplicationGroups"
        ]
        Resource = "*"
      }
    ]
  })
}

# Security Group for JWT Authorizer Lambda
resource "aws_security_group" "jwt_authorizer_lambda" {
  name_prefix = "${var.project_name}-${var.environment}-jwt-authorizer-"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow access to Redis
  egress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-jwt-authorizer-lambda"
    Environment = var.environment
  }
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway_jwt_authorizer" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.jwt_authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}

# IAM Role for API Gateway to invoke authorizer
resource "aws_iam_role" "api_gateway_authorizer" {
  name = "${var.project_name}-${var.environment}-api-gateway-authorizer"

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

  tags = {
    Environment = var.environment
  }
}

# IAM Policy for API Gateway authorizer role
resource "aws_iam_role_policy" "api_gateway_authorizer" {
  name = "${var.project_name}-${var.environment}-api-gateway-authorizer"
  role = aws_iam_role.api_gateway_authorizer.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = aws_lambda_function.jwt_authorizer.arn
      }
    ]
  })
}