# API Gateway Module for AWS Deployment System
# Provides JWT validation, rate limiting, and request transformation

resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.project_name}-${var.environment}-api"
  description = "API Gateway for ${var.project_name} with JWT validation and rate limiting"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = "*"
        Action = "execute-api:Invoke"
        Resource = "*"
        Condition = {
          IpAddress = {
            "aws:SourceIp" = var.allowed_ip_ranges
          }
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-api-gateway"
    Environment = var.environment
  }
}

# JWT Authorizer
resource "aws_api_gateway_authorizer" "jwt_authorizer" {
  name                   = "jwt-authorizer"
  rest_api_id           = aws_api_gateway_rest_api.main.id
  authorizer_uri        = aws_lambda_function.jwt_authorizer.invoke_arn
  authorizer_credentials = aws_iam_role.api_gateway_authorizer.arn
  type                  = "TOKEN"
  identity_source       = "method.request.header.Authorization"
  authorizer_result_ttl_in_seconds = 300

  depends_on = [aws_lambda_function.jwt_authorizer]
}

# Request Validator
resource "aws_api_gateway_request_validator" "main" {
  name                        = "request-validator"
  rest_api_id                = aws_api_gateway_rest_api.main.id
  validate_request_body      = true
  validate_request_parameters = true
}

# Gateway Response for 4xx errors
resource "aws_api_gateway_gateway_response" "response_4xx" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  response_type = "DEFAULT_4XX"

  response_templates = {
    "application/json" = jsonencode({
      error = {
        code    = "$context.error.responseType"
        message = "$context.error.message"
        requestId = "$context.requestId"
      }
    })
  }

  response_parameters = {
    "gatewayresponse.header.Access-Control-Allow-Origin"  = "'*'"
    "gatewayresponse.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "gatewayresponse.header.Access-Control-Allow-Methods" = "'GET,POST,PUT,DELETE,OPTIONS'"
  }
}

# Gateway Response for 5xx errors
resource "aws_api_gateway_gateway_response" "response_5xx" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  response_type = "DEFAULT_5XX"

  response_templates = {
    "application/json" = jsonencode({
      error = {
        code    = "INTERNAL_SERVER_ERROR"
        message = "An internal server error occurred"
        requestId = "$context.requestId"
      }
    })
  }

  response_parameters = {
    "gatewayresponse.header.Access-Control-Allow-Origin" = "'*'"
  }
}

# Usage Plan for Rate Limiting
resource "aws_api_gateway_usage_plan" "main" {
  name         = "${var.project_name}-usage-plan"
  description  = "Usage plan with rate limiting"

  api_stages {
    api_id = aws_api_gateway_rest_api.main.id
    stage  = aws_api_gateway_deployment.main.stage_name
  }

  quota_settings {
    limit  = var.monthly_quota_limit
    period = "MONTH"
  }

  throttle_settings {
    rate_limit  = var.rate_limit_per_second
    burst_limit = var.burst_limit
  }

  tags = {
    Environment = var.environment
  }
}

# API Key for usage plan
resource "aws_api_gateway_api_key" "main" {
  name        = "${var.project_name}-api-key"
  description = "API key for ${var.project_name}"
  enabled     = true

  tags = {
    Environment = var.environment
  }
}

# Usage Plan Key
resource "aws_api_gateway_usage_plan_key" "main" {
  key_id        = aws_api_gateway_api_key.main.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.main.id
}

# Deployment
resource "aws_api_gateway_deployment" "main" {
  depends_on = [
    aws_api_gateway_method.proxy,
    aws_api_gateway_integration.proxy
  ]

  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = var.environment

  variables = {
    deployed_at = timestamp()
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Stage configuration
resource "aws_api_gateway_stage" "main" {
  deployment_id = aws_api_gateway_deployment.main.id
  rest_api_id   = aws_api_gateway_rest_api.main.id
  stage_name    = var.environment

  # Enable logging
  access_log_destination_arn = aws_cloudwatch_log_group.api_gateway.arn
  access_log_format = jsonencode({
    requestId      = "$context.requestId"
    ip            = "$context.identity.sourceIp"
    caller        = "$context.identity.caller"
    user          = "$context.identity.user"
    requestTime   = "$context.requestTime"
    httpMethod    = "$context.httpMethod"
    resourcePath  = "$context.resourcePath"
    status        = "$context.status"
    protocol      = "$context.protocol"
    responseLength = "$context.responseLength"
    responseTime  = "$context.responseTime"
    error         = "$context.error.message"
    integrationError = "$context.integration.error"
  })

  # Enable X-Ray tracing
  xray_tracing_enabled = true

  # Method settings
  method_settings {
    method_path = "*/*"
    
    # Enable CloudWatch metrics
    metrics_enabled = true
    logging_level   = "INFO"
    
    # Enable data trace
    data_trace_enabled = true
    
    # Throttling settings
    throttling_rate_limit  = var.rate_limit_per_second
    throttling_burst_limit = var.burst_limit
  }

  tags = {
    Environment = var.environment
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days

  tags = {
    Environment = var.environment
  }
}# Proxy
 Resource for all paths
resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "{proxy+}"
}

# Proxy Method
resource "aws_api_gateway_method" "proxy" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "CUSTOM"
  authorizer_id = aws_api_gateway_authorizer.jwt_authorizer.id

  request_validator_id = aws_api_gateway_request_validator.main.id

  request_parameters = {
    "method.request.path.proxy"              = true
    "method.request.header.Authorization"    = true
    "method.request.header.X-Tenant-ID"     = false
    "method.request.header.Content-Type"     = false
  }
}

# Proxy Integration
resource "aws_api_gateway_integration" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.proxy.http_method

  integration_http_method = "ANY"
  type                   = "HTTP_PROXY"
  uri                    = "${var.backend_url}/{proxy}"

  request_parameters = {
    "integration.request.path.proxy"           = "method.request.path.proxy"
    "integration.request.header.X-Tenant-ID"  = "context.authorizer.tenantId"
    "integration.request.header.X-User-ID"    = "context.authorizer.userId"
    "integration.request.header.X-Request-ID" = "context.requestId"
  }

  # Request transformation
  request_templates = {
    "application/json" = jsonencode({
      body          = "$input.json('$')"
      headers       = "$input.params().header"
      queryParams   = "$input.params().querystring"
      pathParams    = "$input.params().path"
      requestId     = "$context.requestId"
      sourceIp      = "$context.identity.sourceIp"
      userAgent     = "$context.identity.userAgent"
      tenantId      = "$context.authorizer.tenantId"
      userId        = "$context.authorizer.userId"
    })
  }

  timeout_milliseconds = 29000
}

# CORS Options Method
resource "aws_api_gateway_method" "options" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

# CORS Options Integration
resource "aws_api_gateway_integration" "options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.options.http_method

  type = "MOCK"

  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# CORS Options Method Response
resource "aws_api_gateway_method_response" "options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }

  response_models = {
    "application/json" = "Empty"
  }
}

# CORS Options Integration Response
resource "aws_api_gateway_integration_response" "options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = aws_api_gateway_method_response.options.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Tenant-ID'"
    "method.response.header.Access-Control-Allow-Methods" = "'DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}