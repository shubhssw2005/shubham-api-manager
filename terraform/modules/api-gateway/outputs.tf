# API Gateway Module Outputs

output "api_gateway_id" {
  description = "ID of the API Gateway"
  value       = aws_api_gateway_rest_api.main.id
}

output "api_gateway_arn" {
  description = "ARN of the API Gateway"
  value       = aws_api_gateway_rest_api.main.arn
}

output "api_gateway_execution_arn" {
  description = "Execution ARN of the API Gateway"
  value       = aws_api_gateway_rest_api.main.execution_arn
}

output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = "https://${aws_api_gateway_rest_api.main.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${var.environment}"
}

output "api_gateway_stage_name" {
  description = "Stage name of the API Gateway deployment"
  value       = aws_api_gateway_stage.main.stage_name
}

output "api_gateway_deployment_id" {
  description = "ID of the API Gateway deployment"
  value       = aws_api_gateway_deployment.main.id
}

output "jwt_authorizer_function_name" {
  description = "Name of the JWT authorizer Lambda function"
  value       = aws_lambda_function.jwt_authorizer.function_name
}

output "jwt_authorizer_function_arn" {
  description = "ARN of the JWT authorizer Lambda function"
  value       = aws_lambda_function.jwt_authorizer.arn
}

output "usage_plan_id" {
  description = "ID of the API Gateway usage plan"
  value       = aws_api_gateway_usage_plan.main.id
}

output "api_key_id" {
  description = "ID of the API Gateway API key"
  value       = aws_api_gateway_api_key.main.id
}

output "api_key_value" {
  description = "Value of the API Gateway API key"
  value       = aws_api_gateway_api_key.main.value
  sensitive   = true
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for API Gateway"
  value       = aws_cloudwatch_log_group.api_gateway.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for API Gateway"
  value       = aws_cloudwatch_log_group.api_gateway.arn
}

# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}