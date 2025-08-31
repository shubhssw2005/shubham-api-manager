output "tenant_role_arn" {
  description = "ARN of the tenant isolation IAM role"
  value       = aws_iam_role.tenant_role.arn
}

output "tenant_role_name" {
  description = "Name of the tenant isolation IAM role"
  value       = aws_iam_role.tenant_role.name
}

output "eks_service_account_role_arn" {
  description = "ARN of the EKS service account IAM role"
  value       = aws_iam_role.eks_service_account.arn
}

output "eks_service_account_role_name" {
  description = "Name of the EKS service account IAM role"
  value       = aws_iam_role.eks_service_account.name
}

output "api_gateway_role_arn" {
  description = "ARN of the API Gateway IAM role"
  value       = aws_iam_role.api_gateway.arn
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution IAM role"
  value       = aws_iam_role.lambda_execution.arn
}