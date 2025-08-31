output "ecr_repositories" {
  description = "ECR repository information"
  value = {
    api_service = {
      name = aws_ecr_repository.api_service.name
      url  = aws_ecr_repository.api_service.repository_url
      arn  = aws_ecr_repository.api_service.arn
    }
    media_service = {
      name = aws_ecr_repository.media_service.name
      url  = aws_ecr_repository.media_service.repository_url
      arn  = aws_ecr_repository.media_service.arn
    }
    worker_service = {
      name = aws_ecr_repository.worker_service.name
      url  = aws_ecr_repository.worker_service.repository_url
      arn  = aws_ecr_repository.worker_service.arn
    }
  }
}

output "kms_key_arn" {
  description = "The ARN of the KMS key used for ECR encryption"
  value       = aws_kms_key.ecr.arn
}

output "scan_processor_lambda_arn" {
  description = "The ARN of the scan processor Lambda function"
  value       = aws_lambda_function.process_scan_results.arn
}

output "security_dashboard_url" {
  description = "URL to the CloudWatch security dashboard"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.security_dashboard.dashboard_name}"
}

output "inspector_enabler_status" {
  description = "Status of Inspector V2 enablers"
  value = {
    ecr = aws_inspector2_enabler.ecr.id
    ec2 = aws_inspector2_enabler.ec2.id
  }
}