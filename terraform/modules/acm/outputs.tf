# ACM Module Outputs

output "certificate_arn" {
  description = "ARN of the main certificate"
  value       = aws_acm_certificate.main.arn
}

output "certificate_domain_name" {
  description = "Domain name of the main certificate"
  value       = aws_acm_certificate.main.domain_name
}

output "certificate_status" {
  description = "Status of the main certificate"
  value       = aws_acm_certificate.main.status
}

output "wildcard_certificate_arn" {
  description = "ARN of the wildcard certificate"
  value       = var.create_wildcard_cert ? aws_acm_certificate.wildcard[0].arn : null
}

output "wildcard_certificate_domain_name" {
  description = "Domain name of the wildcard certificate"
  value       = var.create_wildcard_cert ? aws_acm_certificate.wildcard[0].domain_name : null
}

output "certificate_validation_records" {
  description = "Certificate validation DNS records"
  value = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }
}

output "wildcard_certificate_validation_records" {
  description = "Wildcard certificate validation DNS records"
  value = var.create_wildcard_cert ? {
    for dvo in aws_acm_certificate.wildcard[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}
}

output "certificate_expiry_alarm_arn" {
  description = "ARN of the certificate expiry alarm"
  value       = var.enable_expiry_monitoring ? aws_cloudwatch_metric_alarm.cert_expiry[0].arn : null
}

output "wildcard_certificate_expiry_alarm_arn" {
  description = "ARN of the wildcard certificate expiry alarm"
  value       = var.create_wildcard_cert && var.enable_expiry_monitoring ? aws_cloudwatch_metric_alarm.wildcard_cert_expiry[0].arn : null
}