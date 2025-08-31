# Route53 Module Outputs

output "main_record_name" {
  description = "Name of the main A record"
  value       = aws_route53_record.main.name
}

output "main_record_fqdn" {
  description = "FQDN of the main A record"
  value       = aws_route53_record.main.fqdn
}

output "www_record_name" {
  description = "Name of the www A record"
  value       = var.create_www_redirect ? aws_route53_record.www[0].name : null
}

output "www_record_fqdn" {
  description = "FQDN of the www A record"
  value       = var.create_www_redirect ? aws_route53_record.www[0].fqdn : null
}

output "subdomain_records" {
  description = "Map of subdomain record names and FQDNs"
  value = {
    for k, v in aws_route53_record.subdomains : k => {
      name = v.name
      fqdn = v.fqdn
    }
  }
}

output "api_record_name" {
  description = "Name of the API subdomain record"
  value       = var.api_subdomain_enabled ? aws_route53_record.api[0].name : null
}

output "api_record_fqdn" {
  description = "FQDN of the API subdomain record"
  value       = var.api_subdomain_enabled ? aws_route53_record.api[0].fqdn : null
}

output "media_record_name" {
  description = "Name of the media subdomain record"
  value       = var.media_subdomain_enabled ? aws_route53_record.media[0].name : null
}

output "media_record_fqdn" {
  description = "FQDN of the media subdomain record"
  value       = var.media_subdomain_enabled ? aws_route53_record.media[0].fqdn : null
}

output "cdn_record_name" {
  description = "Name of the CDN subdomain record"
  value       = var.cdn_subdomain_enabled ? aws_route53_record.cdn[0].name : null
}

output "cdn_record_fqdn" {
  description = "FQDN of the CDN subdomain record"
  value       = var.cdn_subdomain_enabled ? aws_route53_record.cdn[0].fqdn : null
}

output "health_check_id" {
  description = "ID of the Route53 health check"
  value       = var.enable_health_check ? aws_route53_health_check.main[0].id : null
}

output "health_check_alarm_arn" {
  description = "ARN of the health check CloudWatch alarm"
  value       = var.enable_health_check ? aws_cloudwatch_metric_alarm.health_check[0].arn : null
}

output "domain_verification_records" {
  description = "Map of domain verification record names and FQDNs"
  value = {
    for k, v in aws_route53_record.domain_verification : k => {
      name = v.name
      fqdn = v.fqdn
    }
  }
}