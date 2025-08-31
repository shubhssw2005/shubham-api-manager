# CloudFront Configuration Outputs

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = module.cloudfront.distribution_id
}

output "cloudfront_distribution_arn" {
  description = "ARN of the CloudFront distribution"
  value       = module.cloudfront.distribution_arn
}

output "cloudfront_distribution_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = module.cloudfront.distribution_domain_name
}

output "cloudfront_distribution_hosted_zone_id" {
  description = "Hosted zone ID of the CloudFront distribution"
  value       = module.cloudfront.distribution_hosted_zone_id
}

output "ssl_certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = module.acm.certificate_arn
}

output "wildcard_certificate_arn" {
  description = "ARN of the wildcard SSL certificate"
  value       = module.acm.wildcard_certificate_arn
}

output "cloudfront_key_group_id" {
  description = "ID of the CloudFront key group for signed URLs"
  value       = module.cloudfront.key_group_id
}

output "cloudfront_public_key_id" {
  description = "ID of the CloudFront public key for signed URLs"
  value       = module.cloudfront.public_key_id
}

output "private_key_secret_arn" {
  description = "ARN of the Secrets Manager secret containing the private key"
  value       = aws_secretsmanager_secret.cloudfront_private_key.arn
}

output "route53_records" {
  description = "Map of Route53 record information"
  value = {
    main = {
      name = module.route53.main_record_name
      fqdn = module.route53.main_record_fqdn
    }
    www = {
      name = module.route53.www_record_name
      fqdn = module.route53.www_record_fqdn
    }
    api = {
      name = module.route53.api_record_name
      fqdn = module.route53.api_record_fqdn
    }
    media = {
      name = module.route53.media_record_name
      fqdn = module.route53.media_record_fqdn
    }
    cdn = {
      name = module.route53.cdn_record_name
      fqdn = module.route53.cdn_record_fqdn
    }
    subdomains = module.route53.subdomain_records
  }
}

output "health_check_id" {
  description = "ID of the Route53 health check"
  value       = module.route53.health_check_id
}

output "cloudwatch_dashboard_url" {
  description = "URL to the CloudWatch dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=${aws_cloudwatch_dashboard.cloudfront.dashboard_name}"
}

output "cache_policy_ids" {
  description = "Map of CloudFront cache policy IDs"
  value       = module.cloudfront.cache_policy_ids
}

output "origin_request_policy_ids" {
  description = "Map of CloudFront origin request policy IDs"
  value       = module.cloudfront.origin_request_policy_ids
}

output "cloudfront_urls" {
  description = "Map of CloudFront URLs for different purposes"
  value = {
    main        = "https://${var.domain_name}"
    www         = "https://www.${var.domain_name}"
    api         = var.api_subdomain_enabled ? "https://api.${var.domain_name}" : null
    media       = var.media_subdomain_enabled ? "https://media.${var.domain_name}" : null
    cdn         = var.cdn_subdomain_enabled ? "https://cdn.${var.domain_name}" : null
    cloudfront  = "https://${module.cloudfront.distribution_domain_name}"
  }
}

output "security_configuration" {
  description = "Security configuration summary"
  value = {
    ssl_certificate_status = module.acm.certificate_status
    waf_enabled           = var.waf_web_acl_arn != ""
    geo_restrictions      = var.geo_restriction_type != "none"
    signed_urls_enabled   = true
    hsts_enabled         = true
  }
}