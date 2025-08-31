# Route53 DNS Management for CloudFront

# Main A record pointing to CloudFront distribution
resource "aws_route53_record" "main" {
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# AAAA record for IPv6 support
resource "aws_route53_record" "main_ipv6" {
  count = var.enable_ipv6 ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# WWW subdomain redirect
resource "aws_route53_record" "www" {
  count = var.create_www_redirect ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = "www.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# Additional subdomains
resource "aws_route53_record" "subdomains" {
  for_each = var.subdomains

  zone_id = var.route53_zone_id
  name    = "${each.key}.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# API subdomain (if different CloudFront distribution)
resource "aws_route53_record" "api" {
  count = var.api_subdomain_enabled ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = "api.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.api_cloudfront_domain_name != "" ? var.api_cloudfront_domain_name : var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# Media subdomain for direct media access
resource "aws_route53_record" "media" {
  count = var.media_subdomain_enabled ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = "media.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# CDN subdomain for static assets
resource "aws_route53_record" "cdn" {
  count = var.cdn_subdomain_enabled ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = "cdn.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_hosted_zone_id
    evaluate_target_health = false
  }
}

# Health check for the main domain
resource "aws_route53_health_check" "main" {
  count = var.enable_health_check ? 1 : 0
  
  fqdn                            = var.domain_name
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = var.health_check_path
  failure_threshold               = var.health_check_failure_threshold
  request_interval                = var.health_check_request_interval
  cloudwatch_alarm_region         = var.health_check_alarm_region
  cloudwatch_alarm_name           = "${var.project_name}-${var.environment}-health-check"
  insufficient_data_health_status = "Failure"

  tags = {
    Name        = "${var.project_name}-${var.environment}-health-check"
    Environment = var.environment
    Project     = var.project_name
  }
}

# CloudWatch alarm for health check
resource "aws_cloudwatch_metric_alarm" "health_check" {
  count = var.enable_health_check ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-health-check-failed"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = "60"
  statistic           = "Minimum"
  threshold           = "1"
  alarm_description   = "Health check failed for ${var.domain_name}"
  alarm_actions       = var.health_check_alarm_actions

  dimensions = {
    HealthCheckId = aws_route53_health_check.main[0].id
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-health-check-alarm"
    Environment = var.environment
  }
}

# TXT record for domain verification
resource "aws_route53_record" "domain_verification" {
  for_each = var.domain_verification_records

  zone_id = var.route53_zone_id
  name    = each.key
  type    = "TXT"
  ttl     = 300
  records = [each.value]
}

# MX records for email (if needed)
resource "aws_route53_record" "mx" {
  count = length(var.mx_records) > 0 ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "MX"
  ttl     = 300
  records = var.mx_records
}

# SPF record for email security
resource "aws_route53_record" "spf" {
  count = var.spf_record != "" ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "TXT"
  ttl     = 300
  records = [var.spf_record]
}

# DMARC record for email security
resource "aws_route53_record" "dmarc" {
  count = var.dmarc_record != "" ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = "_dmarc.${var.domain_name}"
  type    = "TXT"
  ttl     = 300
  records = [var.dmarc_record]
}

# CAA record for certificate authority authorization
resource "aws_route53_record" "caa" {
  count = length(var.caa_records) > 0 ? 1 : 0
  
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "CAA"
  ttl     = 300
  records = var.caa_records
}