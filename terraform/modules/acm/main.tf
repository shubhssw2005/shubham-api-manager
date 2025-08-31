# ACM Certificate Management for CloudFront

# Primary certificate for the main domain
resource "aws_acm_certificate" "main" {
  domain_name               = var.domain_name
  subject_alternative_names = var.subject_alternative_names
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cert"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Certificate validation using Route53 DNS
resource "aws_acm_certificate_validation" "main" {
  count = var.validate_certificate ? 1 : 0
  
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]

  timeouts {
    create = "10m"
  }
}

# Route53 records for certificate validation
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.route53_zone_id
}

# Wildcard certificate for subdomains (optional)
resource "aws_acm_certificate" "wildcard" {
  count = var.create_wildcard_cert ? 1 : 0
  
  domain_name               = "*.${var.domain_name}"
  subject_alternative_names = var.wildcard_subject_alternative_names
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-wildcard-cert"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Wildcard certificate validation
resource "aws_acm_certificate_validation" "wildcard" {
  count = var.create_wildcard_cert && var.validate_certificate ? 1 : 0
  
  certificate_arn         = aws_acm_certificate.wildcard[0].arn
  validation_record_fqdns = [for record in aws_route53_record.wildcard_cert_validation : record.fqdn]

  timeouts {
    create = "10m"
  }
}

# Route53 records for wildcard certificate validation
resource "aws_route53_record" "wildcard_cert_validation" {
  for_each = var.create_wildcard_cert ? {
    for dvo in aws_acm_certificate.wildcard[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.route53_zone_id
}

# Certificate expiry monitoring
resource "aws_cloudwatch_metric_alarm" "cert_expiry" {
  count = var.enable_expiry_monitoring ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-cert-expiry"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DaysToExpiry"
  namespace           = "AWS/CertificateManager"
  period              = "86400" # 24 hours
  statistic           = "Minimum"
  threshold           = var.expiry_warning_days
  alarm_description   = "SSL certificate expiring soon"
  alarm_actions       = var.alarm_actions

  dimensions = {
    CertificateArn = aws_acm_certificate.main.arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cert-expiry-alarm"
    Environment = var.environment
  }
}

# Wildcard certificate expiry monitoring
resource "aws_cloudwatch_metric_alarm" "wildcard_cert_expiry" {
  count = var.create_wildcard_cert && var.enable_expiry_monitoring ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-wildcard-cert-expiry"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DaysToExpiry"
  namespace           = "AWS/CertificateManager"
  period              = "86400"
  statistic           = "Minimum"
  threshold           = var.expiry_warning_days
  alarm_description   = "Wildcard SSL certificate expiring soon"
  alarm_actions       = var.alarm_actions

  dimensions = {
    CertificateArn = aws_acm_certificate.wildcard[0].arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-wildcard-cert-expiry-alarm"
    Environment = var.environment
  }
}