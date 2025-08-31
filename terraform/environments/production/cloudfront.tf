# Production CloudFront Configuration

# Generate key pair for signed URLs
resource "tls_private_key" "cloudfront_key" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

# Store private key in AWS Secrets Manager
resource "aws_secretsmanager_secret" "cloudfront_private_key" {
  name                    = "${var.project_name}-${var.environment}-cloudfront-private-key"
  description             = "CloudFront private key for signed URLs"
  recovery_window_in_days = 7

  tags = {
    Name        = "${var.project_name}-${var.environment}-cloudfront-key"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_secretsmanager_secret_version" "cloudfront_private_key" {
  secret_id     = aws_secretsmanager_secret.cloudfront_private_key.id
  secret_string = tls_private_key.cloudfront_key.private_key_pem
}

# ACM Certificate
module "acm" {
  source = "../../modules/acm"

  project_name                = var.project_name
  environment                 = var.environment
  domain_name                 = var.domain_name
  subject_alternative_names   = var.subject_alternative_names
  route53_zone_id            = var.route53_zone_id
  validate_certificate       = true
  create_wildcard_cert       = var.create_wildcard_cert
  enable_expiry_monitoring   = true
  expiry_warning_days        = 30
  alarm_actions              = var.certificate_alarm_actions
}

# CloudFront Distribution
module "cloudfront" {
  source = "../../modules/cloudfront"

  project_name           = var.project_name
  environment           = var.environment
  s3_bucket_name        = module.s3.bucket_name
  s3_bucket_domain_name = module.s3.bucket_domain_name
  api_gateway_domain    = var.api_gateway_domain
  custom_domains        = var.custom_domains
  ssl_certificate_arn   = module.acm.certificate_arn
  price_class          = var.cloudfront_price_class
  
  # Geographic restrictions
  geo_restriction_type      = var.geo_restriction_type
  geo_restriction_locations = var.geo_restriction_locations
  
  # Logging
  logging_bucket = var.cloudfront_logging_bucket
  
  # WAF
  waf_web_acl_arn = var.waf_web_acl_arn
  
  # Signed URLs
  public_key_content = tls_private_key.cloudfront_key.public_key_pem
  
  # CORS
  cors_allowed_origins = var.cors_allowed_origins

  depends_on = [
    module.acm.certificate_arn,
    module.s3
  ]
}

# Route53 DNS Records
module "route53" {
  source = "../../modules/route53"

  project_name               = var.project_name
  environment               = var.environment
  domain_name               = var.domain_name
  route53_zone_id           = var.route53_zone_id
  cloudfront_domain_name    = module.cloudfront.distribution_domain_name
  cloudfront_hosted_zone_id = module.cloudfront.distribution_hosted_zone_id
  
  # Subdomains
  create_www_redirect      = true
  api_subdomain_enabled    = var.api_subdomain_enabled
  media_subdomain_enabled  = var.media_subdomain_enabled
  cdn_subdomain_enabled    = var.cdn_subdomain_enabled
  subdomains              = var.additional_subdomains
  
  # Health checks
  enable_health_check           = var.enable_health_check
  health_check_path            = var.health_check_path
  health_check_failure_threshold = var.health_check_failure_threshold
  health_check_alarm_actions   = var.health_check_alarm_actions
  
  # Email records
  mx_records    = var.mx_records
  spf_record    = var.spf_record
  dmarc_record  = var.dmarc_record
  caa_records   = var.caa_records
  
  # Domain verification
  domain_verification_records = var.domain_verification_records

  depends_on = [module.cloudfront]
}

# S3 bucket policy update for CloudFront OAC
resource "aws_s3_bucket_policy" "cloudfront_oac" {
  bucket = module.s3.bucket_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontServicePrincipal"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${module.s3.bucket_arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = module.cloudfront.distribution_arn
          }
        }
      }
    ]
  })

  depends_on = [module.cloudfront]
}

# CloudWatch Dashboard for CloudFront monitoring
resource "aws_cloudwatch_dashboard" "cloudfront" {
  dashboard_name = "${var.project_name}-${var.environment}-cloudfront"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/CloudFront", "Requests", "DistributionId", module.cloudfront.distribution_id],
            [".", "BytesDownloaded", ".", "."],
            [".", "BytesUploaded", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "CloudFront Traffic"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/CloudFront", "CacheHitRate", "DistributionId", module.cloudfront.distribution_id],
            [".", "OriginLatency", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "CloudFront Performance"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/CloudFront", "4xxErrorRate", "DistributionId", module.cloudfront.distribution_id],
            [".", "5xxErrorRate", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "CloudFront Errors"
          period  = 300
        }
      }
    ]
  })
}

# CloudWatch Alarms for CloudFront
resource "aws_cloudwatch_metric_alarm" "high_4xx_error_rate" {
  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-high-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "5"
  alarm_description   = "High 4xx error rate on CloudFront distribution"
  alarm_actions       = var.cloudfront_alarm_actions

  dimensions = {
    DistributionId = module.cloudfront.distribution_id
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cloudfront-4xx-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "high_5xx_error_rate" {
  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-high-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "5xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "High 5xx error rate on CloudFront distribution"
  alarm_actions       = var.cloudfront_alarm_actions

  dimensions = {
    DistributionId = module.cloudfront.distribution_id
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cloudfront-5xx-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "low_cache_hit_rate" {
  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-low-cache-hit-rate"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "CacheHitRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Low cache hit rate on CloudFront distribution"
  alarm_actions       = var.cloudfront_alarm_actions

  dimensions = {
    DistributionId = module.cloudfront.distribution_id
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cloudfront-cache-hit-alarm"
    Environment = var.environment
  }
}