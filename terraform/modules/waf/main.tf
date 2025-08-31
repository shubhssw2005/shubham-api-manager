# AWS WAF v2 Web ACL for comprehensive protection
resource "aws_wafv2_web_acl" "main" {
  name  = "${var.project_name}-${var.environment}-waf"
  scope = "CLOUDFRONT"

  default_action {
    allow {}
  }

  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1

    override_action {
      none {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"

        scope_down_statement {
          geo_match_statement {
            country_codes = ["US", "CA", "GB", "DE", "FR", "AU", "JP"]
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }

    action {
      block {}
    }
  }

  # AWS Managed Rules - Core Rule Set
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        # Exclude specific rules if needed
        excluded_rule {
          name = "SizeRestrictions_BODY"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - Known Bad Inputs
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # SQL Injection Protection
  rule {
    name     = "AWSManagedRulesSQLiRuleSet"
    priority = 4

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesSQLiRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "SQLiRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Block suspicious user agents
  rule {
    name     = "BlockSuspiciousUserAgents"
    priority = 5

    action {
      block {}
    }

    statement {
      byte_match_statement {
        search_string = "bot"
        field_to_match {
          single_header {
            name = "user-agent"
          }
        }
        text_transformation {
          priority = 0
          type     = "LOWERCASE"
        }
        positional_constraint = "CONTAINS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "BlockSuspiciousUserAgents"
      sampled_requests_enabled   = true
    }
  }

  # Geo-blocking rule (block specific countries if needed)
  rule {
    name     = "GeoBlockRule"
    priority = 6

    action {
      block {}
    }

    statement {
      geo_match_statement {
        country_codes = var.blocked_countries
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoBlockRule"
      sampled_requests_enabled   = true
    }
  }

  # IP reputation rule
  rule {
    name     = "AWSManagedRulesAmazonIpReputationList"
    priority = 7

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAmazonIpReputationList"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IpReputationListMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-${var.environment}-waf"
    sampled_requests_enabled   = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-waf"
    Environment = var.environment
  }
}

# Shield Advanced subscription
resource "aws_shield_protection" "cloudfront" {
  count        = var.enable_shield_advanced ? 1 : 0
  name         = "${var.project_name}-${var.environment}-cloudfront-protection"
  resource_arn = var.cloudfront_distribution_arn
}

resource "aws_shield_protection" "alb" {
  count        = var.enable_shield_advanced ? 1 : 0
  name         = "${var.project_name}-${var.environment}-alb-protection"
  resource_arn = var.alb_arn
}

# Shield Advanced DRT access role
resource "aws_iam_role" "shield_drt_access" {
  count = var.enable_shield_advanced ? 1 : 0
  name  = "${var.project_name}-${var.environment}-shield-drt-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "drt.shield.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "shield_drt_access" {
  count      = var.enable_shield_advanced ? 1 : 0
  role       = aws_iam_role.shield_drt_access[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSShieldDRTAccessPolicy"
}

# WAF logging configuration
resource "aws_cloudwatch_log_group" "waf_log_group" {
  name              = "/aws/wafv2/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-waf-logs"
    Environment = var.environment
  }
}

resource "aws_wafv2_web_acl_logging_configuration" "main" {
  resource_arn            = aws_wafv2_web_acl.main.arn
  log_destination_configs = [aws_cloudwatch_log_group.waf_log_group.arn]

  redacted_fields {
    single_header {
      name = "authorization"
    }
  }

  redacted_fields {
    single_header {
      name = "cookie"
    }
  }
}

# CloudWatch alarms for WAF
resource "aws_cloudwatch_metric_alarm" "waf_blocked_requests" {
  alarm_name          = "${var.project_name}-${var.environment}-waf-blocked-requests"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = "300"
  statistic           = "Sum"
  threshold           = "100"
  alarm_description   = "This metric monitors blocked requests by WAF"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    WebACL = aws_wafv2_web_acl.main.name
    Region = "CloudFront"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-waf-blocked-requests"
    Environment = var.environment
  }
}