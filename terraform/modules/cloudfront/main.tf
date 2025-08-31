# CloudFront Distribution with S3 Origin and Signed URL Support

# Origin Access Control for S3
resource "aws_cloudfront_origin_access_control" "s3_oac" {
  name                              = "${var.project_name}-${var.environment}-s3-oac"
  description                       = "OAC for S3 bucket access"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.project_name} ${var.environment} CDN"
  default_root_object = "index.html"
  price_class         = var.price_class
  
  # Aliases for custom domains
  aliases = var.custom_domains

  # S3 Origin Configuration
  origin {
    domain_name              = var.s3_bucket_domain_name
    origin_id                = "S3-${var.s3_bucket_name}"
    origin_access_control_id = aws_cloudfront_origin_access_control.s3_oac.id

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # API Gateway Origin for dynamic content
  dynamic "origin" {
    for_each = var.api_gateway_domain != "" ? [1] : []
    content {
      domain_name = var.api_gateway_domain
      origin_id   = "API-Gateway"
      
      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }

  # Default cache behavior for media files
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${var.s3_bucket_name}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    # Cache policy for media content
    cache_policy_id = aws_cloudfront_cache_policy.media_cache_policy.id
    
    # Origin request policy
    origin_request_policy_id = aws_cloudfront_origin_request_policy.media_origin_policy.id

    # Trusted key groups for signed URLs
    trusted_key_groups = [aws_cloudfront_key_group.signed_url_key_group.id]
  }

  # Cache behavior for API endpoints
  dynamic "ordered_cache_behavior" {
    for_each = var.api_gateway_domain != "" ? [1] : []
    content {
      path_pattern           = "/api/*"
      allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
      cached_methods         = ["GET", "HEAD", "OPTIONS"]
      target_origin_id       = "API-Gateway"
      compress               = true
      viewer_protocol_policy = "redirect-to-https"
      
      cache_policy_id = aws_cloudfront_cache_policy.api_cache_policy.id
      origin_request_policy_id = aws_cloudfront_origin_request_policy.api_origin_policy.id
    }
  }

  # Geographic restrictions
  restrictions {
    geo_restriction {
      restriction_type = var.geo_restriction_type
      locations        = var.geo_restriction_locations
    }
  }

  # SSL Certificate configuration
  viewer_certificate {
    acm_certificate_arn            = var.ssl_certificate_arn
    ssl_support_method             = "sni-only"
    minimum_protocol_version       = "TLSv1.2_2021"
    cloudfront_default_certificate = var.ssl_certificate_arn == "" ? true : false
  }

  # Custom error pages
  custom_error_response {
    error_code         = 403
    response_code      = 200
    response_page_path = "/index.html"
  }

  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  # Logging configuration
  logging_config {
    include_cookies = false
    bucket          = var.logging_bucket
    prefix          = "cloudfront-logs/"
  }

  # Web Application Firewall
  web_acl_id = var.waf_web_acl_arn

  tags = {
    Name        = "${var.project_name}-${var.environment}-cloudfront"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Cache Policies
resource "aws_cloudfront_cache_policy" "media_cache_policy" {
  name        = "${var.project_name}-${var.environment}-media-cache"
  comment     = "Cache policy for media files"
  default_ttl = 86400  # 1 day
  max_ttl     = 31536000  # 1 year
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    query_strings_config {
      query_string_behavior = "whitelist"
      query_strings {
        items = ["v", "version", "t"]
      }
    }

    headers_config {
      header_behavior = "whitelist"
      headers {
        items = ["CloudFront-Viewer-Country", "CloudFront-Is-Mobile-Viewer"]
      }
    }

    cookies_config {
      cookie_behavior = "none"
    }
  }
}

resource "aws_cloudfront_cache_policy" "api_cache_policy" {
  name        = "${var.project_name}-${var.environment}-api-cache"
  comment     = "Cache policy for API endpoints"
  default_ttl = 0
  max_ttl     = 300  # 5 minutes
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    query_strings_config {
      query_string_behavior = "all"
    }

    headers_config {
      header_behavior = "whitelist"
      headers {
        items = [
          "Authorization",
          "Content-Type",
          "X-Tenant-ID",
          "CloudFront-Viewer-Country"
        ]
      }
    }

    cookies_config {
      cookie_behavior = "none"
    }
  }
}

# Origin Request Policies
resource "aws_cloudfront_origin_request_policy" "media_origin_policy" {
  name    = "${var.project_name}-${var.environment}-media-origin"
  comment = "Origin request policy for media files"

  query_strings_config {
    query_string_behavior = "whitelist"
    query_strings {
      items = ["response-content-disposition", "response-content-type"]
    }
  }

  headers_config {
    header_behavior = "whitelist"
    headers {
      items = [
        "Access-Control-Request-Headers",
        "Access-Control-Request-Method",
        "Origin"
      ]
    }
  }

  cookies_config {
    cookie_behavior = "none"
  }
}

resource "aws_cloudfront_origin_request_policy" "api_origin_policy" {
  name    = "${var.project_name}-${var.environment}-api-origin"
  comment = "Origin request policy for API endpoints"

  query_strings_config {
    query_string_behavior = "all"
  }

  headers_config {
    header_behavior = "whitelist"
    headers {
      items = [
        "Authorization",
        "Content-Type",
        "X-Tenant-ID",
        "X-API-Key",
        "User-Agent",
        "Referer"
      ]
    }
  }

  cookies_config {
    cookie_behavior = "all"
  }
}

# Key Group for Signed URLs
resource "aws_cloudfront_public_key" "signed_url_key" {
  comment     = "Public key for CloudFront signed URLs"
  encoded_key = var.public_key_content
  name        = "${var.project_name}-${var.environment}-signed-url-key"
}

resource "aws_cloudfront_key_group" "signed_url_key_group" {
  comment = "Key group for signed URLs"
  items   = [aws_cloudfront_public_key.signed_url_key.id]
  name    = "${var.project_name}-${var.environment}-signed-url-key-group"
}

# Response Headers Policy for Security
resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name    = "${var.project_name}-${var.environment}-security-headers"
  comment = "Security headers policy"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = true
      override                   = true
    }

    content_type_options {
      override = true
    }

    frame_options {
      frame_option = "DENY"
      override     = true
    }

    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
  }

  cors_config {
    access_control_allow_credentials = false
    access_control_max_age_sec      = 600

    access_control_allow_headers {
      items = ["*"]
    }

    access_control_allow_methods {
      items = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    }

    access_control_allow_origins {
      items = var.cors_allowed_origins
    }

    origin_override = true
  }
}