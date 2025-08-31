# Route53 Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
}

variable "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  type        = string
}

variable "cloudfront_hosted_zone_id" {
  description = "CloudFront distribution hosted zone ID"
  type        = string
}

variable "enable_ipv6" {
  description = "Enable IPv6 AAAA records"
  type        = bool
  default     = true
}

variable "create_www_redirect" {
  description = "Create www subdomain redirect"
  type        = bool
  default     = true
}

variable "subdomains" {
  description = "Map of subdomains to create"
  type        = map(string)
  default     = {}
}

variable "api_subdomain_enabled" {
  description = "Enable API subdomain"
  type        = bool
  default     = false
}

variable "api_cloudfront_domain_name" {
  description = "CloudFront domain name for API subdomain"
  type        = string
  default     = ""
}

variable "media_subdomain_enabled" {
  description = "Enable media subdomain"
  type        = bool
  default     = false
}

variable "cdn_subdomain_enabled" {
  description = "Enable CDN subdomain"
  type        = bool
  default     = false
}

variable "enable_health_check" {
  description = "Enable Route53 health check"
  type        = bool
  default     = true
}

variable "health_check_path" {
  description = "Path for health check"
  type        = string
  default     = "/"
}

variable "health_check_failure_threshold" {
  description = "Number of consecutive failures before marking unhealthy"
  type        = number
  default     = 3
}

variable "health_check_request_interval" {
  description = "Interval between health checks (30 or 10 seconds)"
  type        = number
  default     = 30
  
  validation {
    condition = contains([10, 30], var.health_check_request_interval)
    error_message = "Health check request interval must be 10 or 30 seconds."
  }
}

variable "health_check_alarm_region" {
  description = "AWS region for health check CloudWatch alarms"
  type        = string
  default     = "us-east-1"
}

variable "health_check_alarm_actions" {
  description = "List of ARNs to notify when health check fails"
  type        = list(string)
  default     = []
}

variable "domain_verification_records" {
  description = "Map of domain verification TXT records"
  type        = map(string)
  default     = {}
}

variable "mx_records" {
  description = "List of MX records for email"
  type        = list(string)
  default     = []
}

variable "spf_record" {
  description = "SPF record for email security"
  type        = string
  default     = ""
}

variable "dmarc_record" {
  description = "DMARC record for email security"
  type        = string
  default     = ""
}

variable "caa_records" {
  description = "List of CAA records for certificate authority authorization"
  type        = list(string)
  default     = []
}

variable "ttl_default" {
  description = "Default TTL for DNS records"
  type        = number
  default     = 300
}

variable "enable_dnssec" {
  description = "Enable DNSSEC for the hosted zone"
  type        = bool
  default     = false
}