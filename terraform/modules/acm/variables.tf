# ACM Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "domain_name" {
  description = "Primary domain name for the certificate"
  type        = string
}

variable "subject_alternative_names" {
  description = "Additional domain names for the certificate"
  type        = list(string)
  default     = []
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for DNS validation"
  type        = string
}

variable "validate_certificate" {
  description = "Whether to validate the certificate using DNS"
  type        = bool
  default     = true
}

variable "create_wildcard_cert" {
  description = "Whether to create a wildcard certificate"
  type        = bool
  default     = false
}

variable "wildcard_subject_alternative_names" {
  description = "Additional domain names for the wildcard certificate"
  type        = list(string)
  default     = []
}

variable "enable_expiry_monitoring" {
  description = "Enable CloudWatch alarms for certificate expiry"
  type        = bool
  default     = true
}

variable "expiry_warning_days" {
  description = "Number of days before expiry to trigger alarm"
  type        = number
  default     = 30
}

variable "alarm_actions" {
  description = "List of ARNs to notify when certificate is expiring"
  type        = list(string)
  default     = []
}

variable "certificate_transparency_logging_preference" {
  description = "Certificate transparency logging preference"
  type        = string
  default     = "ENABLED"
  
  validation {
    condition = contains([
      "ENABLED",
      "DISABLED"
    ], var.certificate_transparency_logging_preference)
    error_message = "Certificate transparency logging preference must be ENABLED or DISABLED."
  }
}