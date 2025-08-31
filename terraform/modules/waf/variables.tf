variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "enable_shield_advanced" {
  description = "Enable AWS Shield Advanced protection"
  type        = bool
  default     = false
}

variable "blocked_countries" {
  description = "List of country codes to block"
  type        = list(string)
  default     = ["CN", "RU", "KP"]
}

variable "cloudfront_distribution_arn" {
  description = "ARN of the CloudFront distribution to protect"
  type        = string
}

variable "alb_arn" {
  description = "ARN of the Application Load Balancer to protect"
  type        = string
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  type        = string
}