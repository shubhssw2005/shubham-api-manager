# Disaster Recovery Secondary Region Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "global_cluster_identifier" {
  description = "Aurora Global Cluster identifier"
  type        = string
}

variable "enable_secondary_cluster" {
  description = "Enable secondary Aurora cluster"
  type        = bool
  default     = true
}

variable "enable_failover_automation" {
  description = "Enable automated failover Lambda function"
  type        = bool
  default     = true
}

# Aurora Configuration
variable "aurora_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "secondary_instance_count" {
  description = "Number of instances in secondary cluster"
  type        = number
  default     = 2
}

variable "secondary_instance_class" {
  description = "Instance class for secondary cluster"
  type        = string
  default     = "db.r6g.large"
}

variable "security_group_ids" {
  description = "Security group IDs for Aurora cluster"
  type        = list(string)
}

variable "db_subnet_group_name" {
  description = "DB subnet group name"
  type        = string
}

# Backup Configuration
variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 35
}

variable "backup_window" {
  description = "Preferred backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "Preferred maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

# Monitoring
variable "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  type        = string
}

# Disaster Recovery Configuration
variable "rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 15
}

variable "rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 5
}