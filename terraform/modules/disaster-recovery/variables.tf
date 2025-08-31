# Disaster Recovery Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "enable_global_database" {
  description = "Enable Aurora Global Database for cross-region replication"
  type        = bool
  default     = true
}

variable "enable_s3_replication" {
  description = "Enable S3 cross-region replication"
  type        = bool
  default     = true
}

# Aurora Configuration
variable "aurora_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "main"
}

variable "master_username" {
  description = "Master username for Aurora cluster"
  type        = string
  default     = "postgres"
}

variable "primary_instance_count" {
  description = "Number of instances in primary cluster"
  type        = number
  default     = 2
}

variable "primary_instance_class" {
  description = "Instance class for primary cluster"
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

# S3 Replication Configuration
variable "replica_bucket_arn" {
  description = "ARN of the replica S3 bucket"
  type        = string
  default     = ""
}

variable "replica_bucket_name" {
  description = "Name of the replica S3 bucket"
  type        = string
  default     = ""
}

variable "replica_kms_key_arn" {
  description = "KMS key ARN for replica bucket encryption"
  type        = string
  default     = ""
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

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
}

variable "enable_automated_backups" {
  description = "Enable automated backup schedules"
  type        = bool
  default     = true
}

variable "backup_schedule_expression" {
  description = "CloudWatch Events schedule expression for backups"
  type        = string
  default     = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC
}

# Monitoring and Alerting
variable "alert_email_addresses" {
  description = "List of email addresses for disaster recovery alerts"
  type        = list(string)
  default     = []
}

variable "enable_automated_failover" {
  description = "Enable automated failover monitoring (not actual failover)"
  type        = bool
  default     = true
}

variable "load_balancer_arn_suffix" {
  description = "Load balancer ARN suffix for monitoring"
  type        = string
  default     = ""
}