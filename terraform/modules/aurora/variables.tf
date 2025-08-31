variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where Aurora cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for Aurora cluster"
  type        = list(string)
}

variable "kms_key_arn" {
  description = "KMS key ARN for encryption"
  type        = string
}

variable "engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
}

variable "instance_class" {
  description = "Aurora instance class"
  type        = string
}

variable "backup_retention_days" {
  description = "Number of days to retain Aurora backups"
  type        = number
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}