variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where Redis cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for Redis cluster"
  type        = list(string)
}

variable "kms_key_arn" {
  description = "KMS key ARN for encryption"
  type        = string
}

variable "node_type" {
  description = "ElastiCache Redis node type"
  type        = string
}

variable "num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}