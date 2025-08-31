variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where EKS cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for EKS cluster"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for EKS cluster"
  type        = list(string)
}

variable "kms_key_arn" {
  description = "KMS key ARN for encryption"
  type        = string
}

variable "general_node_desired_size" {
  description = "Desired number of general purpose nodes"
  type        = number
}

variable "general_node_max_size" {
  description = "Maximum number of general purpose nodes"
  type        = number
}

variable "general_node_min_size" {
  description = "Minimum number of general purpose nodes"
  type        = number
}

variable "memory_node_desired_size" {
  description = "Desired number of memory optimized nodes"
  type        = number
}

variable "memory_node_max_size" {
  description = "Maximum number of memory optimized nodes"
  type        = number
}

variable "memory_node_min_size" {
  description = "Minimum number of memory optimized nodes"
  type        = number
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}