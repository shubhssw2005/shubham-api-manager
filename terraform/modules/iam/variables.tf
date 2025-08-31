variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "eks_cluster_arn" {
  description = "EKS cluster ARN"
  type        = string
}

variable "s3_bucket_arns" {
  description = "List of S3 bucket ARNs"
  type        = list(string)
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}