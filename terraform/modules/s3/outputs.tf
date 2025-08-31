output "media_bucket_name" {
  description = "Name of the media S3 bucket"
  value       = aws_s3_bucket.media.bucket
}

output "media_bucket_arn" {
  description = "ARN of the media S3 bucket"
  value       = aws_s3_bucket.media.arn
}

output "backup_bucket_name" {
  description = "Name of the backup S3 bucket"
  value       = aws_s3_bucket.backup.bucket
}

output "backup_bucket_arn" {
  description = "ARN of the backup S3 bucket"
  value       = aws_s3_bucket.backup.arn
}

output "logs_bucket_name" {
  description = "Name of the logs S3 bucket"
  value       = aws_s3_bucket.logs.bucket
}

output "logs_bucket_arn" {
  description = "ARN of the logs S3 bucket"
  value       = aws_s3_bucket.logs.arn
}

output "bucket_arns" {
  description = "List of all S3 bucket ARNs"
  value = [
    aws_s3_bucket.media.arn,
    aws_s3_bucket.backup.arn,
    aws_s3_bucket.logs.arn
  ]
}

output "media_processing_topic_arn" {
  description = "ARN of the SNS topic for media processing"
  value       = aws_sns_topic.media_processing.arn
}