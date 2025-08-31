# S3 Intelligent Tiering and Lifecycle Policies
resource "aws_s3_bucket_intelligent_tiering_configuration" "media_bucket_tiering" {
  bucket = aws_s3_bucket.media.id
  name   = "EntireBucket"

  filter {
    prefix = ""
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 125
  }

  optional_fields = ["BucketKeyStatus", "RequestPayer"]

  status = "Enabled"
}

# Intelligent tiering for frequently accessed media
resource "aws_s3_bucket_intelligent_tiering_configuration" "media_hot_tier" {
  bucket = aws_s3_bucket.media.id
  name   = "HotMediaTier"

  filter {
    and {
      prefix = "tenants/"
      tags = {
        AccessPattern = "hot"
      }
    }
  }

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 365
  }

  status = "Enabled"
}

# Intelligent tiering for thumbnails (keep accessible longer)
resource "aws_s3_bucket_intelligent_tiering_configuration" "thumbnails_tiering" {
  bucket = aws_s3_bucket.media.id
  name   = "ThumbnailsTier"

  filter {
    and {
      prefix = "tenants/"
      tags = {
        Type = "thumbnail"
      }
    }
  }

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 180
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 730
  }

  status = "Enabled"
}

resource "aws_s3_bucket_lifecycle_configuration" "media_bucket_lifecycle" {
  bucket = aws_s3_bucket.media.id

  rule {
    id     = "media_lifecycle"
    status = "Enabled"

    filter {
      prefix = "tenants/"
    }

    # Transition to IA after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Transition to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Transition to Deep Archive after 365 days
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    # Delete incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }

    # Delete non-current versions after 30 days
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }

  rule {
    id     = "temp_uploads_cleanup"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 1
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  rule {
    id     = "logs_lifecycle"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 90
    }
  }

  rule {
    id     = "thumbnails_lifecycle"
    status = "Enabled"

    filter {
      and {
        prefix = "tenants/"
        tags = {
          Type = "thumbnail"
        }
      }
    }

    # Keep thumbnails in Standard for quick access
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }
  }

  depends_on = [aws_s3_bucket_versioning.media]
}

# Cost optimization for backup bucket
resource "aws_s3_bucket_lifecycle_configuration" "backup_bucket_lifecycle" {
  bucket = aws_s3_bucket.backup.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    filter {
      prefix = ""
    }

    # Move to IA immediately for backups
    transition {
      days          = 0
      storage_class = "STANDARD_IA"
    }

    # Move to Glacier after 30 days
    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    # Move to Deep Archive after 90 days for long-term retention
    transition {
      days          = 90
      storage_class = "DEEP_ARCHIVE"
    }

    # Keep backups for 7 years for compliance
    expiration {
      days = 2555 # 7 years
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# CloudFront logs lifecycle
resource "aws_s3_bucket_lifecycle_configuration" "cloudfront_logs_lifecycle" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "cloudfront_logs_lifecycle"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Request metrics for cost monitoring
resource "aws_s3_bucket_request_payment_configuration" "media_bucket_requester_pays" {
  count  = var.enable_requester_pays ? 1 : 0
  bucket = aws_s3_bucket.media.id
  payer  = "Requester"
}

# Analytics configuration for usage insights
resource "aws_s3_bucket_analytics_configuration" "media_bucket_analytics" {
  bucket = aws_s3_bucket.media.id
  name   = "EntireBucket"

  filter {
    prefix = "tenants/"
  }

  storage_class_analysis {
    data_export {
      destination {
        s3_bucket_destination {
          bucket_arn = aws_s3_bucket.analytics_bucket.arn
          prefix     = "analytics/"
          format     = "CSV"
        }
      }
      output_schema_version = "V_1"
    }
  }
}

# Separate analytics bucket
resource "aws_s3_bucket" "analytics_bucket" {
  bucket = "${var.project_name}-${var.environment}-analytics"

  tags = merge(var.common_tags, {
    Name        = "${var.project_name}-${var.environment}-analytics"
    Purpose     = "S3 Analytics Storage"
    CostCenter  = "Infrastructure"
  })
}

resource "aws_s3_bucket_versioning" "analytics_bucket_versioning" {
  bucket = aws_s3_bucket.analytics_bucket.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "analytics_bucket_encryption" {
  bucket = aws_s3_bucket.analytics_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Cost allocation tags for detailed billing
resource "aws_s3_bucket_tagging" "media_bucket_cost_tags" {
  bucket = aws_s3_bucket.media.id

  tag_set {
    Environment = var.environment
    Project     = var.project_name
    CostCenter  = "Media"
    Owner       = "Platform"
    Tier        = "Production"
  }
}

# Add missing variables
variable "enable_requester_pays" {
  description = "Enable requester pays for S3 bucket"
  type        = bool
  default     = false
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}
}