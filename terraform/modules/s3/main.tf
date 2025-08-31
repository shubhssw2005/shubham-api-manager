# S3 Bucket for Media Storage
resource "aws_s3_bucket" "media" {
  bucket = "${var.project_name}-${var.environment}-media-${random_id.bucket_suffix.hex}"

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-media"
    Type = "media"
  })
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backup" {
  bucket = "${var.project_name}-${var.environment}-backup-${random_id.bucket_suffix.hex}"

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-backup"
    Type = "backup"
  })
}

# S3 Bucket for Logs
resource "aws_s3_bucket" "logs" {
  bucket = "${var.project_name}-${var.environment}-logs-${random_id.bucket_suffix.hex}"

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-logs"
    Type = "logs"
  })
}

# Random ID for bucket suffix to ensure uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Bucket Versioning - Media
resource "aws_s3_bucket_versioning" "media" {
  bucket = aws_s3_bucket.media.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Versioning - Backup
resource "aws_s3_bucket_versioning" "backup" {
  bucket = aws_s3_bucket.backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Encryption - Media
resource "aws_s3_bucket_server_side_encryption_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Encryption - Backup
resource "aws_s3_bucket_server_side_encryption_configuration" "backup" {
  bucket = aws_s3_bucket.backup.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Encryption - Logs
resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Public Access Block - Media
resource "aws_s3_bucket_public_access_block" "media" {
  bucket = aws_s3_bucket.media.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Public Access Block - Backup
resource "aws_s3_bucket_public_access_block" "backup" {
  bucket = aws_s3_bucket.backup.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Public Access Block - Logs
resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Lifecycle Configuration - Media
resource "aws_s3_bucket_lifecycle_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    id     = "media_lifecycle"
    status = "Enabled"

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

    # Delete non-current versions after 30 days
    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    # Delete incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# S3 Bucket Lifecycle Configuration - Backup
resource "aws_s3_bucket_lifecycle_configuration" "backup" {
  bucket = aws_s3_bucket.backup.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    # Transition to IA after 7 days
    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }

    # Transition to Glacier after 30 days
    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    # Delete objects after 2555 days (7 years)
    expiration {
      days = 2555
    }

    # Delete non-current versions after 7 days
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# S3 Bucket Lifecycle Configuration - Logs
resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "logs_lifecycle"
    status = "Enabled"

    # Transition to IA after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Delete objects after 90 days
    expiration {
      days = 90
    }
  }
}

# S3 Bucket Notification for Media Processing
resource "aws_s3_bucket_notification" "media_notification" {
  bucket = aws_s3_bucket.media.id

  topic {
    topic_arn     = aws_sns_topic.media_processing.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "tenants/"
  }

  depends_on = [aws_sns_topic_policy.media_processing]
}

# SNS Topic for Media Processing
resource "aws_sns_topic" "media_processing" {
  name         = "${var.project_name}-${var.environment}-media-processing"
  kms_master_key_id = var.kms_key_arn

  tags = var.tags
}

# SNS Topic Policy for S3 to publish
resource "aws_sns_topic_policy" "media_processing" {
  arn = aws_sns_topic.media_processing.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.media_processing.arn
        Condition = {
          StringEquals = {
            "aws:SourceArn" = aws_s3_bucket.media.arn
          }
        }
      }
    ]
  })
}

# Cross-Region Replication Role for Media Bucket
resource "aws_iam_role" "replication" {
  count = var.environment == "production" ? 1 : 0
  name  = "${var.project_name}-${var.environment}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for Cross-Region Replication
resource "aws_iam_role_policy" "replication" {
  count = var.environment == "production" ? 1 : 0
  name  = "${var.project_name}-${var.environment}-s3-replication-policy"
  role  = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Effect = "Allow"
        Resource = [
          "${aws_s3_bucket.media.arn}/*"
        ]
      },
      {
        Action = [
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.media.arn
        ]
      },
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Effect = "Allow"
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-media-replica-*/*"
        ]
      }
    ]
  })
}