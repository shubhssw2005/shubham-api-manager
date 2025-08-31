# Disaster Recovery Module
# Implements cross-region replication and backup automation

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Data source for current region
data "aws_region" "current" {}

# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# KMS key for cross-region encryption
resource "aws_kms_key" "dr_key" {
  description             = "KMS key for disaster recovery encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name        = "${var.project_name}-dr-key"
    Environment = var.environment
    Purpose     = "disaster-recovery"
  }
}

resource "aws_kms_alias" "dr_key_alias" {
  name          = "alias/${var.project_name}-dr-key"
  target_key_id = aws_kms_key.dr_key.key_id
}

# Aurora Global Database for cross-region replication
resource "aws_rds_global_cluster" "main" {
  count                     = var.enable_global_database ? 1 : 0
  global_cluster_identifier = "${var.project_name}-global-cluster"
  engine                    = "aurora-postgresql"
  engine_version           = var.aurora_engine_version
  database_name            = var.database_name
  master_username          = var.master_username
  storage_encrypted        = true
  
  tags = {
    Name        = "${var.project_name}-global-cluster"
    Environment = var.environment
  }
}

# Primary Aurora cluster
resource "aws_rds_cluster" "primary" {
  count                           = var.enable_global_database ? 1 : 0
  cluster_identifier             = "${var.project_name}-primary-cluster"
  global_cluster_identifier      = aws_rds_global_cluster.main[0].id
  engine                         = "aurora-postgresql"
  engine_version                 = var.aurora_engine_version
  database_name                  = var.database_name
  master_username                = var.master_username
  manage_master_user_password    = true
  master_user_secret_kms_key_id  = aws_kms_key.dr_key.arn
  
  # Backup configuration
  backup_retention_period         = var.backup_retention_period
  preferred_backup_window        = var.backup_window
  preferred_maintenance_window   = var.maintenance_window
  copy_tags_to_snapshot          = true
  deletion_protection            = var.deletion_protection
  
  # Point-in-time recovery
  backup_retention_period = 35  # Maximum retention for PITR
  
  # Security
  storage_encrypted               = true
  kms_key_id                     = aws_kms_key.dr_key.arn
  vpc_security_group_ids         = var.security_group_ids
  db_subnet_group_name           = var.db_subnet_group_name
  
  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql"]
  monitoring_interval            = 60
  monitoring_role_arn           = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Name        = "${var.project_name}-primary-cluster"
    Environment = var.environment
    Role        = "primary"
  }
  
  depends_on = [aws_rds_global_cluster.main]
}

# Primary cluster instances
resource "aws_rds_cluster_instance" "primary_instances" {
  count              = var.enable_global_database ? var.primary_instance_count : 0
  identifier         = "${var.project_name}-primary-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.primary[0].id
  instance_class     = var.primary_instance_class
  engine             = aws_rds_cluster.primary[0].engine
  engine_version     = aws_rds_cluster.primary[0].engine_version
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Name        = "${var.project_name}-primary-instance-${count.index + 1}"
    Environment = var.environment
    Role        = "primary"
  }
}# IA
M role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.project_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-rds-monitoring-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# S3 bucket for cross-region replication source
resource "aws_s3_bucket" "primary" {
  bucket = "${var.project_name}-${var.environment}-primary-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.project_name}-primary-bucket"
    Environment = var.environment
    Role        = "primary"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket versioning (required for replication)
resource "aws_s3_bucket_versioning" "primary" {
  bucket = aws_s3_bucket.primary.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "primary" {
  bucket = aws_s3_bucket.primary.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.dr_key.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 bucket public access block
resource "aws_s3_bucket_public_access_block" "primary" {
  bucket = aws_s3_bucket.primary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role for S3 replication
resource "aws_iam_role" "s3_replication" {
  name = "${var.project_name}-s3-replication-role"

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

  tags = {
    Name        = "${var.project_name}-s3-replication-role"
    Environment = var.environment
  }
}

# IAM policy for S3 replication
resource "aws_iam_role_policy" "s3_replication" {
  name = "${var.project_name}-s3-replication-policy"
  role = aws_iam_role.s3_replication.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.primary.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.primary.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${var.replica_bucket_arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = [
          aws_kms_key.dr_key.arn,
          var.replica_kms_key_arn
        ]
      }
    ]
  })
}

# S3 replication configuration
resource "aws_s3_bucket_replication_configuration" "primary" {
  count  = var.enable_s3_replication ? 1 : 0
  role   = aws_iam_role.s3_replication.arn
  bucket = aws_s3_bucket.primary.id

  rule {
    id     = "replicate-all"
    status = "Enabled"

    destination {
      bucket        = var.replica_bucket_arn
      storage_class = "STANDARD_IA"
      
      encryption_configuration {
        replica_kms_key_id = var.replica_kms_key_arn
      }
      
      # Replication Time Control for 15-minute RTO
      replication_time {
        status = "Enabled"
        time {
          minutes = 15
        }
      }
      
      # Metrics for monitoring replication
      metrics {
        status = "Enabled"
        event_threshold {
          minutes = 15
        }
      }
    }
  }

  depends_on = [aws_s3_bucket_versioning.primary]
}

# CloudWatch alarms for replication monitoring
resource "aws_cloudwatch_metric_alarm" "s3_replication_failure" {
  count               = var.enable_s3_replication ? 1 : 0
  alarm_name          = "${var.project_name}-s3-replication-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ReplicationLatency"
  namespace           = "AWS/S3"
  period              = "300"
  statistic           = "Maximum"
  threshold           = "900"  # 15 minutes in seconds
  alarm_description   = "S3 replication latency exceeded threshold"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    SourceBucket      = aws_s3_bucket.primary.bucket
    DestinationBucket = var.replica_bucket_name
  }

  tags = {
    Name        = "${var.project_name}-s3-replication-alarm"
    Environment = var.environment
  }
}