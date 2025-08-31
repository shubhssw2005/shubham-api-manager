# Disaster Recovery Secondary Region Module
# Implements secondary region resources for disaster recovery

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

# KMS key for secondary region encryption
resource "aws_kms_key" "dr_secondary_key" {
  description             = "KMS key for disaster recovery secondary region"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name        = "${var.project_name}-dr-secondary-key"
    Environment = var.environment
    Purpose     = "disaster-recovery-secondary"
  }
}

resource "aws_kms_alias" "dr_secondary_key_alias" {
  name          = "alias/${var.project_name}-dr-secondary-key"
  target_key_id = aws_kms_key.dr_secondary_key.key_id
}

# Secondary Aurora cluster (read replica of global cluster)
resource "aws_rds_cluster" "secondary" {
  count                          = var.enable_secondary_cluster ? 1 : 0
  cluster_identifier            = "${var.project_name}-secondary-cluster"
  global_cluster_identifier     = var.global_cluster_identifier
  engine                        = "aurora-postgresql"
  engine_version               = var.aurora_engine_version
  
  # Security
  storage_encrypted             = true
  kms_key_id                   = aws_kms_key.dr_secondary_key.arn
  vpc_security_group_ids       = var.security_group_ids
  db_subnet_group_name         = var.db_subnet_group_name
  
  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql"]
  monitoring_interval            = 60
  monitoring_role_arn           = aws_iam_role.rds_monitoring_secondary.arn
  
  # Backup configuration (for local backups)
  backup_retention_period       = var.backup_retention_period
  preferred_backup_window      = var.backup_window
  preferred_maintenance_window = var.maintenance_window
  copy_tags_to_snapshot        = true
  deletion_protection          = var.deletion_protection
  
  tags = {
    Name        = "${var.project_name}-secondary-cluster"
    Environment = var.environment
    Role        = "secondary"
  }
}

# Secondary cluster instances
resource "aws_rds_cluster_instance" "secondary_instances" {
  count              = var.enable_secondary_cluster ? var.secondary_instance_count : 0
  identifier         = "${var.project_name}-secondary-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.secondary[0].id
  instance_class     = var.secondary_instance_class
  engine             = aws_rds_cluster.secondary[0].engine
  engine_version     = aws_rds_cluster.secondary[0].engine_version
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring_secondary.arn
  
  tags = {
    Name        = "${var.project_name}-secondary-instance-${count.index + 1}"
    Environment = var.environment
    Role        = "secondary"
  }
}

# IAM role for RDS monitoring in secondary region
resource "aws_iam_role" "rds_monitoring_secondary" {
  name = "${var.project_name}-rds-monitoring-secondary-role"

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
    Name        = "${var.project_name}-rds-monitoring-secondary-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "rds_monitoring_secondary" {
  role       = aws_iam_role.rds_monitoring_secondary.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# S3 bucket for replica (destination for cross-region replication)
resource "aws_s3_bucket" "replica" {
  bucket = "${var.project_name}-${var.environment}-replica-${random_id.replica_bucket_suffix.hex}"

  tags = {
    Name        = "${var.project_name}-replica-bucket"
    Environment = var.environment
    Role        = "replica"
  }
}

resource "random_id" "replica_bucket_suffix" {
  byte_length = 4
}

# S3 bucket versioning (required for replication)
resource "aws_s3_bucket_versioning" "replica" {
  bucket = aws_s3_bucket.replica.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption for replica
resource "aws_s3_bucket_server_side_encryption_configuration" "replica" {
  bucket = aws_s3_bucket.replica.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.dr_secondary_key.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 bucket public access block for replica
resource "aws_s3_bucket_public_access_block" "replica" {
  bucket = aws_s3_bucket.replica.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lambda function for automated failover
resource "aws_lambda_function" "failover_automation" {
  count            = var.enable_failover_automation ? 1 : 0
  filename         = "failover_automation.zip"
  function_name    = "${var.project_name}-failover-automation"
  role            = aws_iam_role.lambda_failover.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.failover_lambda_zip[0].output_base64sha256
  runtime         = "python3.11"
  timeout         = 300

  environment {
    variables = {
      GLOBAL_CLUSTER_ID = var.global_cluster_identifier
      SECONDARY_CLUSTER_ID = var.enable_secondary_cluster ? aws_rds_cluster.secondary[0].id : ""
      SNS_TOPIC_ARN = var.sns_topic_arn
      REGION = data.aws_region.current.name
    }
  }

  tags = {
    Name        = "${var.project_name}-failover-automation"
    Environment = var.environment
  }
}

# Lambda deployment package
data "archive_file" "failover_lambda_zip" {
  count       = var.enable_failover_automation ? 1 : 0
  type        = "zip"
  output_path = "failover_automation.zip"
  source {
    content = templatefile("${path.module}/lambda/failover_automation.py", {
      global_cluster_id = var.global_cluster_identifier
    })
    filename = "index.py"
  }
}

# IAM role for Lambda failover function
resource "aws_iam_role" "lambda_failover" {
  name = "${var.project_name}-lambda-failover-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-lambda-failover-role"
    Environment = var.environment
  }
}

# IAM policy for Lambda failover function
resource "aws_iam_role_policy" "lambda_failover" {
  name = "${var.project_name}-lambda-failover-policy"
  role = aws_iam_role.lambda_failover.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:FailoverGlobalCluster",
          "rds:DescribeGlobalClusters",
          "rds:DescribeDBClusters",
          "rds:ModifyGlobalCluster"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.sns_topic_arn
      },
      {
        Effect = "Allow"
        Action = [
          "route53:ChangeResourceRecordSets",
          "route53:GetChange",
          "route53:ListResourceRecordSets"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_failover_basic" {
  role       = aws_iam_role.lambda_failover.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}