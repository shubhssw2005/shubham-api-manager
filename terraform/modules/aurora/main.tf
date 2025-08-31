# Aurora Subnet Group
resource "aws_db_subnet_group" "aurora" {
  name       = "${var.project_name}-${var.environment}-aurora-subnet-group"
  subnet_ids = var.subnet_ids

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-aurora-subnet-group"
  })
}

# Aurora Security Group
resource "aws_security_group" "aurora" {
  name_prefix = "${var.project_name}-${var.environment}-aurora-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-aurora-sg"
  })
}

# Random password for Aurora
resource "random_password" "aurora_master_password" {
  length  = 16
  special = true
}

# Aurora Parameter Group
resource "aws_rds_cluster_parameter_group" "aurora" {
  family = "aurora-postgresql15"
  name   = "${var.project_name}-${var.environment}-aurora-cluster-pg"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  tags = var.tags
}

# Aurora DB Parameter Group
resource "aws_db_parameter_group" "aurora" {
  family = "aurora-postgresql15"
  name   = "${var.project_name}-${var.environment}-aurora-db-pg"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  tags = var.tags
}

# Aurora Cluster
resource "aws_rds_cluster" "aurora" {
  cluster_identifier      = "${var.project_name}-${var.environment}-aurora"
  engine                 = "aurora-postgresql"
  engine_mode            = "provisioned"
  engine_version         = var.engine_version
  database_name          = replace("${var.project_name}_${var.environment}", "-", "_")
  master_username        = "postgres"
  master_password        = random_password.aurora_master_password.result
  
  # Serverless v2 scaling configuration
  serverlessv2_scaling_configuration {
    max_capacity = 16
    min_capacity = 0.5
  }

  backup_retention_period = var.backup_retention_days
  preferred_backup_window = "03:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  
  db_subnet_group_name   = aws_db_subnet_group.aurora.name
  vpc_security_group_ids = [aws_security_group.aurora.id]
  
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.aurora.name
  
  storage_encrypted = true
  kms_key_id       = var.kms_key_arn
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  deletion_protection = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${var.project_name}-${var.environment}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-aurora-cluster"
  })
}

# Aurora Cluster Instances
resource "aws_rds_cluster_instance" "aurora_instances" {
  count              = 2
  identifier         = "${var.project_name}-${var.environment}-aurora-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.aurora.id
  instance_class     = var.instance_class
  engine             = aws_rds_cluster.aurora.engine
  engine_version     = aws_rds_cluster.aurora.engine_version
  
  db_parameter_group_name = aws_db_parameter_group.aurora.name
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn         = aws_iam_role.rds_enhanced_monitoring.arn

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-aurora-instance-${count.index + 1}"
  })
}

# Store Aurora credentials in AWS Secrets Manager
resource "aws_secretsmanager_secret" "aurora_credentials" {
  name        = "${var.project_name}/${var.environment}/aurora/credentials"
  description = "Aurora PostgreSQL credentials for ${var.project_name}-${var.environment}"
  kms_key_id  = var.kms_key_arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "aurora_credentials" {
  secret_id = aws_secretsmanager_secret.aurora_credentials.id
  secret_string = jsonencode({
    username = aws_rds_cluster.aurora.master_username
    password = random_password.aurora_master_password.result
    engine   = "postgres"
    host     = aws_rds_cluster.aurora.endpoint
    port     = aws_rds_cluster.aurora.port
    dbname   = aws_rds_cluster.aurora.database_name
  })
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${var.project_name}-${var.environment}-rds-enhanced-monitoring"

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

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Data source for VPC
data "aws_vpc" "main" {
  id = var.vpc_id
}