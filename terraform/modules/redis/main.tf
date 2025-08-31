# Redis Subnet Group
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-${var.environment}-redis-subnet-group"
  subnet_ids = var.subnet_ids

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-redis-subnet-group"
  })
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-${var.environment}-redis-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
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
    Name = "${var.project_name}-${var.environment}-redis-sg"
  })
}

# Redis Parameter Group
resource "aws_elasticache_parameter_group" "redis" {
  family = "redis7.x"
  name   = "${var.project_name}-${var.environment}-redis-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  tags = var.tags
}

# Random auth token for Redis
resource "random_password" "redis_auth_token" {
  length  = 32
  special = false
}

# Redis Replication Group (Cluster Mode Enabled)
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "${var.project_name}-${var.environment}-redis"
  description                  = "Redis cluster for ${var.project_name}-${var.environment}"
  
  port                        = 6379
  parameter_group_name        = aws_elasticache_parameter_group.redis.name
  node_type                   = var.node_type
  
  # Cluster mode configuration
  num_cache_clusters          = var.num_cache_nodes
  
  # Security
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = random_password.redis_auth_token.result
  kms_key_id                  = var.kms_key_arn
  
  # Network
  subnet_group_name           = aws_elasticache_subnet_group.redis.name
  security_group_ids          = [aws_security_group.redis.id]
  
  # Backup
  snapshot_retention_limit    = 5
  snapshot_window            = "03:00-05:00"
  maintenance_window         = "sun:05:00-sun:07:00"
  
  # Monitoring
  notification_topic_arn     = aws_sns_topic.redis_notifications.arn
  
  # Multi-AZ
  multi_az_enabled           = true
  automatic_failover_enabled = true
  
  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-redis-cluster"
  })
}

# SNS Topic for Redis notifications
resource "aws_sns_topic" "redis_notifications" {
  name         = "${var.project_name}-${var.environment}-redis-notifications"
  kms_master_key_id = var.kms_key_arn

  tags = var.tags
}

# Store Redis credentials in AWS Secrets Manager
resource "aws_secretsmanager_secret" "redis_credentials" {
  name        = "${var.project_name}/${var.environment}/redis/credentials"
  description = "Redis credentials for ${var.project_name}-${var.environment}"
  kms_key_id  = var.kms_key_arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "redis_credentials" {
  secret_id = aws_secretsmanager_secret.redis_credentials.id
  secret_string = jsonencode({
    auth_token = random_password.redis_auth_token.result
    host       = aws_elasticache_replication_group.redis.primary_endpoint_address
    port       = aws_elasticache_replication_group.redis.port
    configuration_endpoint = aws_elasticache_replication_group.redis.configuration_endpoint_address
  })
}

# CloudWatch Log Group for Redis slow logs
resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name              = "/aws/elasticache/redis/${var.project_name}-${var.environment}/slow-log"
  retention_in_days = 30
  kms_key_id        = var.kms_key_arn

  tags = var.tags
}

# Data source for VPC
data "aws_vpc" "main" {
  id = var.vpc_id
}