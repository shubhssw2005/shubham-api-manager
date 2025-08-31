# Staging Environment Configuration
project_name = "strapi-platform"
environment  = "staging"
aws_region   = "us-east-1"

# VPC Configuration
vpc_cidr = "10.1.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"

# General Purpose Node Group
general_node_desired_size = 3
general_node_max_size     = 8
general_node_min_size     = 2

# Memory Optimized Node Group
memory_node_desired_size = 2
memory_node_max_size     = 4
memory_node_min_size     = 1

# Aurora Configuration
aurora_engine_version         = "15.4"
aurora_instance_class        = "db.serverless"
aurora_backup_retention_days = 5

# Redis Configuration
redis_node_type      = "cache.r6g.large"
redis_num_cache_nodes = 2