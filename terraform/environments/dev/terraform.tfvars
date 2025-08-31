# Development Environment Configuration
project_name = "strapi-platform"
environment  = "dev"
aws_region   = "us-east-1"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"

# General Purpose Node Group
general_node_desired_size = 2
general_node_max_size     = 5
general_node_min_size     = 1

# Memory Optimized Node Group
memory_node_desired_size = 1
memory_node_max_size     = 3
memory_node_min_size     = 0

# Aurora Configuration
aurora_engine_version         = "15.4"
aurora_instance_class        = "db.serverless"
aurora_backup_retention_days = 3

# Redis Configuration
redis_node_type      = "cache.t3.micro"
redis_num_cache_nodes = 1