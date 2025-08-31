# Production Environment Configuration
project_name = "strapi-platform"
environment  = "production"
aws_region   = "us-east-1"

# VPC Configuration
vpc_cidr = "10.2.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"

# General Purpose Node Group
general_node_desired_size = 5
general_node_max_size     = 20
general_node_min_size     = 3

# Memory Optimized Node Group
memory_node_desired_size = 3
memory_node_max_size     = 10
memory_node_min_size     = 2

# Aurora Configuration
aurora_engine_version         = "15.4"
aurora_instance_class        = "db.serverless"
aurora_backup_retention_days = 30

# Redis Configuration
redis_node_type      = "cache.r6g.xlarge"
redis_num_cache_nodes = 3