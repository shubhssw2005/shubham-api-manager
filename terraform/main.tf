# Main Terraform configuration for AWS deployment system
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "s3" {
    # Backend configuration will be provided via backend config files
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# CloudFront requires certificates to be in us-east-1
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
  account_id        = data.aws_caller_identity.current.account_id
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# KMS Key for encryption
module "kms" {
  source = "./modules/kms"
  
  project_name = var.project_name
  environment  = var.environment
  
  tags = local.common_tags
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr          = var.vpc_cidr
  availability_zones = local.availability_zones
  
  tags = local.common_tags
}

# EKS Module
module "eks" {
  source = "./modules/eks"
  
  project_name         = var.project_name
  environment          = var.environment
  kubernetes_version   = var.kubernetes_version
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  public_subnet_ids   = module.vpc.public_subnet_ids
  kms_key_arn         = module.kms.key_arn
  
  # Node group configurations
  general_node_desired_size = var.general_node_desired_size
  general_node_max_size     = var.general_node_max_size
  general_node_min_size     = var.general_node_min_size
  
  memory_node_desired_size = var.memory_node_desired_size
  memory_node_max_size     = var.memory_node_max_size
  memory_node_min_size     = var.memory_node_min_size
  
  tags = local.common_tags
}

# Aurora Postgres Module
module "aurora" {
  source = "./modules/aurora"
  
  project_name           = var.project_name
  environment            = var.environment
  vpc_id                = module.vpc.vpc_id
  subnet_ids            = module.vpc.private_subnet_ids
  kms_key_arn           = module.kms.key_arn
  
  engine_version        = var.aurora_engine_version
  instance_class        = var.aurora_instance_class
  backup_retention_days = var.aurora_backup_retention_days
  
  tags = local.common_tags
}

# Redis ElastiCache Module
module "redis" {
  source = "./modules/redis"
  
  project_name    = var.project_name
  environment     = var.environment
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids
  kms_key_arn    = module.kms.key_arn
  
  node_type      = var.redis_node_type
  num_cache_nodes = var.redis_num_cache_nodes
  
  tags = local.common_tags
}

# S3 Module
module "s3" {
  source = "./modules/s3"
  
  project_name = var.project_name
  environment  = var.environment
  kms_key_arn  = module.kms.key_arn
  
  tags = local.common_tags
}

# IAM Module for multi-tenant isolation
module "iam" {
  source = "./modules/iam"
  
  project_name    = var.project_name
  environment     = var.environment
  eks_cluster_arn = module.eks.cluster_arn
  s3_bucket_arns  = module.s3.bucket_arns
  
  tags = local.common_tags
}

# SSL Certificate for CloudFront (must be in us-east-1)
module "acm_certificate" {
  source = "./modules/acm"
  
  providers = {
    aws = aws.us_east_1
  }
  
  project_name     = var.project_name
  environment      = var.environment
  domain_names     = var.cloudfront_custom_domains
  route53_zone_id  = var.route53_zone_id
}

# CloudFront Distribution
module "cloudfront" {
  source = "./modules/cloudfront"
  
  project_name           = var.project_name
  environment            = var.environment
  s3_bucket_name         = module.s3.media_bucket_name
  s3_bucket_domain_name  = module.s3.media_bucket_domain_name
  custom_domains         = var.cloudfront_custom_domains
  ssl_certificate_arn    = module.acm_certificate.certificate_arn
  logging_bucket         = module.s3.logging_bucket_name
  aws_account_id         = data.aws_caller_identity.current.account_id
  public_key_content     = var.cloudfront_public_key_content
  
  depends_on = [module.s3, module.acm_certificate]
}# Securi
ty modules
module "waf" {
  source = "./modules/waf"
  
  project_name                = var.project_name
  environment                 = var.environment
  enable_shield_advanced      = var.enable_shield_advanced
  blocked_countries          = var.blocked_countries
  cloudfront_distribution_arn = module.cloudfront.distribution_arn
  alb_arn                    = module.eks.alb_arn
  sns_topic_arn              = var.sns_topic_arn
}

module "secrets_manager" {
  source = "./modules/secrets-manager"
  
  project_name        = var.project_name
  environment         = var.environment
  aws_region          = var.aws_region
  replica_region      = var.replica_region
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  sns_topic_arn       = var.sns_topic_arn
  
  # Database credentials
  db_username = var.db_username
  db_password = var.db_password
  
  # JWT secrets
  jwt_access_secret   = var.jwt_access_secret
  jwt_refresh_secret  = var.jwt_refresh_secret
  jwt_encryption_key  = var.jwt_encryption_key
  
  # External API keys
  stripe_secret_key     = var.stripe_secret_key
  sendgrid_api_key      = var.sendgrid_api_key
  cloudflare_api_token  = var.cloudflare_api_token
  github_webhook_secret = var.github_webhook_secret
}

module "security_scanning" {
  source = "./modules/security-scanning"
  
  project_name       = var.project_name
  environment        = var.environment
  aws_region         = var.aws_region
  sns_topic_arn      = var.sns_topic_arn
  slack_webhook_url  = var.slack_webhook_url
}