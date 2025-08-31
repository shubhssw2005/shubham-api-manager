# AWS Infrastructure as Code

This directory contains Terraform modules and configurations for deploying the Strapi-like API platform infrastructure on AWS.

## Architecture Overview

The infrastructure includes:

- **VPC**: Multi-AZ VPC with public and private subnets
- **EKS**: Kubernetes cluster with auto-scaling node groups
- **Aurora PostgreSQL**: Serverless v2 database with read replicas
- **ElastiCache Redis**: Redis cluster for caching
- **S3**: Buckets for media storage, backups, and logs
- **KMS**: Encryption keys for all services
- **IAM**: Roles and policies for multi-tenant isolation

## Directory Structure

```
terraform/
├── main.tf                 # Main Terraform configuration
├── variables.tf            # Input variables
├── outputs.tf             # Output values
├── modules/               # Reusable Terraform modules
│   ├── kms/              # KMS encryption module
│   ├── vpc/              # VPC networking module
│   ├── eks/              # EKS cluster module
│   ├── aurora/           # Aurora PostgreSQL module
│   ├── redis/            # ElastiCache Redis module
│   ├── s3/               # S3 storage module
│   └── iam/              # IAM roles and policies module
├── environments/          # Environment-specific configurations
│   ├── dev/
│   ├── staging/
│   └── production/
└── scripts/              # Deployment and utility scripts
    ├── deploy.sh         # Main deployment script
    └── setup-backend.sh  # Backend setup script
```

## Prerequisites

1. **AWS CLI**: Install and configure AWS CLI with appropriate credentials
   ```bash
   aws configure
   ```

2. **Terraform**: Install Terraform >= 1.0
   ```bash
   # macOS
   brew install terraform
   
   # Or download from https://www.terraform.io/downloads.html
   ```

3. **kubectl**: Install kubectl for Kubernetes management
   ```bash
   # macOS
   brew install kubectl
   ```

## Quick Start

### 1. Set up Terraform Backend

Before deploying infrastructure, set up the S3 bucket and DynamoDB table for Terraform state management:

```bash
# Set up backend for development environment
./scripts/setup-backend.sh dev

# Set up backend for staging environment
./scripts/setup-backend.sh staging

# Set up backend for production environment
./scripts/setup-backend.sh production
```

### 2. Deploy Infrastructure

```bash
# Plan deployment for development environment
./scripts/deploy.sh dev plan

# Apply the plan
./scripts/deploy.sh dev apply

# For other environments
./scripts/deploy.sh staging plan
./scripts/deploy.sh staging apply

./scripts/deploy.sh production plan
./scripts/deploy.sh production apply
```

### 3. Configure kubectl

After EKS cluster deployment, configure kubectl to connect to the cluster:

```bash
# Update kubeconfig for the deployed cluster
aws eks update-kubeconfig --region us-east-1 --name strapi-platform-dev

# Verify connection
kubectl get nodes
```

## Environment Configuration

Each environment has its own configuration in the `environments/` directory:

### Development (`environments/dev/`)
- Minimal resources for development and testing
- Single-node Redis cluster
- Smaller EKS node groups
- 3-day backup retention

### Staging (`environments/staging/`)
- Production-like setup with reduced capacity
- Multi-node Redis cluster
- Medium-sized EKS node groups
- 5-day backup retention

### Production (`environments/production/`)
- Full production setup with high availability
- Multi-node Redis cluster with failover
- Large EKS node groups with auto-scaling
- 30-day backup retention
- Cross-region replication enabled

## Security Features

### Multi-Tenant Isolation
- **S3 Access**: Tenant-specific IAM policies restrict access to `tenants/{tenant-id}/` prefixes
- **Database**: Tenant-based sharding with row-level security
- **API**: JWT tokens include tenant context for authorization

### Encryption
- **At Rest**: All data encrypted using AWS KMS
- **In Transit**: TLS 1.3 for all communications
- **Secrets**: Stored in AWS Secrets Manager with automatic rotation

### Network Security
- **VPC**: Private subnets for all compute resources
- **Security Groups**: Least-privilege access rules
- **NACLs**: Additional network-level protection

## Monitoring and Observability

The infrastructure includes:
- **CloudWatch**: Centralized logging and metrics
- **VPC Flow Logs**: Network traffic monitoring
- **EKS Control Plane Logs**: Kubernetes API audit logs
- **Aurora Performance Insights**: Database performance monitoring

## Backup and Disaster Recovery

### Automated Backups
- **Aurora**: Point-in-time recovery with configurable retention
- **S3**: Cross-region replication for critical data
- **EKS**: Persistent volume snapshots

### Disaster Recovery
- **Multi-AZ**: All services deployed across multiple availability zones
- **Cross-Region**: Production data replicated to secondary region
- **Automated Failover**: Aurora and Redis support automatic failover

## Cost Optimization

### Resource Optimization
- **Spot Instances**: Memory-optimized node group uses spot instances
- **S3 Intelligent Tiering**: Automatic storage class transitions
- **Aurora Serverless v2**: Automatic scaling based on demand

### Monitoring
- **Cost Allocation Tags**: All resources tagged for cost tracking
- **Lifecycle Policies**: Automatic cleanup of old backups and logs

## Troubleshooting

### Common Issues

1. **Backend Setup Fails**
   ```bash
   # Ensure AWS credentials are configured
   aws sts get-caller-identity
   
   # Check if bucket name is globally unique
   aws s3api head-bucket --bucket your-bucket-name
   ```

2. **EKS Node Groups Fail to Launch**
   ```bash
   # Check if you have sufficient EC2 limits
   aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
   ```

3. **Aurora Connection Issues**
   ```bash
   # Verify security group rules allow connections from EKS
   aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
   ```

### Useful Commands

```bash
# View Terraform state
terraform show

# List all resources
terraform state list

# Get specific output values
terraform output eks_cluster_endpoint

# Refresh state
terraform refresh -var-file=environments/dev/terraform.tfvars

# Import existing resources
terraform import aws_s3_bucket.example bucket-name
```

## Cleanup

To destroy infrastructure:

```bash
# Destroy development environment
./scripts/deploy.sh dev destroy

# Destroy staging environment
./scripts/deploy.sh staging destroy

# Destroy production environment (use with caution!)
./scripts/deploy.sh production destroy
```

**Warning**: Destroying production infrastructure will result in data loss. Ensure you have proper backups before proceeding.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review AWS CloudWatch logs for error details
3. Consult the Terraform documentation
4. Contact the platform team

## Contributing

When making changes to the infrastructure:
1. Test changes in development environment first
2. Run `terraform fmt` to format code
3. Run `terraform validate` to check syntax
4. Update documentation as needed
5. Follow the change management process for production deployments