#!/bin/bash

# Terraform Deployment Script
# Usage: ./deploy.sh <environment> [plan|apply|destroy]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if environment is provided
if [ -z "$1" ]; then
    print_error "Environment is required. Usage: ./deploy.sh <environment> [plan|apply|destroy]"
    print_error "Available environments: dev, staging, production"
    exit 1
fi

ENVIRONMENT=$1
ACTION=${2:-plan}

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    print_error "Available environments: dev, staging, production"
    exit 1
fi

# Validate action
if [[ ! "$ACTION" =~ ^(plan|apply|destroy)$ ]]; then
    print_error "Invalid action: $ACTION"
    print_error "Available actions: plan, apply, destroy"
    exit 1
fi

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$TERRAFORM_DIR/environments/$ENVIRONMENT"

print_status "Deploying to environment: $ENVIRONMENT"
print_status "Action: $ACTION"
print_status "Working directory: $TERRAFORM_DIR"

# Check if environment directory exists
if [ ! -d "$ENV_DIR" ]; then
    print_error "Environment directory not found: $ENV_DIR"
    exit 1
fi

# Change to terraform directory
cd "$TERRAFORM_DIR"

# Initialize Terraform with backend configuration
print_status "Initializing Terraform..."
terraform init -backend-config="$ENV_DIR/backend.hcl" -reconfigure

# Validate Terraform configuration
print_status "Validating Terraform configuration..."
terraform validate

# Format Terraform files
print_status "Formatting Terraform files..."
terraform fmt -recursive

# Execute the requested action
case $ACTION in
    plan)
        print_status "Creating Terraform plan..."
        terraform plan -var-file="$ENV_DIR/terraform.tfvars" -out="$ENV_DIR/terraform.plan"
        print_status "Plan created successfully. Review the plan above."
        print_status "To apply the plan, run: ./deploy.sh $ENVIRONMENT apply"
        ;;
    apply)
        if [ -f "$ENV_DIR/terraform.plan" ]; then
            print_status "Applying Terraform plan..."
            terraform apply "$ENV_DIR/terraform.plan"
            rm -f "$ENV_DIR/terraform.plan"
        else
            print_warning "No plan file found. Creating and applying plan..."
            terraform apply -var-file="$ENV_DIR/terraform.tfvars" -auto-approve
        fi
        print_status "Infrastructure deployed successfully!"
        
        # Show outputs
        print_status "Infrastructure outputs:"
        terraform output
        ;;
    destroy)
        print_warning "This will destroy all infrastructure in the $ENVIRONMENT environment!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            print_status "Destroying infrastructure..."
            terraform destroy -var-file="$ENV_DIR/terraform.tfvars" -auto-approve
            print_status "Infrastructure destroyed successfully!"
        else
            print_status "Destroy cancelled."
        fi
        ;;
esac

print_status "Operation completed successfully!"