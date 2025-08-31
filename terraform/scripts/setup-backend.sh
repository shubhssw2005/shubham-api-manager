#!/bin/bash

# Script to set up Terraform backend resources (S3 bucket and DynamoDB table)
# Usage: ./setup-backend.sh <environment>

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
    print_error "Environment is required. Usage: ./setup-backend.sh <environment>"
    print_error "Available environments: dev, staging, production"
    exit 1
fi

ENVIRONMENT=$1
PROJECT_NAME="strapi-platform"
AWS_REGION="us-east-1"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    print_error "Available environments: dev, staging, production"
    exit 1
fi

BUCKET_NAME="${PROJECT_NAME}-terraform-state-${ENVIRONMENT}"
DYNAMODB_TABLE="${PROJECT_NAME}-terraform-locks-${ENVIRONMENT}"

print_status "Setting up Terraform backend for environment: $ENVIRONMENT"
print_status "S3 Bucket: $BUCKET_NAME"
print_status "DynamoDB Table: $DYNAMODB_TABLE"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials are not configured. Please run 'aws configure' first."
    exit 1
fi

# Create S3 bucket for Terraform state
print_status "Creating S3 bucket for Terraform state..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    print_warning "S3 bucket $BUCKET_NAME already exists"
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION" 2>/dev/null || \
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$AWS_REGION"
    
    print_status "S3 bucket $BUCKET_NAME created successfully"
fi

# Enable versioning on the S3 bucket
print_status "Enabling versioning on S3 bucket..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Enable server-side encryption on the S3 bucket
print_status "Enabling server-side encryption on S3 bucket..."
aws s3api put-bucket-encryption \
    --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'

# Block public access to the S3 bucket
print_status "Blocking public access to S3 bucket..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Create DynamoDB table for Terraform state locking
print_status "Creating DynamoDB table for Terraform state locking..."
if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE" --region "$AWS_REGION" &>/dev/null; then
    print_warning "DynamoDB table $DYNAMODB_TABLE already exists"
else
    aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
        --region "$AWS_REGION"
    
    print_status "DynamoDB table $DYNAMODB_TABLE created successfully"
    
    # Wait for table to be active
    print_status "Waiting for DynamoDB table to be active..."
    aws dynamodb wait table-exists --table-name "$DYNAMODB_TABLE" --region "$AWS_REGION"
fi

print_status "Terraform backend setup completed successfully!"
print_status "You can now run: ./deploy.sh $ENVIRONMENT plan"