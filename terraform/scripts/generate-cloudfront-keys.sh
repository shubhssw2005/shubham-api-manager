#!/bin/bash

# Script to generate CloudFront key pair for signed URLs
# This script generates RSA key pairs required for CloudFront signed URLs

set -e

# Configuration
KEY_SIZE=2048
PRIVATE_KEY_FILE="cloudfront-private-key.pem"
PUBLIC_KEY_FILE="cloudfront-public-key.pem"
ENV_FILE="../environments/${ENVIRONMENT:-dev}/terraform.tfvars"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Generating CloudFront key pair for signed URLs...${NC}"

# Check if OpenSSL is installed
if ! command -v openssl &> /dev/null; then
    echo -e "${RED}Error: OpenSSL is required but not installed.${NC}"
    exit 1
fi

# Create keys directory if it doesn't exist
mkdir -p keys

# Generate private key
echo -e "${YELLOW}Generating private key...${NC}"
openssl genrsa -out "keys/${PRIVATE_KEY_FILE}" ${KEY_SIZE}

# Generate public key
echo -e "${YELLOW}Generating public key...${NC}"
openssl rsa -in "keys/${PRIVATE_KEY_FILE}" -pubout -out "keys/${PUBLIC_KEY_FILE}"

# Set appropriate permissions
chmod 600 "keys/${PRIVATE_KEY_FILE}"
chmod 644 "keys/${PUBLIC_KEY_FILE}"

echo -e "${GREEN}Key pair generated successfully!${NC}"
echo -e "Private key: ${YELLOW}keys/${PRIVATE_KEY_FILE}${NC}"
echo -e "Public key: ${YELLOW}keys/${PUBLIC_KEY_FILE}${NC}"

# Base64 encode the keys for environment variables
PRIVATE_KEY_B64=$(base64 -i "keys/${PRIVATE_KEY_FILE}" | tr -d '\n')
PUBLIC_KEY_CONTENT=$(cat "keys/${PUBLIC_KEY_FILE}")

echo -e "\n${GREEN}Environment Variables:${NC}"
echo -e "${YELLOW}Add these to your environment configuration:${NC}"
echo ""
echo "# CloudFront Signed URL Keys"
echo "CLOUDFRONT_PRIVATE_KEY=\"${PRIVATE_KEY_B64}\""
echo ""
echo "# For Terraform (add to ${ENV_FILE}):"
echo "cloudfront_public_key_content = <<EOF"
echo "${PUBLIC_KEY_CONTENT}"
echo "EOF"

# Optionally update terraform.tfvars if environment is specified
if [ ! -z "${ENVIRONMENT}" ] && [ -f "${ENV_FILE}" ]; then
    echo -e "\n${YELLOW}Updating ${ENV_FILE}...${NC}"
    
    # Remove existing cloudfront_public_key_content if it exists
    sed -i '/^cloudfront_public_key_content/,/^EOF$/d' "${ENV_FILE}"
    
    # Add new public key content
    echo "" >> "${ENV_FILE}"
    echo "# CloudFront public key for signed URLs" >> "${ENV_FILE}"
    echo "cloudfront_public_key_content = <<EOF" >> "${ENV_FILE}"
    cat "keys/${PUBLIC_KEY_FILE}" >> "${ENV_FILE}"
    echo "EOF" >> "${ENV_FILE}"
    
    echo -e "${GREEN}Updated ${ENV_FILE} with public key content${NC}"
fi

echo -e "\n${GREEN}Next steps:${NC}"
echo "1. Add the CLOUDFRONT_PRIVATE_KEY environment variable to your application"
echo "2. Update your Terraform variables with the public key content"
echo "3. Apply Terraform configuration to create CloudFront distribution"
echo "4. Store the private key securely (e.g., AWS Secrets Manager)"

echo -e "\n${YELLOW}Security Notes:${NC}"
echo "- Keep the private key secure and never commit it to version control"
echo "- Consider using AWS Secrets Manager or similar for production"
echo "- Rotate keys regularly for enhanced security"
echo "- The private key is required for generating signed URLs"

echo -e "\n${GREEN}Key generation completed successfully!${NC}"