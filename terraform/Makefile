# Terraform Infrastructure Management Makefile

.PHONY: help init plan apply destroy fmt validate clean setup-backend

# Default environment
ENV ?= dev

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup-backend: ## Set up Terraform backend (S3 + DynamoDB)
	@echo "$(GREEN)Setting up Terraform backend for $(ENV) environment...$(NC)"
	@./scripts/setup-backend.sh $(ENV)

init: ## Initialize Terraform
	@echo "$(GREEN)Initializing Terraform for $(ENV) environment...$(NC)"
	@terraform init -backend-config=environments/$(ENV)/backend.hcl -reconfigure

validate: ## Validate Terraform configuration
	@echo "$(GREEN)Validating Terraform configuration...$(NC)"
	@terraform validate

fmt: ## Format Terraform files
	@echo "$(GREEN)Formatting Terraform files...$(NC)"
	@terraform fmt -recursive

plan: init validate fmt ## Create Terraform plan
	@echo "$(GREEN)Creating Terraform plan for $(ENV) environment...$(NC)"
	@terraform plan -var-file=environments/$(ENV)/terraform.tfvars -out=environments/$(ENV)/terraform.plan

apply: ## Apply Terraform plan
	@echo "$(GREEN)Applying Terraform plan for $(ENV) environment...$(NC)"
	@if [ -f "environments/$(ENV)/terraform.plan" ]; then \
		terraform apply environments/$(ENV)/terraform.plan; \
		rm -f environments/$(ENV)/terraform.plan; \
	else \
		echo "$(RED)No plan file found. Run 'make plan ENV=$(ENV)' first.$(NC)"; \
		exit 1; \
	fi

deploy: plan apply ## Plan and apply in one command
	@echo "$(GREEN)Deployment completed for $(ENV) environment!$(NC)"

destroy: init ## Destroy infrastructure
	@echo "$(RED)WARNING: This will destroy all infrastructure in $(ENV) environment!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ]
	@terraform destroy -var-file=environments/$(ENV)/terraform.tfvars -auto-approve

output: ## Show Terraform outputs
	@terraform output

state-list: ## List all resources in Terraform state
	@terraform state list

clean: ## Clean up temporary files
	@echo "$(GREEN)Cleaning up temporary files...$(NC)"
	@rm -f environments/*/terraform.plan
	@rm -rf .terraform/
	@rm -f .terraform.lock.hcl

# Environment-specific shortcuts
dev-setup: ## Set up development environment
	@$(MAKE) setup-backend ENV=dev

dev-deploy: ## Deploy to development
	@$(MAKE) deploy ENV=dev

dev-destroy: ## Destroy development environment
	@$(MAKE) destroy ENV=dev

staging-setup: ## Set up staging environment
	@$(MAKE) setup-backend ENV=staging

staging-deploy: ## Deploy to staging
	@$(MAKE) deploy ENV=staging

staging-destroy: ## Destroy staging environment
	@$(MAKE) destroy ENV=staging

prod-setup: ## Set up production environment
	@$(MAKE) setup-backend ENV=production

prod-deploy: ## Deploy to production
	@$(MAKE) deploy ENV=production

prod-destroy: ## Destroy production environment (DANGEROUS!)
	@$(MAKE) destroy ENV=production

# Utility commands
check-aws: ## Check AWS credentials and configuration
	@echo "$(GREEN)Checking AWS configuration...$(NC)"
	@aws sts get-caller-identity
	@aws configure list

check-tools: ## Check if required tools are installed
	@echo "$(GREEN)Checking required tools...$(NC)"
	@command -v terraform >/dev/null 2>&1 || { echo "$(RED)Terraform is not installed$(NC)"; exit 1; }
	@command -v aws >/dev/null 2>&1 || { echo "$(RED)AWS CLI is not installed$(NC)"; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "$(RED)kubectl is not installed$(NC)"; exit 1; }
	@echo "$(GREEN)All required tools are installed!$(NC)"

update-kubeconfig: ## Update kubectl configuration for EKS cluster
	@echo "$(GREEN)Updating kubeconfig for $(ENV) environment...$(NC)"
	@aws eks update-kubeconfig --region us-east-1 --name strapi-platform-$(ENV)

# Documentation
docs: ## Generate Terraform documentation
	@echo "$(GREEN)Generating Terraform documentation...$(NC)"
	@terraform-docs markdown table --output-file README.md --output-mode inject .

# Security
security-scan: ## Run security scan on Terraform code
	@echo "$(GREEN)Running security scan...$(NC)"
	@command -v tfsec >/dev/null 2>&1 || { echo "$(YELLOW)tfsec not installed. Install with: brew install tfsec$(NC)"; exit 1; }
	@tfsec .

# Cost estimation
cost-estimate: ## Estimate infrastructure costs
	@echo "$(GREEN)Estimating infrastructure costs...$(NC)"
	@command -v infracost >/dev/null 2>&1 || { echo "$(YELLOW)infracost not installed. See: https://www.infracost.io/docs/$(NC)"; exit 1; }
	@infracost breakdown --path .

# Examples
example-dev: ## Example: Full development deployment
	@echo "$(GREEN)Example: Deploying development environment$(NC)"
	@$(MAKE) check-tools
	@$(MAKE) check-aws
	@$(MAKE) dev-setup
	@$(MAKE) dev-deploy
	@$(MAKE) update-kubeconfig ENV=dev

example-staging: ## Example: Full staging deployment
	@echo "$(GREEN)Example: Deploying staging environment$(NC)"
	@$(MAKE) check-tools
	@$(MAKE) check-aws
	@$(MAKE) staging-setup
	@$(MAKE) staging-deploy
	@$(MAKE) update-kubeconfig ENV=staging