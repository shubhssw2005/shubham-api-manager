bucket         = "strapi-platform-terraform-state-production"
key            = "production/terraform.tfstate"
region         = "us-east-1"
encrypt        = true
dynamodb_table = "strapi-platform-terraform-locks-production"