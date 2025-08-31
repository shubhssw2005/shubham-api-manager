bucket         = "strapi-platform-terraform-state-staging"
key            = "staging/terraform.tfstate"
region         = "us-east-1"
encrypt        = true
dynamodb_table = "strapi-platform-terraform-locks-staging"