output "kms_key_id" {
  description = "The ID of the KMS key used for secrets encryption"
  value       = aws_kms_key.secrets.id
}

output "kms_key_arn" {
  description = "The ARN of the KMS key used for secrets encryption"
  value       = aws_kms_key.secrets.arn
}

output "db_master_password_secret_arn" {
  description = "The ARN of the database master password secret"
  value       = aws_secretsmanager_secret.db_master_password.arn
}

output "jwt_keys_secret_arn" {
  description = "The ARN of the JWT keys secret"
  value       = aws_secretsmanager_secret.jwt_keys.arn
}

output "api_keys_secret_arn" {
  description = "The ARN of the API keys secret"
  value       = aws_secretsmanager_secret.api_keys.arn
}

output "rotation_lambda_arn" {
  description = "The ARN of the secrets rotation Lambda function"
  value       = aws_lambda_function.rotate_secrets.arn
}

output "secret_names" {
  description = "Map of secret names for application configuration"
  value = {
    db_master_password = aws_secretsmanager_secret.db_master_password.name
    jwt_keys          = aws_secretsmanager_secret.jwt_keys.name
    api_keys          = aws_secretsmanager_secret.api_keys.name
  }
}