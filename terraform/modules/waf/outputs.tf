output "web_acl_id" {
  description = "The ID of the WAF WebACL"
  value       = aws_wafv2_web_acl.main.id
}

output "web_acl_arn" {
  description = "The ARN of the WAF WebACL"
  value       = aws_wafv2_web_acl.main.arn
}

output "web_acl_name" {
  description = "The name of the WAF WebACL"
  value       = aws_wafv2_web_acl.main.name
}

output "log_group_name" {
  description = "The name of the CloudWatch log group for WAF logs"
  value       = aws_cloudwatch_log_group.waf_log_group.name
}

output "shield_protection_ids" {
  description = "The IDs of Shield Advanced protections"
  value = {
    cloudfront = var.enable_shield_advanced ? aws_shield_protection.cloudfront[0].id : null
    alb        = var.enable_shield_advanced ? aws_shield_protection.alb[0].id : null
  }
}