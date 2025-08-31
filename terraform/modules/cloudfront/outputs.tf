# CloudFront Module Outputs

output "distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.id
}

output "distribution_arn" {
  description = "ARN of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.arn
}

output "distribution_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "distribution_hosted_zone_id" {
  description = "Hosted zone ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.hosted_zone_id
}

output "distribution_status" {
  description = "Current status of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.status
}

output "origin_access_control_id" {
  description = "ID of the Origin Access Control"
  value       = aws_cloudfront_origin_access_control.s3_oac.id
}

output "key_group_id" {
  description = "ID of the key group for signed URLs"
  value       = aws_cloudfront_key_group.signed_url_key_group.id
}

output "public_key_id" {
  description = "ID of the public key for signed URLs"
  value       = aws_cloudfront_public_key.signed_url_key.id
}

output "cache_policy_ids" {
  description = "Map of cache policy IDs"
  value = {
    media = aws_cloudfront_cache_policy.media_cache_policy.id
    api   = aws_cloudfront_cache_policy.api_cache_policy.id
  }
}

output "origin_request_policy_ids" {
  description = "Map of origin request policy IDs"
  value = {
    media = aws_cloudfront_origin_request_policy.media_origin_policy.id
    api   = aws_cloudfront_origin_request_policy.api_origin_policy.id
  }
}

output "response_headers_policy_id" {
  description = "ID of the response headers policy"
  value       = aws_cloudfront_response_headers_policy.security_headers.id
}