# API Gateway Implementation

This document describes the implementation of Task 8: "API Gateway and Rate Limiting" from the AWS deployment system specification.

## Overview

The API Gateway implementation provides:

1. **AWS API Gateway Configuration** with JWT validation and request/response transformation
2. **Per-tenant Rate Limiting** with Redis-backed counters
3. **Request Size Limits** and content validation
4. **Security Features** including SQL injection and XSS protection

## Architecture

### Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AWS API       │    │   Lambda JWT     │    │   Application   │
│   Gateway       │───▶│   Authorizer     │───▶│   Backend       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐             │
         │              │   Redis Cluster  │             │
         │              │  (Rate Limiting) │             │
         │              └──────────────────┘             │
         │                                                │
         ▼                                                ▼
┌─────────────────┐                            ┌─────────────────┐
│   CloudWatch    │                            │   Request       │
│   Logs          │                            │   Validation    │
└─────────────────┘                            └─────────────────┘
```

## Implementation Details

### 1. AWS API Gateway Configuration

#### Terraform Module Structure
```
terraform/modules/api-gateway/
├── main.tf                 # Main API Gateway resources
├── jwt-authorizer.tf       # JWT authorizer Lambda
├── variables.tf            # Input variables
├── outputs.tf              # Output values
└── lambda/
    └── jwt-authorizer/
        ├── index.js        # Lambda function code
        └── package.json    # Dependencies
```

#### Key Features

- **Regional API Gateway** with custom domain support
- **JWT Authorizer** Lambda function for token validation
- **Request Validation** with size limits and content type checking
- **CORS Configuration** with customizable origins and headers
- **Usage Plans** with rate limiting and quotas
- **CloudWatch Logging** with structured log format
- **X-Ray Tracing** for distributed tracing

### 2. JWT Authorizer Lambda

The JWT authorizer validates tokens and implements rate limiting:

```javascript
// Key features of the JWT authorizer
const authorizer = {
  validateJWT: (token) => {
    // Verify JWT signature and expiration
    // Check token blacklist in Redis
    // Extract user and tenant context
  },
  
  checkRateLimit: (tenantId, plan) => {
    // Sliding window rate limiting
    // Per-tenant limits based on plan
    // Burst protection
    // Daily quotas
  },
  
  generatePolicy: (user, effect, resource) => {
    // IAM policy for API Gateway
    // Include user context in policy
  }
};
```

#### Rate Limiting Strategies

1. **Sliding Window**: Precise rate limiting over time windows
2. **Burst Protection**: Short-term spike protection
3. **Daily Quotas**: Long-term usage limits
4. **Per-Tenant**: Isolated limits per tenant
5. **Plan-Based**: Different limits for free/pro/enterprise

### 3. Application-Level Middleware

#### Rate Limiting Middleware

```javascript
import { createRateLimitMiddleware } from './middleware/rateLimiting.js';

// Basic usage
app.use(createRateLimitMiddleware());

// Custom configuration
app.use('/api/uploads', createRateLimitMiddleware({
  rateLimit: { requests: 100, window: 3600, burst: 20 },
  perTenant: true,
  perUser: true
}));
```

#### Request Validation Middleware

```javascript
import { createRequestValidationMiddleware } from './middleware/requestValidation.js';

// Basic usage
app.use(createRequestValidationMiddleware());

// Custom configuration
app.use('/api/admin', createRequestValidationMiddleware({
  maxRequestSize: 1024 * 1024, // 1MB
  enableSqlInjectionCheck: true,
  enableXssCheck: true,
  allowedContentTypes: ['application/json']
}));
```

## Configuration

### Environment Variables

```bash
# JWT Configuration
JWT_SECRET=your-jwt-secret-key
JWT_REFRESH_SECRET=your-refresh-secret-key

# Redis Configuration
REDIS_HOST_1=redis-cluster-node-1.cache.amazonaws.com
REDIS_HOST_2=redis-cluster-node-2.cache.amazonaws.com
REDIS_HOST_3=redis-cluster-node-3.cache.amazonaws.com

# Rate Limiting
RATE_LIMIT_WINDOW=3600
LOG_LEVEL=INFO

# API Gateway
API_GATEWAY_STAGE=production
BACKEND_URL=https://your-backend-service.com
```

### Terraform Variables

```hcl
# terraform/environments/production/terraform.tfvars
project_name = "your-project"
environment = "production"

# API Gateway Configuration
rate_limit_per_second = 1000
burst_limit = 2000
monthly_quota_limit = 10000000

# Security Configuration
allowed_ip_ranges = ["0.0.0.0/0"]
cors_allowed_origins = ["https://your-frontend.com"]

# Lambda Configuration
jwt_secret = "your-jwt-secret"
redis_cluster_endpoint = "your-redis-cluster.cache.amazonaws.com"
```

## Usage Examples

### 1. Basic API Gateway Setup

```javascript
import { createAPIGatewayIntegration } from './lib/apiGateway/integration.js';

const app = createAPIGatewayIntegration({
  rateLimits: {
    free: { requests: 1000, window: 3600, burst: 50 },
    pro: { requests: 10000, window: 3600, burst: 200 },
    enterprise: { requests: 100000, window: 3600, burst: 1000 }
  }
});
```

### 2. Custom Route with Specific Limits

```javascript
app.use('/api/heavy-operation',
  createRateLimitMiddleware({
    rateLimit: { requests: 10, window: 3600, burst: 2 },
    perUser: true
  }),
  createRequestValidationMiddleware({
    maxRequestSize: 50 * 1024 * 1024, // 50MB
    allowedContentTypes: ['application/json', 'multipart/form-data']
  }),
  heavyOperationHandler
);
```

### 3. Tenant-Specific Configuration

```javascript
// Different limits per tenant plan
const getTenantLimits = (plan) => {
  const limits = {
    free: { requests: 1000, burst: 50 },
    pro: { requests: 10000, burst: 200 },
    enterprise: { requests: 100000, burst: 1000 }
  };
  return limits[plan] || limits.free;
};
```

## Monitoring and Observability

### CloudWatch Metrics

The API Gateway automatically publishes metrics to CloudWatch:

- `Count`: Number of API calls
- `Latency`: Request latency (P50, P90, P95, P99)
- `4XXError`: Client errors
- `5XXError`: Server errors
- `CacheHitCount`: Cache hits
- `CacheMissCount`: Cache misses

### Custom Metrics

```javascript
// Rate limiting metrics
const rateLimitMetrics = {
  requests_blocked: 'Number of requests blocked by rate limiting',
  rate_limit_exceeded: 'Number of rate limit violations',
  tenant_quota_usage: 'Per-tenant quota usage percentage'
};

// Validation metrics
const validationMetrics = {
  requests_rejected: 'Number of requests rejected by validation',
  sql_injection_attempts: 'Number of SQL injection attempts detected',
  xss_attempts: 'Number of XSS attempts detected'
};
```

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "requestId": "abc123-def456-ghi789",
  "tenantId": "tenant-123",
  "userId": "user-456",
  "method": "POST",
  "path": "/api/posts",
  "statusCode": 200,
  "duration": 150,
  "rateLimitRemaining": 995,
  "userAgent": "MyApp/1.0"
}
```

## Security Features

### 1. JWT Validation

- **Signature Verification**: RSA/HMAC signature validation
- **Expiration Check**: Token expiration validation
- **Blacklist Check**: Redis-based token blacklist
- **Issuer Validation**: Trusted issuer verification

### 2. Rate Limiting Security

- **DDoS Protection**: Burst and sustained rate limiting
- **Tenant Isolation**: Per-tenant rate limiting
- **Fail-Safe**: Fail-open when Redis is unavailable
- **Attack Mitigation**: Automatic blocking of abusive IPs

### 3. Request Validation

- **Size Limits**: Configurable request size limits
- **Content Type**: Whitelist of allowed content types
- **SQL Injection**: Pattern-based SQL injection detection
- **XSS Protection**: Script and HTML injection detection
- **Header Validation**: Security header validation

## Performance Considerations

### 1. Rate Limiting Performance

- **Redis Cluster**: Distributed rate limiting with high availability
- **Sliding Window**: Efficient time-based rate limiting
- **Connection Pooling**: Optimized Redis connections
- **Caching**: In-memory caching for frequently accessed data

### 2. Validation Performance

- **Early Validation**: Fail fast on invalid requests
- **Regex Optimization**: Efficient pattern matching
- **Streaming Validation**: Large request handling
- **Async Processing**: Non-blocking validation

### 3. Lambda Performance

- **Cold Start Optimization**: Minimal dependencies
- **Connection Reuse**: Persistent Redis connections
- **Memory Optimization**: Right-sized Lambda memory
- **Timeout Configuration**: Appropriate timeout settings

## Deployment

### 1. Terraform Deployment

```bash
# Initialize Terraform
cd terraform/environments/production
terraform init

# Plan deployment
terraform plan -var-file="terraform.tfvars"

# Apply changes
terraform apply -var-file="terraform.tfvars"
```

### 2. Lambda Deployment

```bash
# Build Lambda package
cd terraform/modules/api-gateway/lambda/jwt-authorizer
npm install --production
zip -r ../jwt-authorizer.zip .

# Deploy with Terraform
terraform apply -target=aws_lambda_function.jwt_authorizer
```

### 3. Application Deployment

```bash
# Build application
npm run build

# Deploy to EKS/ECS
kubectl apply -f k8s/api-gateway/
```

## Testing

### 1. Unit Tests

```bash
# Run middleware tests
npm test -- tests/middleware/rateLimiting.test.js
npm test -- tests/middleware/requestValidation.test.js
```

### 2. Integration Tests

```bash
# Test API Gateway integration
npm test -- tests/integration/api-gateway.test.js
```

### 3. Load Testing

```bash
# K6 load testing
k6 run tests/load/api-gateway-load-test.js
```

## Troubleshooting

### Common Issues

1. **Rate Limit False Positives**
   - Check Redis connectivity
   - Verify tenant ID extraction
   - Review rate limit configuration

2. **JWT Validation Failures**
   - Verify JWT secret configuration
   - Check token expiration
   - Validate token format

3. **Request Validation Errors**
   - Review content type headers
   - Check request size limits
   - Validate security patterns

### Debugging

```bash
# Check API Gateway logs
aws logs tail /aws/apigateway/your-api-gateway --follow

# Check Lambda logs
aws logs tail /aws/lambda/jwt-authorizer --follow

# Check Redis connectivity
redis-cli -h your-redis-cluster.cache.amazonaws.com ping
```

## Requirements Compliance

This implementation satisfies the following requirements:

- **Requirement 2.2**: Automated CI/CD with JWT validation
- **Requirement 4.2**: Multi-tenant security with JWT and rate limiting
- **Requirement 4.5**: Request size limits and validation
- **Requirement 7.4**: Cost-efficient scaling with usage-based limits

## Next Steps

1. **Enhanced Monitoring**: Implement custom CloudWatch dashboards
2. **Advanced Security**: Add WAF rules and DDoS protection
3. **Performance Optimization**: Implement caching strategies
4. **Multi-Region**: Deploy across multiple AWS regions
5. **Compliance**: Add audit logging and compliance reporting