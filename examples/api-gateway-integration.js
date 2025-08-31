/**
 * API Gateway Integration Example
 * Demonstrates how to set up the complete API Gateway system with:
 * - JWT validation
 * - Rate limiting
 * - Request validation
 * - AWS API Gateway integration
 */

import express from 'express';
import { createRateLimitMiddleware, rateLimitConfigs } from '../middleware/rateLimiting.js';
import { createRequestValidationMiddleware, validationConfigs } from '../middleware/requestValidation.js';
import { createAPIGatewayIntegration } from '../lib/apiGateway/integration.js';

const app = express();

// Basic Express setup
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Trust proxy (important for API Gateway)
app.set('trust proxy', true);

// Global middleware stack
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path} - ${req.ip}`);
  next();
});

// API Gateway integration
const apiGatewayApp = createAPIGatewayIntegration({
  rateLimits: {
    free: { requests: 1000, window: 3600, burst: 50 },
    pro: { requests: 10000, window: 3600, burst: 200 },
    enterprise: { requests: 100000, window: 3600, burst: 1000 }
  }
});

// Mount API Gateway routes
app.use('/api', apiGatewayApp);

// Example of custom route with specific middleware
app.use('/api/custom',
  // Custom rate limiting for this endpoint
  createRateLimitMiddleware({
    ...rateLimitConfigs.api,
    rateLimit: { requests: 500, window: 3600, burst: 25 }
  }),
  
  // Custom validation for this endpoint
  createRequestValidationMiddleware({
    ...validationConfigs.api,
    maxRequestSize: 5 * 1024 * 1024, // 5MB limit
    enableSqlInjectionCheck: true,
    enableXssCheck: true
  }),
  
  // Route handler
  (req, res) => {
    res.json({
      message: 'Custom endpoint with specific middleware',
      rateLimit: req.rateLimit,
      tenantId: req.tenantId,
      userId: req.userId
    });
  }
);

// Health check endpoint (no rate limiting)
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Metrics endpoint for monitoring
app.get('/metrics', (req, res) => {
  // In a real application, this would return Prometheus metrics
  res.set('Content-Type', 'text/plain');
  res.send(`
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",status="200"} 1000
api_requests_total{method="POST",status="200"} 500
api_requests_total{method="POST",status="400"} 50

# HELP api_request_duration_seconds API request duration
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{le="0.1"} 800
api_request_duration_seconds_bucket{le="0.5"} 950
api_request_duration_seconds_bucket{le="1.0"} 990
api_request_duration_seconds_bucket{le="+Inf"} 1000
api_request_duration_seconds_sum 450.5
api_request_duration_seconds_count 1000
  `.trim());
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('API Gateway Error:', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    headers: req.headers,
    body: req.body
  });

  // Rate limit errors
  if (error.name === 'RateLimitError') {
    return res.status(429).json({
      error: {
        code: 'RATE_LIMIT_EXCEEDED',
        message: error.message,
        details: error.details,
        retryAfter: error.details?.retryAfter
      }
    });
  }

  // Validation errors
  if (error.name === 'ValidationError') {
    return res.status(400).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: error.message,
        details: error.details
      }
    });
  }

  // JWT errors
  if (error.name === 'JsonWebTokenError') {
    return res.status(401).json({
      error: {
        code: 'INVALID_TOKEN',
        message: 'Invalid or expired token'
      }
    });
  }

  // Default error
  res.status(500).json({
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred',
      requestId: req.headers['x-request-id']
    }
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: 'Endpoint not found',
      path: req.originalUrl
    }
  });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`API Gateway server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log('Available endpoints:');
  console.log('  GET  /health - Health check');
  console.log('  GET  /metrics - Prometheus metrics');
  console.log('  *    /api/* - API Gateway routes');
  console.log('  *    /api/custom - Custom middleware example');
});

export default app;