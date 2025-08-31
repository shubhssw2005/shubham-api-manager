const jwt = require('jsonwebtoken');
const Redis = require('ioredis');

// Initialize Redis client
let redisClient;

const initRedis = () => {
  if (!redisClient) {
    redisClient = new Redis.Cluster([
      {
        host: process.env.REDIS_CLUSTER_ENDPOINT,
        port: 6379
      }
    ], {
      enableOfflineQueue: false,
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100,
      lazyConnect: true
    });
  }
  return redisClient;
};

// Rate limiting configuration
const RATE_LIMITS = {
  free: { requests: 1000, window: 3600 },     // 1000 requests per hour
  pro: { requests: 10000, window: 3600 },     // 10k requests per hour
  enterprise: { requests: 100000, window: 3600 } // 100k requests per hour
};

/**
 * JWT Authorizer Lambda Function
 * Validates JWT tokens and implements per-tenant rate limiting
 */
exports.handler = async (event) => {
  console.log('JWT Authorizer invoked:', JSON.stringify(event, null, 2));

  try {
    const token = extractToken(event.authorizationToken);
    
    if (!token) {
      throw new Error('No token provided');
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    if (!decoded || decoded.type !== 'access') {
      throw new Error('Invalid token type');
    }

    // Check if token is blacklisted (optional)
    const redis = initRedis();
    const isBlacklisted = await redis.get(`blacklist:${token}`);
    
    if (isBlacklisted) {
      throw new Error('Token is blacklisted');
    }

    // Perform rate limiting check
    const rateLimitResult = await checkRateLimit(redis, decoded);
    
    if (!rateLimitResult.allowed) {
      throw new Error(`Rate limit exceeded. Try again in ${rateLimitResult.resetTime} seconds`);
    }

    // Generate policy
    const policy = generatePolicy(decoded, 'Allow', event.methodArn);
    
    // Add context information
    policy.context = {
      userId: decoded.userId,
      tenantId: decoded.tenantId,
      role: decoded.role,
      permissions: JSON.stringify(decoded.permissions || []),
      rateLimitRemaining: rateLimitResult.remaining.toString(),
      rateLimitReset: rateLimitResult.resetTime.toString()
    };

    console.log('Authorization successful for user:', decoded.userId);
    return policy;

  } catch (error) {
    console.error('Authorization failed:', error.message);
    
    // Return deny policy
    return generatePolicy(null, 'Deny', event.methodArn);
  }
};

/**
 * Extract token from Authorization header
 */
function extractToken(authorizationToken) {
  if (!authorizationToken) {
    return null;
  }

  const parts = authorizationToken.split(' ');
  
  if (parts.length !== 2 || parts[0] !== 'Bearer') {
    return null;
  }

  return parts[1];
}

/**
 * Check rate limit for tenant
 */
async function checkRateLimit(redis, decoded) {
  const tenantId = decoded.tenantId;
  const plan = decoded.plan || 'free';
  const limits = RATE_LIMITS[plan] || RATE_LIMITS.free;
  
  const window = parseInt(process.env.RATE_LIMIT_WINDOW) || limits.window;
  const maxRequests = limits.requests;
  
  const now = Math.floor(Date.now() / 1000);
  const windowStart = Math.floor(now / window) * window;
  const windowEnd = windowStart + window;
  
  const key = `rate_limit:${tenantId}:${windowStart}`;
  
  try {
    // Use Redis pipeline for atomic operations
    const pipeline = redis.pipeline();
    pipeline.incr(key);
    pipeline.expire(key, window);
    
    const results = await pipeline.exec();
    const currentCount = results[0][1];
    
    const remaining = Math.max(0, maxRequests - currentCount);
    const resetTime = windowEnd - now;
    
    return {
      allowed: currentCount <= maxRequests,
      remaining,
      resetTime,
      limit: maxRequests
    };
    
  } catch (error) {
    console.error('Rate limit check failed:', error);
    
    // Allow request if Redis is unavailable (fail open)
    return {
      allowed: true,
      remaining: 1000,
      resetTime: 3600,
      limit: 1000
    };
  }
}

/**
 * Generate IAM policy
 */
function generatePolicy(principal, effect, resource) {
  const policy = {
    principalId: principal ? principal.userId : 'unknown',
    policyDocument: {
      Version: '2012-10-17',
      Statement: [
        {
          Action: 'execute-api:Invoke',
          Effect: effect,
          Resource: resource
        }
      ]
    }
  };

  return policy;
}

/**
 * Validate request size and content
 */
function validateRequest(event) {
  const maxSize = 10 * 1024 * 1024; // 10MB
  const contentLength = parseInt(event.headers['Content-Length'] || '0');
  
  if (contentLength > maxSize) {
    throw new Error(`Request size ${contentLength} exceeds maximum allowed size ${maxSize}`);
  }

  // Validate content type for POST/PUT requests
  const method = event.httpMethod;
  const contentType = event.headers['Content-Type'] || '';
  
  if (['POST', 'PUT', 'PATCH'].includes(method)) {
    const allowedTypes = [
      'application/json',
      'application/x-www-form-urlencoded',
      'multipart/form-data',
      'text/plain'
    ];
    
    const isValidType = allowedTypes.some(type => 
      contentType.toLowerCase().startsWith(type)
    );
    
    if (!isValidType) {
      throw new Error(`Invalid content type: ${contentType}`);
    }
  }

  return true;
}