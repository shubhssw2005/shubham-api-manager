/**
 * Advanced Rate Limiting Middleware
 * Implements per-tenant rate limiting with Redis-backed counters
 * Supports multiple rate limiting strategies and sliding window
 */

import Redis from 'ioredis';
import { RateLimitError } from '../lib/errors/index.js';

class RateLimitingService {
  constructor(options = {}) {
    this.redis = new Redis.Cluster([
      { host: process.env.REDIS_HOST_1, port: 6379 },
      { host: process.env.REDIS_HOST_2, port: 6379 },
      { host: process.env.REDIS_HOST_3, port: 6379 }
    ], {
      enableOfflineQueue: false,
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100
    });

    // Default rate limit configurations
    this.rateLimits = {
      free: {
        requests: 1000,
        window: 3600,        // 1 hour
        burst: 50
      },
      pro: {
        requests: 10000,
        window: 3600,        // 1 hour
        burst: 200
      },
      enterprise: {
        requests: 100000,
        window: 3600,        // 1 hour
        burst: 1000
      }
    };

    // Override with custom configurations
    this.rateLimits = { ...this.rateLimits, ...options.rateLimits };
    this.keyPrefix = options.keyPrefix || 'rate_limit';
    this.skipSuccessfulRequests = options.skipSuccessfulRequests || false;
    this.skipFailedRequests = options.skipFailedRequests || false;
  }

  /**
   * Create rate limiting middleware
   */
  createMiddleware(options = {}) {
    return async (req, res, next) => {
      try {
        const result = await this.checkRateLimit(req, options);
        
        // Add rate limit headers
        res.set({
          'X-RateLimit-Limit': result.limit,
          'X-RateLimit-Remaining': result.remaining,
          'X-RateLimit-Reset': result.resetTime,
          'X-RateLimit-Window': result.window
        });

        if (!result.allowed) {
          throw new RateLimitError(
            `Rate limit exceeded. ${result.remaining} requests remaining. Reset in ${result.resetTime} seconds.`,
            {
              limit: result.limit,
              remaining: result.remaining,
              resetTime: result.resetTime,
              retryAfter: result.resetTime
            }
          );
        }

        // Store rate limit info in request for later use
        req.rateLimit = result;
        next();

      } catch (error) {
        if (error instanceof RateLimitError) {
          res.status(429).json({
            error: {
              code: 'RATE_LIMIT_EXCEEDED',
              message: error.message,
              details: error.details
            }
          });
        } else {
          console.error('Rate limiting error:', error);
          // Fail open - allow request if rate limiting fails
          next();
        }
      }
    };
  }

  /**
   * Check rate limit for a request
   */
  async checkRateLimit(req, options = {}) {
    const tenantId = this.getTenantId(req);
    const userId = this.getUserId(req);
    const endpoint = this.getEndpoint(req);
    
    // Determine rate limit configuration
    const config = this.getRateLimitConfig(req, options);
    
    // Create rate limit key
    const key = this.createRateLimitKey(tenantId, userId, endpoint, options);
    
    // Check different rate limiting strategies
    const results = await Promise.all([
      this.checkSlidingWindow(key, config),
      this.checkBurstLimit(key + ':burst', config),
      this.checkDailyLimit(key + ':daily', tenantId)
    ]);

    const [slidingWindow, burstLimit, dailyLimit] = results;

    // Return the most restrictive result
    const mostRestrictive = [slidingWindow, burstLimit, dailyLimit]
      .filter(result => !result.allowed)[0] || slidingWindow;

    return mostRestrictive;
  }

  /**
   * Sliding window rate limiting
   */
  async checkSlidingWindow(key, config) {
    const now = Date.now();
    const window = config.window * 1000; // Convert to milliseconds
    const windowStart = now - window;

    try {
      const pipeline = this.redis.pipeline();
      
      // Remove expired entries
      pipeline.zremrangebyscore(key, 0, windowStart);
      
      // Count current requests in window
      pipeline.zcard(key);
      
      // Add current request
      pipeline.zadd(key, now, `${now}-${Math.random()}`);
      
      // Set expiration
      pipeline.expire(key, Math.ceil(window / 1000));

      const results = await pipeline.exec();
      const currentCount = results[1][1];
      
      const remaining = Math.max(0, config.requests - currentCount);
      const resetTime = Math.ceil(window / 1000);

      return {
        allowed: currentCount < config.requests,
        remaining,
        resetTime,
        limit: config.requests,
        window: config.window,
        strategy: 'sliding_window'
      };

    } catch (error) {
      console.error('Sliding window rate limit check failed:', error);
      return this.getFailOpenResult(config);
    }
  }

  /**
   * Burst limit checking (short-term rate limiting)
   */
  async checkBurstLimit(key, config) {
    const burstWindow = 60; // 1 minute burst window
    const now = Math.floor(Date.now() / 1000);
    const windowStart = now - burstWindow;

    try {
      const pipeline = this.redis.pipeline();
      
      // Remove expired entries
      pipeline.zremrangebyscore(key, 0, windowStart);
      
      // Count current requests in burst window
      pipeline.zcard(key);
      
      // Add current request
      pipeline.zadd(key, now, `${now}-${Math.random()}`);
      
      // Set expiration
      pipeline.expire(key, burstWindow);

      const results = await pipeline.exec();
      const currentCount = results[1][1];
      
      const remaining = Math.max(0, config.burst - currentCount);

      return {
        allowed: currentCount < config.burst,
        remaining,
        resetTime: burstWindow,
        limit: config.burst,
        window: burstWindow,
        strategy: 'burst_limit'
      };

    } catch (error) {
      console.error('Burst limit check failed:', error);
      return this.getFailOpenResult(config);
    }
  }

  /**
   * Daily limit checking
   */
  async checkDailyLimit(key, tenantId) {
    const now = new Date();
    const dayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const dayKey = `${key}:${dayStart.getTime()}`;

    // Get tenant's daily limit based on plan
    const tenantPlan = await this.getTenantPlan(tenantId);
    const dailyLimit = this.getDailyLimit(tenantPlan);

    try {
      const pipeline = this.redis.pipeline();
      pipeline.incr(dayKey);
      pipeline.expire(dayKey, 86400); // 24 hours

      const results = await pipeline.exec();
      const currentCount = results[0][1];
      
      const remaining = Math.max(0, dailyLimit - currentCount);
      const resetTime = Math.ceil((dayStart.getTime() + 86400000 - now.getTime()) / 1000);

      return {
        allowed: currentCount <= dailyLimit,
        remaining,
        resetTime,
        limit: dailyLimit,
        window: 86400,
        strategy: 'daily_limit'
      };

    } catch (error) {
      console.error('Daily limit check failed:', error);
      return this.getFailOpenResult({ requests: dailyLimit });
    }
  }

  /**
   * Get tenant ID from request
   */
  getTenantId(req) {
    return req.user?.tenantId || 
           req.headers['x-tenant-id'] || 
           req.query.tenantId || 
           'anonymous';
  }

  /**
   * Get user ID from request
   */
  getUserId(req) {
    return req.user?.id || 
           req.headers['x-user-id'] || 
           req.ip || 
           'anonymous';
  }

  /**
   * Get endpoint identifier
   */
  getEndpoint(req) {
    return `${req.method}:${req.route?.path || req.path}`;
  }

  /**
   * Get rate limit configuration for request
   */
  getRateLimitConfig(req, options) {
    const tenantPlan = req.user?.plan || 'free';
    const customConfig = options.rateLimit;
    
    if (customConfig) {
      return customConfig;
    }

    return this.rateLimits[tenantPlan] || this.rateLimits.free;
  }

  /**
   * Create rate limit key
   */
  createRateLimitKey(tenantId, userId, endpoint, options) {
    const parts = [this.keyPrefix];
    
    if (options.perTenant !== false) {
      parts.push(`tenant:${tenantId}`);
    }
    
    if (options.perUser) {
      parts.push(`user:${userId}`);
    }
    
    if (options.perEndpoint) {
      parts.push(`endpoint:${endpoint}`);
    }

    return parts.join(':');
  }

  /**
   * Get tenant plan from cache or database
   */
  async getTenantPlan(tenantId) {
    try {
      const cached = await this.redis.get(`tenant_plan:${tenantId}`);
      if (cached) {
        return cached;
      }

      // Fallback to database lookup (implement based on your data layer)
      // const tenant = await Tenant.findById(tenantId);
      // const plan = tenant?.plan || 'free';
      
      const plan = 'free'; // Default fallback
      
      // Cache for 1 hour
      await this.redis.setex(`tenant_plan:${tenantId}`, 3600, plan);
      
      return plan;
    } catch (error) {
      console.error('Failed to get tenant plan:', error);
      return 'free';
    }
  }

  /**
   * Get daily limit based on plan
   */
  getDailyLimit(plan) {
    const dailyLimits = {
      free: 10000,
      pro: 100000,
      enterprise: 1000000
    };
    
    return dailyLimits[plan] || dailyLimits.free;
  }

  /**
   * Get fail-open result when rate limiting fails
   */
  getFailOpenResult(config) {
    return {
      allowed: true,
      remaining: config.requests || 1000,
      resetTime: config.window || 3600,
      limit: config.requests || 1000,
      window: config.window || 3600,
      strategy: 'fail_open'
    };
  }

  /**
   * Reset rate limit for a tenant/user
   */
  async resetRateLimit(tenantId, userId = null) {
    const pattern = userId 
      ? `${this.keyPrefix}:tenant:${tenantId}:user:${userId}*`
      : `${this.keyPrefix}:tenant:${tenantId}*`;
    
    const keys = await this.redis.keys(pattern);
    
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
    
    return keys.length;
  }

  /**
   * Get rate limit status
   */
  async getRateLimitStatus(tenantId, userId = null) {
    const key = this.createRateLimitKey(tenantId, userId, '', {});
    const config = this.rateLimits[await this.getTenantPlan(tenantId)];
    
    return await this.checkSlidingWindow(key, config);
  }
}

// Export middleware factory
export function createRateLimitMiddleware(options = {}) {
  const rateLimitService = new RateLimitingService(options);
  return rateLimitService.createMiddleware(options);
}

// Export service class
export { RateLimitingService };

// Export specific middleware configurations
export const rateLimitConfigs = {
  // Strict rate limiting for authentication endpoints
  auth: {
    rateLimit: { requests: 10, window: 300, burst: 5 }, // 10 requests per 5 minutes
    perUser: true,
    perEndpoint: true
  },
  
  // Moderate rate limiting for API endpoints
  api: {
    rateLimit: { requests: 1000, window: 3600, burst: 100 }, // 1000 requests per hour
    perTenant: true,
    perEndpoint: false
  },
  
  // Lenient rate limiting for media uploads
  media: {
    rateLimit: { requests: 100, window: 3600, burst: 20 }, // 100 uploads per hour
    perTenant: true,
    perUser: true
  },
  
  // Very strict rate limiting for admin endpoints
  admin: {
    rateLimit: { requests: 50, window: 3600, burst: 10 }, // 50 requests per hour
    perUser: true,
    perEndpoint: true
  }
};