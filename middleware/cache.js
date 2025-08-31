import CacheService from '../lib/cache/index.js';

// Global cache service instance
let cacheService = null;

/**
 * Initialize cache service
 */
export function initializeCacheService(options = {}) {
  if (cacheService) {
    console.warn('Cache service already initialized');
    return cacheService;
  }

  const defaultOptions = {
    // Memory cache settings
    memoryTTL: parseInt(process.env.CACHE_MEMORY_TTL) || 300,
    memoryMaxKeys: parseInt(process.env.CACHE_MEMORY_MAX_KEYS) || 10000,
    
    // Redis cache settings
    redisTTL: parseInt(process.env.CACHE_REDIS_TTL) || 3600,
    redisKeyPrefix: process.env.CACHE_REDIS_PREFIX || 'app:cache:',
    
    // Invalidation strategies
    invalidationStrategies: {
      immediate: process.env.CACHE_INVALIDATION_IMMEDIATE !== 'false',
      delayed: process.env.CACHE_INVALIDATION_DELAYED === 'true',
      ttlBased: process.env.CACHE_INVALIDATION_TTL_BASED !== 'false',
      tagBased: process.env.CACHE_INVALIDATION_TAG_BASED !== 'false'
    },
    
    // Warming strategies
    warmingStrategies: {
      startup: process.env.CACHE_WARMING_STARTUP !== 'false',
      scheduled: process.env.CACHE_WARMING_SCHEDULED !== 'false',
      predictive: process.env.CACHE_WARMING_PREDICTIVE === 'true',
      onDemand: process.env.CACHE_WARMING_ON_DEMAND !== 'false'
    }
  };

  cacheService = new CacheService({ ...defaultOptions, ...options });
  
  // Setup event logging
  setupCacheEventLogging(cacheService);
  
  console.log('Cache service initialized');
  return cacheService;
}

/**
 * Get cache service instance
 */
export function getCacheService() {
  if (!cacheService) {
    throw new Error('Cache service not initialized. Call initializeCacheService() first.');
  }
  return cacheService;
}

/**
 * Setup cache event logging
 */
function setupCacheEventLogging(cache) {
  cache.on('cache:hit', (data) => {
    if (process.env.CACHE_DEBUG === 'true') {
      console.log(`Cache HIT [${data.layer}]: ${data.key} (${data.duration}ms)`);
    }
  });

  cache.on('cache:miss', (data) => {
    if (process.env.CACHE_DEBUG === 'true') {
      console.log(`Cache MISS: ${data.key} (${data.duration}ms)`);
    }
  });

  cache.on('cache:error', (data) => {
    console.error(`Cache ERROR [${data.operation}]: ${data.key}`, data.error);
  });

  cache.on('invalidation:processed', (data) => {
    if (process.env.CACHE_DEBUG === 'true') {
      console.log(`Cache INVALIDATED: ${data.reason} (${data.invalidatedCount} keys, ${data.duration}ms)`);
    }
  });

  cache.on('warming:completed', (data) => {
    console.log(`Cache WARMED [${data.type}]: ${data.totalWarmed} entries (${data.duration}ms)`);
  });
}

/**
 * Cache middleware for Express
 * Provides caching functionality to request handlers
 */
export function cacheMiddleware(options = {}) {
  return (req, res, next) => {
    const cache = getCacheService();
    
    // Add cache methods to request object
    req.cache = {
      get: cache.get.bind(cache),
      set: cache.set.bind(cache),
      del: cache.del.bind(cache),
      getOrSet: cache.getOrSet.bind(cache),
      invalidate: cache.invalidate.bind(cache),
      setWithTags: cache.setWithTags.bind(cache),
      invalidateByTags: cache.invalidateByTags.bind(cache)
    };

    // Add tenant-specific cache helpers
    if (req.user && req.user.tenantId) {
      req.cache.tenant = {
        get: (key, opts) => cache.get(`tenant:${req.user.tenantId}:${key}`, opts),
        set: (key, value, ttl, opts) => cache.set(`tenant:${req.user.tenantId}:${key}`, value, ttl, opts),
        del: (key) => cache.del(`tenant:${req.user.tenantId}:${key}`),
        invalidate: (pattern) => cache.invalidate(`tenant:${req.user.tenantId}:${pattern}`),
        warm: (priority) => cache.warmTenant(req.user.tenantId, priority)
      };
    }

    next();
  };
}

/**
 * Response caching middleware
 * Automatically cache GET responses based on configuration
 */
export function responseCacheMiddleware(options = {}) {
  const {
    ttl = 300,
    keyGenerator = (req) => `response:${req.method}:${req.originalUrl}`,
    shouldCache = (req, res) => req.method === 'GET' && res.statusCode === 200,
    skipPatterns = ['/health', '/metrics', '/admin']
  } = options;

  return async (req, res, next) => {
    // Skip caching for certain patterns
    if (skipPatterns.some(pattern => req.path.startsWith(pattern))) {
      return next();
    }

    const cache = getCacheService();
    const cacheKey = keyGenerator(req);

    // Try to get cached response
    if (req.method === 'GET') {
      try {
        const cachedResponse = await cache.get(cacheKey);
        if (cachedResponse) {
          res.set(cachedResponse.headers);
          return res.status(cachedResponse.status).json(cachedResponse.body);
        }
      } catch (error) {
        console.error('Error retrieving cached response:', error);
      }
    }

    // Intercept response to cache it
    const originalJson = res.json;
    res.json = function(body) {
      // Cache the response if conditions are met
      if (shouldCache(req, res)) {
        const responseData = {
          status: res.statusCode,
          headers: res.getHeaders(),
          body: body
        };
        
        cache.set(cacheKey, responseData, ttl).catch(error => {
          console.error('Error caching response:', error);
        });
      }
      
      return originalJson.call(this, body);
    };

    next();
  };
}

/**
 * Model event middleware
 * Automatically emit cache invalidation events for model changes
 */
export function modelEventMiddleware() {
  return (req, res, next) => {
    const cache = getCacheService();
    
    // Add model event emitter to request
    req.emitModelEvent = (eventName, data) => {
      cache.emitModelEvent(eventName, data);
    };

    // Intercept response to emit events based on operations
    const originalJson = res.json;
    res.json = function(body) {
      // Emit events based on the operation
      if (req.method === 'POST' && res.statusCode === 201) {
        // Created
        const modelName = extractModelNameFromPath(req.path);
        if (modelName && body) {
          req.emitModelEvent(`${modelName}:created`, body);
        }
      } else if (req.method === 'PUT' && res.statusCode === 200) {
        // Updated
        const modelName = extractModelNameFromPath(req.path);
        if (modelName && body) {
          req.emitModelEvent(`${modelName}:updated`, body);
        }
      } else if (req.method === 'DELETE' && res.statusCode === 200) {
        // Deleted
        const modelName = extractModelNameFromPath(req.path);
        if (modelName && body) {
          req.emitModelEvent(`${modelName}:deleted`, body);
        }
      }
      
      return originalJson.call(this, body);
    };

    next();
  };
}

/**
 * Extract model name from API path
 */
function extractModelNameFromPath(path) {
  const matches = path.match(/\/api\/([^\/]+)/);
  return matches ? matches[1].slice(0, -1) : null; // Remove 's' from plural
}

/**
 * Cache health check endpoint
 */
export async function cacheHealthCheck(req, res) {
  try {
    const cache = getCacheService();
    const health = await cache.healthCheck();
    
    res.status(health.healthy ? 200 : 503).json(health);
  } catch (error) {
    res.status(503).json({
      healthy: false,
      error: error.message,
      timestamp: Date.now()
    });
  }
}

/**
 * Cache statistics endpoint
 */
export async function cacheStats(req, res) {
  try {
    const cache = getCacheService();
    const stats = cache.getStats();
    
    res.json({
      success: true,
      stats,
      timestamp: Date.now()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: Date.now()
    });
  }
}

export default {
  initializeCacheService,
  getCacheService,
  cacheMiddleware,
  responseCacheMiddleware,
  modelEventMiddleware,
  cacheHealthCheck,
  cacheStats
};