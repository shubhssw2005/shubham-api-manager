/**
 * C++ Integration Middleware
 * Provides seamless integration between Node.js and C++ ultra-low-latency system
 */

import CppIntegrationService from '../lib/integration/CppIntegrationService.js';
import logger from '../lib/utils/logger.js';

let integrationService = null;

/**
 * Initialize C++ integration service
 */
export async function initializeCppIntegration(options = {}) {
  if (integrationService) {
    logger.warn('C++ integration service already initialized');
    return integrationService;
  }

  try {
    integrationService = new CppIntegrationService({
      cppSystemUrl: process.env.CPP_SYSTEM_URL || 'http://localhost:8080',
      redisHost: process.env.REDIS_HOST || 'localhost',
      redisPort: parseInt(process.env.REDIS_PORT) || 6379,
      redisPassword: process.env.REDIS_PASSWORD,
      instanceId: process.env.INSTANCE_ID || `nodejs-${process.pid}`,
      enableEventBridge: process.env.ENABLE_EVENT_BRIDGE !== 'false',
      enableSessionSync: process.env.ENABLE_SESSION_SYNC !== 'false',
      enableHealthCheck: process.env.ENABLE_HEALTH_CHECK !== 'false',
      ...options,
    });

    await integrationService.initialize();

    // Set up event handlers
    setupEventHandlers(integrationService);

    logger.info('C++ integration service initialized successfully');
    return integrationService;

  } catch (error) {
    logger.error('Failed to initialize C++ integration service:', error);
    throw error;
  }
}

/**
 * Set up event handlers for integration service
 */
function setupEventHandlers(service) {
  // Handle C++ system health changes
  service.on('cpp-system-healthy', () => {
    logger.info('C++ system is healthy - enabling high-performance routing');
  });

  service.on('cpp-system-unhealthy', () => {
    logger.warn('C++ system is unhealthy - falling back to Node.js');
  });

  // Handle cache invalidation events from C++
  service.on('cache-invalidation', (event) => {
    logger.debug('Received cache invalidation event:', event.payload);
    // Invalidate local caches if needed
    // This would integrate with your existing cache system
  });

  // Handle session events from C++
  service.on('session-event', (event) => {
    logger.debug('Received session event:', event.payload);
    // Handle session changes (logout, expiry, etc.)
  });

  // Handle performance metrics from C++
  service.on('performance-metric', (event) => {
    logger.debug('Received performance metric:', event.payload);
    // Forward to monitoring system if needed
  });

  // Handle errors from C++
  service.on('error-event', (event) => {
    logger.warn('Received error event from C++:', event.payload);
    // Handle error notifications
  });

  // Handle Redis connection issues
  service.on('redis-error', (error) => {
    logger.error('Redis connection error in C++ integration:', error);
  });
}

/**
 * Main C++ integration middleware
 */
export function cppIntegrationMiddleware() {
  return async (req, res, next) => {
    // Skip if integration service is not available
    if (!integrationService || !integrationService.isConnected) {
      return next();
    }

    try {
      // Add integration service to request context
      req.cppIntegration = integrationService;

      // Check if request should be routed to C++
      if (integrationService.shouldRouteToCpp(req)) {
        logger.debug(`Routing to C++: ${req.method} ${req.path}`);
        
        // Publish user action event for analytics
        if (req.user) {
          await integrationService.publishEvent(
            0, // USER_ACTION
            'user.actions',
            {
              user_id: req.user._id || req.user.id,
              action: req.method.toLowerCase(),
              resource: req.path,
              timestamp: Date.now(),
              ip: req.ip,
              user_agent: req.get('User-Agent'),
            }
          );
        }

        // Proxy to C++ system
        return await integrationService.proxyToCpp(req, res);
      }

      // Continue with Node.js processing
      next();

    } catch (error) {
      logger.error('Error in C++ integration middleware:', error);
      // Fall back to Node.js processing on error
      next();
    }
  };
}

/**
 * Session synchronization middleware
 */
export function sessionSyncMiddleware() {
  return async (req, res, next) => {
    if (!integrationService || !integrationService.isConnected) {
      return next();
    }

    try {
      // Sync session data to Redis for C++ access
      if (req.session && req.session.id) {
        const sessionData = {
          userId: req.user?._id || req.user?.id,
          tenantId: req.user?.tenantId || req.headers['x-tenant-id'],
          attributes: {
            userAgent: req.get('User-Agent'),
            ip: req.ip,
            lastActivity: Date.now(),
          },
          createdAt: req.session.createdAt || Date.now(),
          expiresAt: req.session.cookie?.expires || (Date.now() + 3600000),
          isAuthenticated: !!req.user,
          userRole: req.user?.role || 'guest',
          permissions: req.user?.permissions || [],
        };

        await integrationService.syncSession(req.session.id, sessionData);
      }

      next();

    } catch (error) {
      logger.error('Error in session sync middleware:', error);
      next();
    }
  };
}

/**
 * Event publishing helper middleware
 */
export function eventPublishingMiddleware() {
  return (req, res, next) => {
    if (!integrationService || !integrationService.isConnected) {
      return next();
    }

    // Add event publishing helper to response
    res.publishEvent = async (eventType, channel, payload, targetSystem = '') => {
      try {
        return await integrationService.publishEvent(eventType, channel, payload, targetSystem);
      } catch (error) {
        logger.error('Failed to publish event:', error);
        return null;
      }
    };

    // Add session management helpers
    res.syncSession = async (sessionId, sessionData) => {
      try {
        return await integrationService.syncSession(sessionId, sessionData);
      } catch (error) {
        logger.error('Failed to sync session:', error);
        return false;
      }
    };

    res.deleteSession = async (sessionId) => {
      try {
        return await integrationService.deleteSession(sessionId);
      } catch (error) {
        logger.error('Failed to delete session:', error);
        return false;
      }
    };

    next();
  };
}

/**
 * Performance monitoring middleware
 */
export function performanceMonitoringMiddleware() {
  return (req, res, next) => {
    if (!integrationService || !integrationService.isConnected) {
      return next();
    }

    const startTime = Date.now();

    // Override res.end to capture response metrics
    const originalEnd = res.end;
    res.end = function(...args) {
      const duration = Date.now() - startTime;
      const statusCode = res.statusCode;

      // Publish performance metric event
      integrationService.publishEvent(
        5, // PERFORMANCE_METRIC
        'performance.metrics',
        {
          metric_name: 'http_request_duration',
          value: duration,
          unit: 'milliseconds',
          tags: {
            method: req.method,
            endpoint: req.route?.path || req.path,
            status_code: statusCode.toString(),
            user_id: req.user?._id || req.user?.id || 'anonymous',
          },
        }
      ).catch(error => {
        logger.error('Failed to publish performance metric:', error);
      });

      // Call original end
      originalEnd.apply(this, args);
    };

    next();
  };
}

/**
 * Health check endpoint for C++ integration
 */
export function createHealthCheckEndpoint() {
  return (req, res) => {
    const stats = integrationService ? integrationService.getStats() : null;
    
    const health = {
      status: stats?.isConnected ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      integration: {
        connected: stats?.isConnected || false,
        cppSystemHealthy: stats?.cppSystemHealthy || false,
        redisConnected: stats?.redisConnected || false,
      },
      stats: stats || {},
    };

    const statusCode = health.status === 'healthy' ? 200 : 503;
    res.status(statusCode).json(health);
  };
}

/**
 * Metrics endpoint for integration statistics
 */
export function createMetricsEndpoint() {
  return (req, res) => {
    if (!integrationService) {
      return res.status(503).json({ error: 'Integration service not available' });
    }

    const stats = integrationService.getStats();
    
    res.json({
      service: 'nodejs-cpp-integration',
      timestamp: new Date().toISOString(),
      stats,
    });
  };
}

/**
 * Graceful shutdown handler
 */
export async function shutdownCppIntegration() {
  if (integrationService) {
    logger.info('Shutting down C++ integration service...');
    await integrationService.shutdown();
    integrationService = null;
    logger.info('C++ integration service shut down');
  }
}

/**
 * Get the integration service instance
 */
export function getCppIntegrationService() {
  return integrationService;
}

// Event type constants for convenience
export const EventTypes = {
  USER_ACTION: 0,
  SYSTEM_EVENT: 1,
  CACHE_INVALIDATION: 2,
  SESSION_EVENT: 3,
  MEDIA_PROCESSING: 4,
  PERFORMANCE_METRIC: 5,
  ERROR_EVENT: 6,
  CUSTOM: 7,
};

export default {
  initializeCppIntegration,
  cppIntegrationMiddleware,
  sessionSyncMiddleware,
  eventPublishingMiddleware,
  performanceMonitoringMiddleware,
  createHealthCheckEndpoint,
  createMetricsEndpoint,
  shutdownCppIntegration,
  getCppIntegrationService,
  EventTypes,
};