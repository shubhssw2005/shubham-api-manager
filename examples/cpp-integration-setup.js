/**
 * Example: C++ Integration Setup
 * 
 * This example shows how to integrate the ultra-low-latency C++ system
 * with your existing Node.js application.
 */

import express from 'express';
import {
  initializeCppIntegration,
  cppIntegrationMiddleware,
  sessionSyncMiddleware,
  eventPublishingMiddleware,
  performanceMonitoringMiddleware,
  createHealthCheckEndpoint,
  createMetricsEndpoint,
  shutdownCppIntegration,
  EventTypes,
} from '../middleware/cppIntegration.js';
import logger from '../lib/utils/logger.js';

const app = express();

/**
 * Initialize C++ integration during application startup
 */
async function setupCppIntegration() {
  try {
    // Initialize the C++ integration service
    const integrationService = await initializeCppIntegration({
      // C++ system configuration
      cppSystemUrl: process.env.CPP_SYSTEM_URL || 'http://localhost:8080',
      
      // Redis configuration for shared state
      redisHost: process.env.REDIS_HOST || 'localhost',
      redisPort: parseInt(process.env.REDIS_PORT) || 6379,
      redisPassword: process.env.REDIS_PASSWORD,
      
      // Event bridge configuration
      enableEventBridge: true,
      subscriptionPatterns: ['*'], // Subscribe to all events
      
      // Session synchronization
      enableSessionSync: true,
      sessionSyncInterval: 30000, // 30 seconds
      
      // Health monitoring
      enableHealthCheck: true,
      healthCheckInterval: 10000, // 10 seconds
      
      // Fallback routes (keep these in Node.js)
      fallbackRoutes: [
        '/api/auth/*',
        '/api/users/*',
        '/api/admin/*',
        '/api/posts/*', // Blog posts stay in Node.js
        '/api/products/*', // E-commerce stays in Node.js
      ],
    });

    // Set up event handlers
    setupEventHandlers(integrationService);

    logger.info('C++ integration setup completed successfully');
    return integrationService;

  } catch (error) {
    logger.error('Failed to setup C++ integration:', error);
    throw error;
  }
}

/**
 * Set up event handlers for the integration service
 */
function setupEventHandlers(integrationService) {
  // Handle cache invalidation events from C++
  integrationService.on('cache-invalidation', async (event) => {
    const { cache_key_pattern, reason } = event.payload;
    
    logger.info(`Cache invalidation requested: ${cache_key_pattern} (${reason})`);
    
    // Invalidate local Node.js caches
    // This would integrate with your existing cache system
    // Example: await cacheManager.invalidate(cache_key_pattern);
  });

  // Handle session events from C++
  integrationService.on('session-event', async (event) => {
    const { session_id, event_type, user_id } = event.payload;
    
    logger.info(`Session event: ${event_type} for session ${session_id}`);
    
    if (event_type === 'expired' || event_type === 'deleted') {
      // Handle session cleanup in Node.js
      // Example: await sessionStore.destroy(session_id);
    }
  });

  // Handle media processing events from C++
  integrationService.on('media-processing', async (event) => {
    const { media_id, operation, status, metadata } = event.payload;
    
    logger.info(`Media processing: ${operation} ${status} for ${media_id}`);
    
    if (status === 'completed') {
      // Update database with processing results
      // Example: await Media.findByIdAndUpdate(media_id, { processed: true, ...metadata });
    } else if (status === 'failed') {
      // Handle processing failure
      logger.error(`Media processing failed for ${media_id}:`, metadata);
    }
  });

  // Handle performance metrics from C++
  integrationService.on('performance-metric', async (event) => {
    const { metric_name, value, unit, tags } = event.payload;
    
    // Forward to your monitoring system
    // Example: await metricsCollector.record(metric_name, value, tags);
    
    logger.debug(`Performance metric: ${metric_name} = ${value} ${unit}`);
  });

  // Handle error events from C++
  integrationService.on('error-event', async (event) => {
    const { error_type, error_message, component, context } = event.payload;
    
    logger.error(`C++ system error in ${component}: ${error_message}`, context);
    
    // Forward to error tracking system
    // Example: await errorTracker.captureException(new Error(error_message), { tags: context });
  });

  // Handle user actions from C++
  integrationService.on('user-action', async (event) => {
    const { user_id, action, resource, details } = event.payload;
    
    logger.debug(`User action: ${user_id} ${action} ${resource}`);
    
    // Log user actions for analytics
    // Example: await analyticsService.track(user_id, action, { resource, ...details });
  });
}

/**
 * Set up Express middleware for C++ integration
 */
function setupMiddleware(app) {
  // Performance monitoring (should be early in the middleware stack)
  app.use(performanceMonitoringMiddleware());

  // Session synchronization (after session middleware)
  app.use(sessionSyncMiddleware());

  // Event publishing helpers
  app.use(eventPublishingMiddleware());

  // C++ integration routing (should be after auth but before route handlers)
  app.use(cppIntegrationMiddleware());
}

/**
 * Set up health and metrics endpoints
 */
function setupEndpoints(app) {
  // Health check endpoint
  app.get('/api/cpp-integration/health', createHealthCheckEndpoint());

  // Metrics endpoint
  app.get('/api/cpp-integration/metrics', createMetricsEndpoint());

  // Manual cache invalidation endpoint
  app.post('/api/cpp-integration/invalidate-cache', async (req, res) => {
    try {
      const { pattern, reason = 'manual' } = req.body;
      
      if (!pattern) {
        return res.status(400).json({ error: 'Cache pattern is required' });
      }

      const eventId = await res.publishEvent(
        EventTypes.CACHE_INVALIDATION,
        'cache.invalidation',
        {
          cache_key_pattern: pattern,
          reason: reason,
          requested_by: req.user?.id || 'anonymous',
          timestamp: Date.now(),
        }
      );

      res.json({
        success: true,
        eventId,
        message: `Cache invalidation requested for pattern: ${pattern}`,
      });

    } catch (error) {
      logger.error('Failed to invalidate cache:', error);
      res.status(500).json({ error: 'Failed to invalidate cache' });
    }
  });

  // Manual session cleanup endpoint
  app.post('/api/cpp-integration/cleanup-sessions', async (req, res) => {
    try {
      const { userId, reason = 'manual' } = req.body;

      const eventId = await res.publishEvent(
        EventTypes.SESSION_EVENT,
        'session.cleanup',
        {
          user_id: userId,
          event_type: 'cleanup_requested',
          reason: reason,
          requested_by: req.user?.id || 'anonymous',
          timestamp: Date.now(),
        }
      );

      res.json({
        success: true,
        eventId,
        message: `Session cleanup requested for user: ${userId}`,
      });

    } catch (error) {
      logger.error('Failed to cleanup sessions:', error);
      res.status(500).json({ error: 'Failed to cleanup sessions' });
    }
  });
}

/**
 * Example route that publishes events to C++
 */
function setupExampleRoutes(app) {
  // Example: Media upload that notifies C++ system
  app.post('/api/media/upload', async (req, res) => {
    try {
      // Handle file upload (existing logic)
      const mediaId = 'media_' + Date.now();
      
      // Publish media processing event to C++
      const eventId = await res.publishEvent(
        EventTypes.MEDIA_PROCESSING,
        'media.processing',
        {
          media_id: mediaId,
          operation: 'upload',
          status: 'started',
          metadata: {
            filename: req.file?.originalname,
            size: req.file?.size,
            mimetype: req.file?.mimetype,
            user_id: req.user?.id,
          },
        }
      );

      res.json({
        success: true,
        mediaId,
        eventId,
        message: 'Media upload initiated',
      });

    } catch (error) {
      logger.error('Media upload failed:', error);
      res.status(500).json({ error: 'Media upload failed' });
    }
  });

  // Example: User login that syncs session
  app.post('/api/auth/login', async (req, res) => {
    try {
      // Handle authentication (existing logic)
      const user = { id: 'user_123', role: 'user', permissions: ['read', 'write'] };
      
      // Create session
      req.session.userId = user.id;
      req.session.createdAt = Date.now();
      
      // Sync session with C++ system
      await res.syncSession(req.session.id, {
        userId: user.id,
        tenantId: 'default',
        attributes: {
          loginTime: Date.now(),
          userAgent: req.get('User-Agent'),
          ip: req.ip,
        },
        createdAt: Date.now(),
        expiresAt: Date.now() + (24 * 60 * 60 * 1000), // 24 hours
        isAuthenticated: true,
        userRole: user.role,
        permissions: user.permissions,
      });

      // Publish login event
      await res.publishEvent(
        EventTypes.USER_ACTION,
        'user.actions',
        {
          user_id: user.id,
          action: 'login',
          resource: '/api/auth/login',
          timestamp: Date.now(),
          details: {
            ip: req.ip,
            user_agent: req.get('User-Agent'),
          },
        }
      );

      res.json({
        success: true,
        user,
        message: 'Login successful',
      });

    } catch (error) {
      logger.error('Login failed:', error);
      res.status(500).json({ error: 'Login failed' });
    }
  });

  // Example: User logout that cleans up session
  app.post('/api/auth/logout', async (req, res) => {
    try {
      const userId = req.user?.id;
      const sessionId = req.session.id;

      // Destroy session in Node.js
      req.session.destroy();

      // Delete session from shared storage
      await res.deleteSession(sessionId);

      // Publish logout event
      await res.publishEvent(
        EventTypes.SESSION_EVENT,
        'session.events',
        {
          session_id: sessionId,
          event_type: 'logout',
          user_id: userId,
          timestamp: Date.now(),
        }
      );

      res.json({
        success: true,
        message: 'Logout successful',
      });

    } catch (error) {
      logger.error('Logout failed:', error);
      res.status(500).json({ error: 'Logout failed' });
    }
  });
}

/**
 * Graceful shutdown handler
 */
function setupGracefulShutdown() {
  const shutdown = async (signal) => {
    logger.info(`Received ${signal}, shutting down gracefully...`);
    
    try {
      await shutdownCppIntegration();
      process.exit(0);
    } catch (error) {
      logger.error('Error during shutdown:', error);
      process.exit(1);
    }
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

/**
 * Main application setup
 */
async function main() {
  try {
    // Initialize C++ integration
    await setupCppIntegration();

    // Set up middleware
    setupMiddleware(app);

    // Set up endpoints
    setupEndpoints(app);

    // Set up example routes
    setupExampleRoutes(app);

    // Set up graceful shutdown
    setupGracefulShutdown();

    // Start server
    const port = process.env.PORT || 3005;
    app.listen(port, () => {
      logger.info(`Server running on port ${port} with C++ integration`);
    });

  } catch (error) {
    logger.error('Failed to start application with C++ integration:', error);
    process.exit(1);
  }
}

// Export for use in other files
export {
  setupCppIntegration,
  setupEventHandlers,
  setupMiddleware,
  setupEndpoints,
  setupExampleRoutes,
  setupGracefulShutdown,
};

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}