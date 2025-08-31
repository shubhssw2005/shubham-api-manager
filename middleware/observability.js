// Comprehensive observability middleware for Express applications
import { initializeTracing, getTracing } from '../lib/observability/tracing.js';
import metricsCollector from '../lib/observability/metrics.js';
import sloMonitor from '../lib/observability/slo.js';
import { v4 as uuidv4 } from 'uuid';

class ObservabilityMiddleware {
  constructor(serviceName, serviceVersion = '1.0.0') {
    this.serviceName = serviceName;
    this.serviceVersion = serviceVersion;
    this.tracing = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    try {
      // Initialize tracing
      this.tracing = initializeTracing(this.serviceName, this.serviceVersion);
      
      // Initialize metrics collector
      metricsCollector.initialize();
      
      // Initialize SLO monitor
      sloMonitor.initialize();
      
      this.initialized = true;
      console.log(`Observability middleware initialized for ${this.serviceName}`);
    } catch (error) {
      console.error('Failed to initialize observability middleware:', error);
      throw error;
    }
  }

  // Main middleware function
  createMiddleware() {
    return async (req, res, next) => {
      if (!this.initialized) {
        await this.initialize();
      }

      const startTime = Date.now();
      const requestId = req.headers['x-request-id'] || uuidv4();
      const tenantId = req.headers['x-tenant-id'] || 'unknown';
      
      // Add request ID to headers for downstream services
      req.requestId = requestId;
      req.tenantId = tenantId;
      res.setHeader('x-request-id', requestId);

      // Create request span
      const span = this.tracing.createSpan(`${req.method} ${req.route?.path || req.path}`, {
        kind: this.tracing.tracer.SpanKind?.SERVER,
        attributes: {
          'http.method': req.method,
          'http.url': req.url,
          'http.scheme': req.protocol,
          'http.host': req.get('host'),
          'http.target': req.path,
          'http.user_agent': req.get('user-agent') || 'unknown',
          'http.request_id': requestId,
          'http.tenant_id': tenantId,
          'service.name': this.serviceName,
          'service.version': this.serviceVersion,
        },
      });

      // Set span as active for the request
      const activeContext = this.tracing.tracer.setSpan(
        this.tracing.tracer.active(),
        span
      );

      // Override res.end to capture response metrics
      const originalEnd = res.end;
      res.end = (...args) => {
        const duration = Date.now() - startTime;
        const statusCode = res.statusCode;
        const contentLength = res.getHeader('content-length') || 0;

        try {
          // Update span with response information
          span.setAttributes({
            'http.status_code': statusCode,
            'http.response.size': contentLength,
            'http.response.content_type': res.getHeader('content-type') || 'unknown',
          });

          // Set span status based on response
          if (statusCode >= 400) {
            span.setStatus({
              code: statusCode >= 500 ? 2 : 1, // ERROR : OK
              message: statusCode >= 500 ? 'Server Error' : 'Client Error',
            });
          } else {
            span.setStatus({ code: 1 }); // OK
          }

          // Record metrics
          this.recordRequestMetrics(req, res, duration, tenantId);

          // Record SLO metrics
          this.recordSLOMetrics(req, res, duration, tenantId);

        } catch (error) {
          console.error('Error recording observability metrics:', error);
          span.recordException(error);
        } finally {
          // End the span
          span.end();
        }

        // Call original end
        originalEnd.apply(res, args);
      };

      // Handle errors
      const originalNext = next;
      next = (error) => {
        if (error) {
          span.recordException(error);
          span.setStatus({
            code: 2, // ERROR
            message: error.message,
          });

          // Record error metrics
          metricsCollector.recordHttpRequest(
            req.method,
            req.route?.path || req.path,
            500,
            Date.now() - startTime,
            tenantId
          );
        }
        originalNext(error);
      };

      // Continue with request processing in the active context
      this.tracing.tracer.with(activeContext, () => {
        next();
      });
    };
  }

  recordRequestMetrics(req, res, duration, tenantId) {
    const route = req.route?.path || req.path;
    const method = req.method;
    const statusCode = res.statusCode;

    // Record HTTP request metrics
    metricsCollector.recordHttpRequest(method, route, statusCode, duration, tenantId);

    // Record tenant-specific metrics
    metricsCollector.recordTenantRequest(tenantId, route, method);

    // Record API token usage if present
    const apiToken = req.headers['authorization']?.replace('Bearer ', '');
    if (apiToken) {
      metricsCollector.recordApiTokenUsage(apiToken, tenantId, route);
    }
  }

  recordSLOMetrics(req, res, duration, tenantId) {
    const statusCode = res.statusCode;
    const isSuccess = statusCode < 500;
    const route = req.route?.path || req.path;

    // Emit SLO tracking event
    sloMonitor.emit('request_completed', {
      service: this.serviceName,
      success: isSuccess,
      duration: duration / 1000, // Convert to seconds
      statusCode,
      route,
      tenantId,
      method: req.method,
    });
  }

  // Database operation middleware
  createDatabaseMiddleware() {
    return (operation, table) => {
      return async (next) => {
        const startTime = Date.now();
        const span = this.tracing.createDatabaseSpan(operation, table);

        try {
          const result = await this.tracing.tracer.with(
            this.tracing.tracer.setSpan(this.tracing.tracer.active(), span),
            async () => await next()
          );

          const duration = Date.now() - startTime;
          
          // Record successful database operation
          metricsCollector.recordDatabaseQuery(operation, table, duration, true);
          
          span.setStatus({ code: 1 }); // OK
          return result;

        } catch (error) {
          const duration = Date.now() - startTime;
          
          // Record failed database operation
          metricsCollector.recordDatabaseQuery(operation, table, duration, false);
          
          span.recordException(error);
          span.setStatus({
            code: 2, // ERROR
            message: error.message,
          });
          
          throw error;
        } finally {
          span.end();
        }
      };
    };
  }

  // Cache operation middleware
  createCacheMiddleware() {
    return (operation, key) => {
      return async (next) => {
        const startTime = Date.now();
        const span = this.tracing.createCacheSpan(operation, key);

        try {
          const result = await this.tracing.tracer.with(
            this.tracing.tracer.setSpan(this.tracing.tracer.active(), span),
            async () => await next()
          );

          const duration = Date.now() - startTime;
          const hit = result !== null && result !== undefined;

          // Record cache metrics
          if (operation === 'get') {
            if (hit) {
              metricsCollector.recordCacheHit(operation, key);
            } else {
              metricsCollector.recordCacheMiss(operation, key);
            }
          }
          
          metricsCollector.recordCacheOperation(operation, duration, true);
          
          span.setAttributes({
            'cache.hit': hit,
            'cache.result.size': typeof result === 'string' ? result.length : 0,
          });
          
          span.setStatus({ code: 1 }); // OK
          return result;

        } catch (error) {
          const duration = Date.now() - startTime;
          
          metricsCollector.recordCacheOperation(operation, duration, false);
          
          span.recordException(error);
          span.setStatus({
            code: 2, // ERROR
            message: error.message,
          });
          
          throw error;
        } finally {
          span.end();
        }
      };
    };
  }

  // Media processing middleware
  createMediaProcessingMiddleware() {
    return (mediaType) => {
      return async (next) => {
        const startTime = Date.now();
        const span = this.tracing.createSpan(`media.process.${mediaType}`, {
          attributes: {
            'media.type': mediaType,
            'media.operation': 'process',
          },
        });

        try {
          const result = await this.tracing.tracer.with(
            this.tracing.tracer.setSpan(this.tracing.tracer.active(), span),
            async () => await next()
          );

          const duration = Date.now() - startTime;
          
          // Record successful media processing
          metricsCollector.recordMediaProcessing(mediaType, duration, true);
          
          span.setAttributes({
            'media.processing.success': true,
            'media.processing.duration': duration,
          });
          
          span.setStatus({ code: 1 }); // OK
          return result;

        } catch (error) {
          const duration = Date.now() - startTime;
          
          // Record failed media processing
          metricsCollector.recordMediaProcessing(mediaType, duration, false);
          
          span.recordException(error);
          span.setStatus({
            code: 2, // ERROR
            message: error.message,
          });
          
          throw error;
        } finally {
          span.end();
        }
      };
    };
  }

  // Health check endpoint
  createHealthCheckEndpoint() {
    return (req, res) => {
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: this.serviceName,
        version: this.serviceVersion,
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        slo: sloMonitor.getSLOSummary(),
      };

      res.status(200).json(health);
    };
  }

  // Metrics endpoint
  createMetricsEndpoint() {
    return (req, res) => {
      // This would typically be handled by the Prometheus exporter
      // but we can provide a summary endpoint
      const metrics = {
        service: this.serviceName,
        version: this.serviceVersion,
        timestamp: new Date().toISOString(),
        slos: sloMonitor.getAllSLOs().map(slo => ({
          id: slo.id,
          name: slo.name,
          compliance: slo.currentCompliance,
          errorBudgetRemaining: slo.errorBudgetRemaining,
          burnRate: slo.burnRate,
        })),
      };

      res.status(200).json(metrics);
    };
  }

  // Graceful shutdown
  async shutdown() {
    if (this.tracing) {
      await this.tracing.shutdown();
    }
    console.log(`Observability middleware shut down for ${this.serviceName}`);
  }
}

// Factory function for creating observability middleware
export function createObservabilityMiddleware(serviceName, serviceVersion) {
  return new ObservabilityMiddleware(serviceName, serviceVersion);
}

export default ObservabilityMiddleware;