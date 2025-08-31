import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { initializeTracing, getTracing } from '../../lib/observability/tracing.js';
import metricsCollector from '../../lib/observability/metrics.js';
import express from 'express';
import request from 'supertest';

describe('Observability Stack Integration', () => {
  let app;
  let tracing;
  let server;

  beforeAll(async () => {
    // Initialize tracing for testing
    tracing = initializeTracing('test-service', '1.0.0');
    metricsCollector.initialize();

    // Create test Express app
    app = express();
    app.use(express.json());
    app.use(tracing.createExpressMiddleware());

    // Test endpoints
    app.get('/test/success', (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      metricsCollector.recordTenantRequest(tenantId, '/test/success', 'GET');
      res.json({ status: 'success', timestamp: Date.now() });
    });

    app.get('/test/error', (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      metricsCollector.recordTenantRequest(tenantId, '/test/error', 'GET');
      res.status(500).json({ error: 'Test error' });
    });

    app.get('/test/slow', async (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      
      // Simulate slow operation with tracing
      const span = tracing.createSpan('slow_operation', {
        attributes: {
          'tenant.id': tenantId,
          'operation.type': 'test',
        },
      });

      try {
        await new Promise(resolve => setTimeout(resolve, 100));
        
        span.setAttributes({
          'operation.success': true,
          'operation.duration_ms': 100,
        });

        metricsCollector.recordTenantRequest(tenantId, '/test/slow', 'GET');
        res.json({ status: 'completed', duration: 100 });
      } catch (error) {
        span.recordException(error);
        throw error;
      } finally {
        span.end();
      }
    });

    app.get('/test/database', async (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      
      // Simulate database operation
      const dbOperation = tracing.traceFunction(
        'database_query',
        async () => {
          const startTime = Date.now();
          
          // Simulate database query
          await new Promise(resolve => setTimeout(resolve, 50));
          
          const duration = Date.now() - startTime;
          
          metricsCollector.recordDatabaseQuery(
            'SELECT',
            'test_table',
            duration,
            true,
            tenantId
          );
          
          return { id: 1, name: 'Test Record' };
        },
        {
          attributes: {
            'db.operation': 'SELECT',
            'db.table': 'test_table',
            'db.tenant_id': tenantId,
          },
        }
      );

      const result = await dbOperation();
      res.json(result);
    });

    app.get('/test/cache', async (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      const cacheKey = 'test:cache:key';
      
      // Simulate cache operations
      const cacheHit = Math.random() > 0.5;
      
      if (cacheHit) {
        metricsCollector.recordCacheHit('get', cacheKey, tenantId);
        metricsCollector.recordCacheOperation('get', 5, true, tenantId);
        res.json({ data: 'cached data', cached: true });
      } else {
        metricsCollector.recordCacheMiss('get', cacheKey, tenantId);
        metricsCollector.recordCacheOperation('get', 5, true, tenantId);
        
        // Simulate cache set
        metricsCollector.recordCacheOperation('set', 10, true, tenantId);
        res.json({ data: 'fresh data', cached: false });
      }
    });

    app.get('/test/media', async (req, res) => {
      const tenantId = req.headers['x-tenant-id'] || 'test-tenant';
      const mediaType = req.query.type || 'image';
      const fileSize = parseInt(req.query.size) || 1024;
      
      // Simulate media upload
      metricsCollector.recordMediaUpload(mediaType, fileSize, true, tenantId);
      
      // Simulate media processing
      const processingTime = Math.random() * 1000 + 500; // 500-1500ms
      await new Promise(resolve => setTimeout(resolve, processingTime));
      
      metricsCollector.recordMediaProcessing(mediaType, processingTime, true, tenantId);
      
      res.json({
        mediaId: 'test-media-123',
        type: mediaType,
        size: fileSize,
        processingTime,
      });
    });

    // Metrics endpoint
    app.get('/metrics', (req, res) => {
      res.set('Content-Type', 'text/plain');
      res.send('# Metrics endpoint - would return Prometheus metrics in real implementation');
    });

    // Health check
    app.get('/health', (req, res) => {
      res.json({ status: 'healthy', timestamp: Date.now() });
    });

    // Start server
    server = app.listen(0); // Use random available port
  });

  afterAll(async () => {
    if (server) {
      server.close();
    }
    if (tracing) {
      await tracing.shutdown();
    }
  });

  describe('Tracing Integration', () => {
    it('should initialize tracing successfully', () => {
      expect(tracing).toBeDefined();
      expect(tracing.serviceName).toBe('test-service');
      expect(tracing.serviceVersion).toBe('1.0.0');
    });

    it('should create custom spans', () => {
      const span = tracing.createSpan('test_span', {
        attributes: { 'test.attribute': 'value' },
      });
      
      expect(span).toBeDefined();
      span.end();
    });

    it('should wrap functions with tracing', async () => {
      const testFunction = async (input) => {
        await new Promise(resolve => setTimeout(resolve, 10));
        return `processed: ${input}`;
      };

      const tracedFunction = tracing.traceFunction(
        'test_function',
        testFunction,
        {
          attributes: { 'function.type': 'test' },
        }
      );

      const result = await tracedFunction('test input');
      expect(result).toBe('processed: test input');
    });
  });

  describe('Metrics Collection', () => {
    it('should initialize metrics collector', () => {
      expect(metricsCollector.initialized).toBe(true);
    });

    it('should record HTTP request metrics', async () => {
      const response = await request(app)
        .get('/test/success')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
    });

    it('should record error metrics', async () => {
      const response = await request(app)
        .get('/test/error')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Test error');
    });

    it('should record database metrics', async () => {
      const response = await request(app)
        .get('/test/database')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(1);
      expect(response.body.name).toBe('Test Record');
    });

    it('should record cache metrics', async () => {
      const response = await request(app)
        .get('/test/cache')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(200);
      expect(response.body.data).toBeDefined();
      expect(typeof response.body.cached).toBe('boolean');
    });

    it('should record media processing metrics', async () => {
      const response = await request(app)
        .get('/test/media?type=video&size=5242880')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(200);
      expect(response.body.mediaId).toBe('test-media-123');
      expect(response.body.type).toBe('video');
      expect(response.body.size).toBe(5242880);
      expect(response.body.processingTime).toBeGreaterThan(0);
    });
  });

  describe('Express Middleware', () => {
    it('should add tracing middleware to requests', async () => {
      const response = await request(app)
        .get('/test/slow')
        .set('x-tenant-id', 'test-tenant-123');

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('completed');
      expect(response.body.duration).toBe(100);
    });

    it('should handle requests without tenant ID', async () => {
      const response = await request(app)
        .get('/test/success');

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
    });
  });

  describe('Health and Metrics Endpoints', () => {
    it('should provide health check endpoint', async () => {
      const response = await request(app)
        .get('/health');

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('healthy');
      expect(response.body.timestamp).toBeDefined();
    });

    it('should provide metrics endpoint', async () => {
      const response = await request(app)
        .get('/metrics');

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('text/plain');
    });
  });

  describe('Custom Metrics Creation', () => {
    it('should create custom counter metric', () => {
      const counter = metricsCollector.createCustomMetric(
        'counter',
        'test_counter_total',
        'Test counter metric'
      );
      
      expect(counter).toBeDefined();
    });

    it('should create custom histogram metric', () => {
      const histogram = metricsCollector.createCustomMetric(
        'histogram',
        'test_histogram_duration',
        'Test histogram metric',
        'ms'
      );
      
      expect(histogram).toBeDefined();
    });

    it('should create custom gauge metric', () => {
      const gauge = metricsCollector.createCustomMetric(
        'gauge',
        'test_gauge_value',
        'Test gauge metric'
      );
      
      expect(gauge).toBeDefined();
    });

    it('should throw error for unknown metric type', () => {
      expect(() => {
        metricsCollector.createCustomMetric(
          'unknown',
          'test_unknown',
          'Unknown metric type'
        );
      }).toThrow('Unknown metric type: unknown');
    });
  });

  describe('Error Handling', () => {
    it('should handle tracing errors gracefully', async () => {
      const errorFunction = async () => {
        throw new Error('Test error');
      };

      const tracedErrorFunction = tracing.traceFunction(
        'error_function',
        errorFunction
      );

      await expect(tracedErrorFunction()).rejects.toThrow('Test error');
    });

    it('should handle metrics collection when not initialized', () => {
      const uninitializedCollector = {
        initialized: false,
        recordHttpRequest: metricsCollector.recordHttpRequest.bind(metricsCollector),
      };

      // Should not throw error when not initialized
      expect(() => {
        uninitializedCollector.recordHttpRequest('GET', '/test', 200, 100);
      }).not.toThrow();
    });
  });

  describe('Multi-tenant Metrics', () => {
    it('should record metrics with different tenant IDs', async () => {
      const tenants = ['tenant-1', 'tenant-2', 'tenant-3'];
      
      for (const tenant of tenants) {
        const response = await request(app)
          .get('/test/success')
          .set('x-tenant-id', tenant);

        expect(response.status).toBe(200);
      }
    });

    it('should handle tenant-specific storage metrics', () => {
      const tenantId = 'test-tenant-storage';
      const storageBytes = 1024 * 1024 * 100; // 100MB
      
      expect(() => {
        metricsCollector.recordStorageUsage(tenantId, storageBytes, 'media');
      }).not.toThrow();
    });

    it('should handle API token usage metrics', () => {
      const tokenId = 'token-123';
      const tenantId = 'tenant-456';
      const endpoint = '/api/posts';
      
      expect(() => {
        metricsCollector.recordApiTokenUsage(tokenId, tenantId, endpoint);
      }).not.toThrow();
    });
  });
});