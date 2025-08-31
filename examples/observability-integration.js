// Example: Comprehensive observability integration in Express application
import express from 'express';
import { createObservabilityMiddleware } from '../middleware/observability.js';
import metricsCollector from '../lib/observability/metrics.js';
import sloMonitor from '../lib/observability/slo.js';

// Initialize Express app
const app = express();
const port = process.env.PORT || 3000;

// Initialize observability middleware
const observability = createObservabilityMiddleware('api-service', '1.0.0');

// Middleware setup
app.use(express.json());

// Add observability middleware (should be early in the middleware chain)
app.use(observability.createMiddleware());

// Health check endpoint
app.get('/health', observability.createHealthCheckEndpoint());

// Metrics endpoint (for Prometheus scraping)
app.get('/metrics', observability.createMetricsEndpoint());

// Example API endpoints with observability

// Simple GET endpoint
app.get('/api/posts', async (req, res) => {
  try {
    // Simulate database operation with observability
    const posts = await observability.createDatabaseMiddleware()('find', 'posts')(async () => {
      // Simulate database query
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
      return [
        { id: 1, title: 'Post 1', content: 'Content 1' },
        { id: 2, title: 'Post 2', content: 'Content 2' },
      ];
    });

    // Simulate cache operation
    const cachedPosts = await observability.createCacheMiddleware()('get', 'posts:all')(async () => {
      // Simulate cache miss
      return null;
    });

    if (!cachedPosts) {
      // Cache the result
      await observability.createCacheMiddleware()('set', 'posts:all')(async () => {
        // Simulate cache set
        await new Promise(resolve => setTimeout(resolve, 10));
        return posts;
      });
    }

    res.json({ posts, cached: !!cachedPosts });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST endpoint with validation
app.post('/api/posts', async (req, res) => {
  try {
    const { title, content } = req.body;

    // Validate input
    if (!title || !content) {
      return res.status(400).json({ error: 'Title and content are required' });
    }

    // Simulate database operation
    const newPost = await observability.createDatabaseMiddleware()('insert', 'posts')(async () => {
      // Simulate database insert
      await new Promise(resolve => setTimeout(resolve, Math.random() * 200));
      return { id: Date.now(), title, content, createdAt: new Date() };
    });

    // Invalidate cache
    await observability.createCacheMiddleware()('del', 'posts:all')(async () => {
      // Simulate cache invalidation
      await new Promise(resolve => setTimeout(resolve, 5));
      return true;
    });

    res.status(201).json({ post: newPost });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Media upload endpoint
app.post('/api/media/upload', async (req, res) => {
  try {
    const { filename, contentType, size } = req.body;

    if (!filename || !contentType) {
      return res.status(400).json({ error: 'Filename and content type are required' });
    }

    // Record media upload metrics
    metricsCollector.recordMediaUpload(
      contentType.split('/')[0], // image, video, etc.
      size || 0,
      true,
      req.tenantId
    );

    // Simulate media processing
    const result = await observability.createMediaProcessingMiddleware()(contentType.split('/')[0])(async () => {
      // Simulate processing time based on media type
      const processingTime = contentType.startsWith('video/') ? 
        Math.random() * 5000 : // Video: 0-5 seconds
        Math.random() * 1000;  // Image: 0-1 second

      await new Promise(resolve => setTimeout(resolve, processingTime));

      return {
        id: Date.now(),
        filename,
        contentType,
        size,
        processedAt: new Date(),
        thumbnails: contentType.startsWith('image/') ? ['thumb_small.jpg', 'thumb_large.jpg'] : [],
      };
    });

    res.json({ media: result });
  } catch (error) {
    // Record failed upload
    metricsCollector.recordMediaUpload(
      req.body.contentType?.split('/')[0] || 'unknown',
      req.body.size || 0,
      false,
      req.tenantId
    );
    
    res.status(500).json({ error: error.message });
  }
});

// Endpoint that simulates different response times and error rates
app.get('/api/test/load', async (req, res) => {
  const { delay, errorRate } = req.query;
  
  // Simulate variable delay
  if (delay) {
    await new Promise(resolve => setTimeout(resolve, parseInt(delay)));
  }
  
  // Simulate random errors
  const shouldError = Math.random() < (parseFloat(errorRate) || 0);
  if (shouldError) {
    return res.status(500).json({ error: 'Simulated error' });
  }
  
  res.json({ 
    message: 'Success',
    timestamp: new Date().toISOString(),
    delay: delay || 0,
  });
});

// SLO monitoring endpoint
app.get('/api/slo/status', async (req, res) => {
  try {
    const summary = sloMonitor.getSLOSummary();
    const slos = sloMonitor.getAllSLOs();
    
    res.json({
      summary,
      slos: slos.map(slo => ({
        id: slo.id,
        name: slo.name,
        target: slo.target,
        compliance: slo.currentCompliance,
        errorBudgetRemaining: slo.errorBudgetRemaining,
        burnRate: slo.burnRate,
        timeToExhaustion: slo.timeToExhaustion,
        lastEvaluated: slo.lastEvaluated,
      })),
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Custom metrics endpoint
app.post('/api/metrics/custom', (req, res) => {
  const { metricName, value, labels } = req.body;
  
  try {
    // Create and record custom metric
    const customMetric = metricsCollector.createCustomMetric('counter', metricName, 'Custom metric');
    if (customMetric) {
      customMetric.add(value || 1, labels || {});
      res.json({ message: 'Custom metric recorded' });
    } else {
      res.status(400).json({ error: 'Failed to create custom metric' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    requestId: req.requestId,
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    error: 'Not found',
    path: req.path,
    requestId: req.requestId,
  });
});

// Start server
const server = app.listen(port, async () => {
  console.log(`Server running on port ${port}`);
  
  // Initialize observability
  await observability.initialize();
  
  console.log('Observability initialized');
  console.log(`Health check: http://localhost:${port}/health`);
  console.log(`Metrics: http://localhost:${port}/metrics`);
  console.log(`SLO Status: http://localhost:${port}/api/slo/status`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  
  server.close(async () => {
    await observability.shutdown();
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  console.log('SIGINT received, shutting down gracefully');
  
  server.close(async () => {
    await observability.shutdown();
    process.exit(0);
  });
});

// Example usage and testing functions
export function simulateLoad() {
  console.log('Starting load simulation...');
  
  const endpoints = [
    '/api/posts',
    '/api/test/load?delay=100',
    '/api/test/load?delay=200&errorRate=0.05',
  ];
  
  // Simulate requests every second
  setInterval(() => {
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    
    fetch(`http://localhost:${port}${endpoint}`)
      .then(res => res.json())
      .catch(err => console.log('Request failed:', err.message));
  }, 1000);
}

// Example SLO evaluation
export async function evaluateSLOs() {
  console.log('Evaluating SLOs...');
  
  // In a real application, these metrics would come from Prometheus
  // Here we simulate the metrics for demonstration
  const mockMetrics = {
    successCount: 950,
    totalCount: 1000,
    fastCount: 990,
  };
  
  try {
    const apiAvailability = await sloMonitor.evaluateSLO('api_availability', mockMetrics);
    const apiLatency = await sloMonitor.evaluateSLO('api_latency', mockMetrics);
    
    console.log('API Availability SLO:', apiAvailability);
    console.log('API Latency SLO:', apiLatency);
  } catch (error) {
    console.error('SLO evaluation failed:', error);
  }
}

// Export for testing
export { app, observability };

// If running directly, start load simulation
if (import.meta.url === `file://${process.argv[1]}`) {
  // Wait a bit for server to start, then begin simulation
  setTimeout(() => {
    simulateLoad();
    setInterval(evaluateSLOs, 60000); // Evaluate SLOs every minute
  }, 2000);
}