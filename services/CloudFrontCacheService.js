/**
 * CloudFront Cache Management Service
 * Handles cache invalidation strategies and cache optimization
 */

import { 
  CloudFrontClient, 
  CreateInvalidationCommand,
  GetInvalidationCommand,
  ListInvalidationsCommand
} from '@aws-sdk/client-cloudfront';
import Redis from 'ioredis';

class CloudFrontCacheService {
  constructor(options = {}) {
    this.distributionId = options.distributionId || process.env.CLOUDFRONT_DISTRIBUTION_ID;
    this.region = options.region || process.env.AWS_REGION || 'us-east-1';
    
    // Initialize CloudFront client
    this.cloudFrontClient = new CloudFrontClient({
      region: this.region,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    });

    // Initialize Redis for invalidation tracking
    this.redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
    
    // Configuration
    this.maxInvalidationsPerBatch = 3000; // CloudFront limit
    this.invalidationCooldown = 60000; // 1 minute between invalidations
    this.batchTimeout = 5000; // 5 seconds to collect batch invalidations
    
    // Batch invalidation queue
    this.invalidationQueue = new Set();
    this.batchTimer = null;
    
    if (!this.distributionId) {
      throw new Error('CloudFront distribution ID is required');
    }
  }

  /**
   * Invalidate specific paths immediately
   * @param {string|Array} paths - Path(s) to invalidate
   * @param {Object} options - Invalidation options
   * @returns {Promise<Object>} Invalidation result
   */
  async invalidatePaths(paths, options = {}) {
    const pathArray = Array.isArray(paths) ? paths : [paths];
    const { immediate = false, reference = null } = options;

    if (immediate) {
      return this.createInvalidation(pathArray, reference);
    }

    // Add to batch queue
    pathArray.forEach(path => this.invalidationQueue.add(path));
    
    // Start batch timer if not already running
    if (!this.batchTimer) {
      this.batchTimer = setTimeout(() => {
        this.processBatchInvalidation();
      }, this.batchTimeout);
    }

    return { queued: true, paths: pathArray };
  }

  /**
   * Invalidate media files for a specific tenant
   * @param {string} tenantId - Tenant ID
   * @param {Array} mediaKeys - Optional specific media keys
   * @returns {Promise<Object>} Invalidation result
   */
  async invalidateTenantMedia(tenantId, mediaKeys = null) {
    let paths;
    
    if (mediaKeys && mediaKeys.length > 0) {
      // Invalidate specific media files
      paths = mediaKeys.map(key => `/tenants/${tenantId}/media/${key}`);
    } else {
      // Invalidate all tenant media
      paths = [`/tenants/${tenantId}/media/*`];
    }

    return this.invalidatePaths(paths, { immediate: true });
  }

  /**
   * Invalidate API responses for a tenant
   * @param {string} tenantId - Tenant ID
   * @param {Array} endpoints - Optional specific endpoints
   * @returns {Promise<Object>} Invalidation result
   */
  async invalidateTenantAPI(tenantId, endpoints = null) {
    let paths;
    
    if (endpoints && endpoints.length > 0) {
      paths = endpoints.map(endpoint => `/api/${endpoint}*`);
    } else {
      paths = [`/api/*`];
    }

    // Add tenant-specific query parameters
    const tenantPaths = paths.map(path => `${path}?tenant=${tenantId}`);
    
    return this.invalidatePaths([...paths, ...tenantPaths], { immediate: true });
  }

  /**
   * Smart invalidation based on content type
   * @param {string} contentType - Type of content (media, api, static)
   * @param {Object} metadata - Content metadata
   * @returns {Promise<Object>} Invalidation result
   */
  async smartInvalidation(contentType, metadata = {}) {
    const { tenantId, resourceId, tags = [] } = metadata;
    
    switch (contentType) {
      case 'media':
        return this.invalidateMediaContent(tenantId, resourceId, tags);
      
      case 'api':
        return this.invalidateAPIContent(tenantId, resourceId, tags);
      
      case 'static':
        return this.invalidateStaticContent(resourceId, tags);
      
      default:
        throw new Error(`Unknown content type: ${contentType}`);
    }
  }

  /**
   * Invalidate media content with related resources
   */
  async invalidateMediaContent(tenantId, mediaId, tags = []) {
    const paths = [
      `/tenants/${tenantId}/media/${mediaId}`,
      `/tenants/${tenantId}/media/${mediaId}/thumbnails/*`,
      `/tenants/${tenantId}/media/${mediaId}/variants/*`
    ];

    // Add tag-based invalidations
    if (tags.length > 0) {
      tags.forEach(tag => {
        paths.push(`/api/media?tag=${tag}`);
        paths.push(`/api/tenants/${tenantId}/media?tag=${tag}`);
      });
    }

    return this.invalidatePaths(paths);
  }

  /**
   * Invalidate API content with cache dependencies
   */
  async invalidateAPIContent(tenantId, resourceId, dependencies = []) {
    const paths = [
      `/api/tenants/${tenantId}/${resourceId}`,
      `/api/tenants/${tenantId}/${resourceId}/*`
    ];

    // Add dependency invalidations
    dependencies.forEach(dep => {
      paths.push(`/api/tenants/${tenantId}/${dep}*`);
    });

    return this.invalidatePaths(paths);
  }

  /**
   * Invalidate static content
   */
  async invalidateStaticContent(resourcePath, relatedPaths = []) {
    const paths = [resourcePath, ...relatedPaths];
    return this.invalidatePaths(paths, { immediate: true });
  }

  /**
   * Process batch invalidation queue
   */
  async processBatchInvalidation() {
    if (this.invalidationQueue.size === 0) {
      this.batchTimer = null;
      return;
    }

    const paths = Array.from(this.invalidationQueue);
    this.invalidationQueue.clear();
    this.batchTimer = null;

    // Check cooldown
    const lastInvalidation = await this.redis.get('cloudfront:last_invalidation');
    const now = Date.now();
    
    if (lastInvalidation && (now - parseInt(lastInvalidation)) < this.invalidationCooldown) {
      // Re-queue for later
      paths.forEach(path => this.invalidationQueue.add(path));
      this.batchTimer = setTimeout(() => {
        this.processBatchInvalidation();
      }, this.invalidationCooldown);
      return;
    }

    try {
      const result = await this.createInvalidation(paths);
      await this.redis.set('cloudfront:last_invalidation', now.toString());
      return result;
    } catch (error) {
      console.error('Batch invalidation failed:', error);
      // Re-queue failed paths
      paths.forEach(path => this.invalidationQueue.add(path));
    }
  }

  /**
   * Create CloudFront invalidation
   */
  async createInvalidation(paths, reference = null) {
    // Limit paths to CloudFront maximum
    const limitedPaths = paths.slice(0, this.maxInvalidationsPerBatch);
    
    const callerReference = reference || `invalidation-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const command = new CreateInvalidationCommand({
      DistributionId: this.distributionId,
      InvalidationBatch: {
        Paths: {
          Quantity: limitedPaths.length,
          Items: limitedPaths
        },
        CallerReference: callerReference
      }
    });

    try {
      const response = await this.cloudFrontClient.send(command);
      
      // Track invalidation
      await this.trackInvalidation(response.Invalidation.Id, limitedPaths);
      
      return {
        invalidationId: response.Invalidation.Id,
        status: response.Invalidation.Status,
        paths: limitedPaths,
        createTime: response.Invalidation.CreateTime
      };
    } catch (error) {
      console.error('CloudFront invalidation failed:', error);
      throw error;
    }
  }

  /**
   * Get invalidation status
   * @param {string} invalidationId - Invalidation ID
   * @returns {Promise<Object>} Invalidation status
   */
  async getInvalidationStatus(invalidationId) {
    const command = new GetInvalidationCommand({
      DistributionId: this.distributionId,
      Id: invalidationId
    });

    try {
      const response = await this.cloudFrontClient.send(command);
      return {
        id: response.Invalidation.Id,
        status: response.Invalidation.Status,
        createTime: response.Invalidation.CreateTime,
        paths: response.Invalidation.InvalidationBatch.Paths.Items
      };
    } catch (error) {
      console.error('Failed to get invalidation status:', error);
      throw error;
    }
  }

  /**
   * List recent invalidations
   * @param {number} maxItems - Maximum number of items to return
   * @returns {Promise<Array>} List of invalidations
   */
  async listInvalidations(maxItems = 100) {
    const command = new ListInvalidationsCommand({
      DistributionId: this.distributionId,
      MaxItems: maxItems.toString()
    });

    try {
      const response = await this.cloudFrontClient.send(command);
      return response.InvalidationList.Items.map(item => ({
        id: item.Id,
        status: item.Status,
        createTime: item.CreateTime
      }));
    } catch (error) {
      console.error('Failed to list invalidations:', error);
      throw error;
    }
  }

  /**
   * Track invalidation in Redis for monitoring
   */
  async trackInvalidation(invalidationId, paths) {
    const invalidationData = {
      id: invalidationId,
      paths,
      createdAt: new Date().toISOString(),
      status: 'InProgress'
    };

    await this.redis.setex(
      `cloudfront:invalidation:${invalidationId}`,
      86400, // 24 hours
      JSON.stringify(invalidationData)
    );

    // Add to recent invalidations list
    await this.redis.lpush('cloudfront:recent_invalidations', invalidationId);
    await this.redis.ltrim('cloudfront:recent_invalidations', 0, 99); // Keep last 100
  }

  /**
   * Get cache hit rate statistics
   * @param {number} hours - Number of hours to look back
   * @returns {Promise<Object>} Cache statistics
   */
  async getCacheStatistics(hours = 24) {
    // This would integrate with CloudWatch metrics
    // For now, return mock data structure
    return {
      hitRate: 0.85,
      missRate: 0.15,
      totalRequests: 1000000,
      cacheHits: 850000,
      cacheMisses: 150000,
      period: `${hours} hours`,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Optimize cache settings based on usage patterns
   */
  async optimizeCacheSettings() {
    // Analyze cache patterns and suggest optimizations
    const stats = await this.getCacheStatistics(168); // 7 days
    
    const recommendations = [];
    
    if (stats.hitRate < 0.8) {
      recommendations.push({
        type: 'TTL_INCREASE',
        message: 'Consider increasing TTL for static assets',
        impact: 'HIGH'
      });
    }
    
    if (stats.missRate > 0.3) {
      recommendations.push({
        type: 'CACHE_BEHAVIOR',
        message: 'Review cache behaviors for dynamic content',
        impact: 'MEDIUM'
      });
    }
    
    return {
      currentStats: stats,
      recommendations,
      optimizationScore: Math.round(stats.hitRate * 100)
    };
  }

  /**
   * Cleanup old invalidation tracking data
   */
  async cleanup() {
    const keys = await this.redis.keys('cloudfront:invalidation:*');
    const now = Date.now();
    
    for (const key of keys) {
      const data = await this.redis.get(key);
      if (data) {
        const invalidation = JSON.parse(data);
        const createdAt = new Date(invalidation.createdAt).getTime();
        
        // Remove invalidations older than 7 days
        if (now - createdAt > 7 * 24 * 60 * 60 * 1000) {
          await this.redis.del(key);
        }
      }
    }
  }
}

export default CloudFrontCacheService;