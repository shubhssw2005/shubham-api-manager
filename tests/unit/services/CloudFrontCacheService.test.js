/**
 * CloudFront Cache Service Unit Tests
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import CloudFrontCacheService from '../../../services/CloudFrontCacheService.js';

// Mock AWS SDK
vi.mock('@aws-sdk/client-cloudfront', () => ({
  CloudFrontClient: vi.fn(() => ({
    send: vi.fn()
  })),
  CreateInvalidationCommand: vi.fn(),
  GetInvalidationCommand: vi.fn(),
  ListInvalidationsCommand: vi.fn()
}));

// Mock Redis
vi.mock('ioredis', () => {
  return {
    default: vi.fn(() => ({
      get: vi.fn(),
      set: vi.fn(),
      setex: vi.fn(),
      del: vi.fn(),
      keys: vi.fn(),
      lpush: vi.fn(),
      ltrim: vi.fn(),
      disconnect: vi.fn()
    }))
  };
});

describe('CloudFrontCacheService', () => {
  let cacheService;
  let mockCloudFrontClient;
  let mockRedis;

  beforeEach(() => {
    // Reset environment variables
    process.env.CLOUDFRONT_DISTRIBUTION_ID = 'test-distribution-id';
    process.env.AWS_REGION = 'us-east-1';
    process.env.REDIS_URL = 'redis://localhost:6379';

    cacheService = new CloudFrontCacheService();
    mockCloudFrontClient = cacheService.cloudFrontClient;
    mockRedis = cacheService.redis;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    test('should initialize with default configuration', () => {
      expect(cacheService.distributionId).toBe('test-distribution-id');
      expect(cacheService.region).toBe('us-east-1');
      expect(cacheService.maxInvalidationsPerBatch).toBe(3000);
      expect(cacheService.invalidationCooldown).toBe(60000);
    });

    test('should throw error without distribution ID', () => {
      delete process.env.CLOUDFRONT_DISTRIBUTION_ID;
      
      expect(() => {
        new CloudFrontCacheService();
      }).toThrow('CloudFront distribution ID is required');
    });
  });

  describe('invalidatePaths', () => {
    test('should queue paths for batch invalidation', async () => {
      const paths = ['/test-path-1', '/test-path-2'];
      
      const result = await cacheService.invalidatePaths(paths);
      
      expect(result.queued).toBe(true);
      expect(result.paths).toEqual(paths);
      expect(cacheService.invalidationQueue.size).toBe(2);
    });

    test('should create immediate invalidation', async () => {
      const paths = ['/immediate-test-path'];
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.invalidatePaths(paths, { immediate: true });
      
      expect(result.invalidationId).toBe('test-invalidation-id');
      expect(result.status).toBe('InProgress');
      expect(result.paths).toEqual(paths);
    });

    test('should handle single path as string', async () => {
      const path = '/single-test-path';
      
      const result = await cacheService.invalidatePaths(path);
      
      expect(result.queued).toBe(true);
      expect(result.paths).toEqual([path]);
    });
  });

  describe('invalidateTenantMedia', () => {
    test('should invalidate specific tenant media files', async () => {
      const tenantId = 'tenant-123';
      const mediaKeys = ['image1.jpg', 'image2.jpg'];
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.invalidateTenantMedia(tenantId, mediaKeys);
      
      expect(result.invalidationId).toBe('test-invalidation-id');
      expect(mockCloudFrontClient.send).toHaveBeenCalled();
    });

    test('should invalidate all tenant media when no keys provided', async () => {
      const tenantId = 'tenant-456';
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.invalidateTenantMedia(tenantId);
      
      expect(result.invalidationId).toBe('test-invalidation-id');
    });
  });

  describe('invalidateTenantAPI', () => {
    test('should invalidate specific API endpoints', async () => {
      const tenantId = 'tenant-789';
      const endpoints = ['users', 'products'];
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.invalidateTenantAPI(tenantId, endpoints);
      
      expect(result.invalidationId).toBe('test-invalidation-id');
    });

    test('should invalidate all API endpoints when none specified', async () => {
      const tenantId = 'tenant-101112';
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.invalidateTenantAPI(tenantId);
      
      expect(result.invalidationId).toBe('test-invalidation-id');
    });
  });

  describe('smartInvalidation', () => {
    test('should handle media content invalidation', async () => {
      const metadata = {
        tenantId: 'tenant-123',
        resourceId: 'media-456',
        tags: ['product', 'featured']
      };
      
      const result = await cacheService.smartInvalidation('media', metadata);
      
      expect(result.queued).toBe(true);
    });

    test('should handle API content invalidation', async () => {
      const metadata = {
        tenantId: 'tenant-123',
        resourceId: 'api-endpoint',
        dependencies: ['related-endpoint']
      };
      
      const result = await cacheService.smartInvalidation('api', metadata);
      
      expect(result.queued).toBe(true);
    });

    test('should handle static content invalidation', async () => {
      const metadata = {
        resourceId: '/static/app.js',
        relatedPaths: ['/static/app.css']
      };
      const mockResponse = {
        Invalidation: {
          Id: 'test-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.smartInvalidation('static', metadata);
      
      expect(result.invalidationId).toBe('test-invalidation-id');
    });

    test('should throw error for unknown content type', async () => {
      await expect(
        cacheService.smartInvalidation('unknown', {})
      ).rejects.toThrow('Unknown content type: unknown');
    });
  });

  describe('processBatchInvalidation', () => {
    test('should process queued invalidations', async () => {
      // Add items to queue
      cacheService.invalidationQueue.add('/path1');
      cacheService.invalidationQueue.add('/path2');
      
      const mockResponse = {
        Invalidation: {
          Id: 'batch-invalidation-id',
          Status: 'InProgress',
          CreateTime: new Date()
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      mockRedis.get.mockResolvedValue(null); // No previous invalidation
      mockRedis.set.mockResolvedValue('OK');
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      const result = await cacheService.processBatchInvalidation();
      
      expect(result.invalidationId).toBe('batch-invalidation-id');
      expect(cacheService.invalidationQueue.size).toBe(0);
    });

    test('should respect cooldown period', async () => {
      cacheService.invalidationQueue.add('/path1');
      
      const recentTimestamp = Date.now() - 30000; // 30 seconds ago
      mockRedis.get.mockResolvedValue(recentTimestamp.toString());
      
      await cacheService.processBatchInvalidation();
      
      // Should re-queue items due to cooldown
      expect(cacheService.invalidationQueue.size).toBe(1);
    });

    test('should handle empty queue', async () => {
      const result = await cacheService.processBatchInvalidation();
      
      expect(result).toBeUndefined();
      expect(cacheService.batchTimer).toBeNull();
    });
  });

  describe('getInvalidationStatus', () => {
    test('should return invalidation status', async () => {
      const invalidationId = 'test-invalidation-id';
      const mockResponse = {
        Invalidation: {
          Id: invalidationId,
          Status: 'Completed',
          CreateTime: new Date(),
          InvalidationBatch: {
            Paths: {
              Items: ['/test-path']
            }
          }
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      
      const result = await cacheService.getInvalidationStatus(invalidationId);
      
      expect(result.id).toBe(invalidationId);
      expect(result.status).toBe('Completed');
      expect(result.paths).toEqual(['/test-path']);
    });
  });

  describe('listInvalidations', () => {
    test('should return list of invalidations', async () => {
      const mockResponse = {
        InvalidationList: {
          Items: [
            {
              Id: 'invalidation-1',
              Status: 'Completed',
              CreateTime: new Date()
            },
            {
              Id: 'invalidation-2',
              Status: 'InProgress',
              CreateTime: new Date()
            }
          ]
        }
      };
      
      mockCloudFrontClient.send.mockResolvedValue(mockResponse);
      
      const result = await cacheService.listInvalidations();
      
      expect(result).toHaveLength(2);
      expect(result[0].id).toBe('invalidation-1');
      expect(result[1].id).toBe('invalidation-2');
    });
  });

  describe('getCacheStatistics', () => {
    test('should return cache statistics', async () => {
      const stats = await cacheService.getCacheStatistics(24);
      
      expect(stats.hitRate).toBeDefined();
      expect(stats.missRate).toBeDefined();
      expect(stats.totalRequests).toBeDefined();
      expect(stats.period).toBe('24 hours');
      expect(stats.timestamp).toBeDefined();
    });
  });

  describe('optimizeCacheSettings', () => {
    test('should provide optimization recommendations', async () => {
      const recommendations = await cacheService.optimizeCacheSettings();
      
      expect(recommendations.currentStats).toBeDefined();
      expect(recommendations.recommendations).toBeInstanceOf(Array);
      expect(recommendations.optimizationScore).toBeTypeOf('number');
    });

    test('should recommend TTL increase for low hit rate', async () => {
      // Mock low hit rate
      vi.spyOn(cacheService, 'getCacheStatistics').mockResolvedValue({
        hitRate: 0.75,
        missRate: 0.25,
        totalRequests: 1000,
        cacheHits: 750,
        cacheMisses: 250
      });
      
      const recommendations = await cacheService.optimizeCacheSettings();
      
      expect(recommendations.recommendations).toContainEqual(
        expect.objectContaining({
          type: 'TTL_INCREASE',
          impact: 'HIGH'
        })
      );
    });
  });

  describe('cleanup', () => {
    test('should remove old invalidation tracking data', async () => {
      const oldKey = 'cloudfront:invalidation:old-id';
      const recentKey = 'cloudfront:invalidation:recent-id';
      
      const oldData = JSON.stringify({
        createdAt: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString() // 8 days ago
      });
      
      const recentData = JSON.stringify({
        createdAt: new Date().toISOString()
      });
      
      mockRedis.keys.mockResolvedValue([oldKey, recentKey]);
      mockRedis.get.mockImplementation((key) => {
        if (key === oldKey) return Promise.resolve(oldData);
        if (key === recentKey) return Promise.resolve(recentData);
        return Promise.resolve(null);
      });
      mockRedis.del.mockResolvedValue(1);
      
      await cacheService.cleanup();
      
      expect(mockRedis.del).toHaveBeenCalledWith(oldKey);
      expect(mockRedis.del).not.toHaveBeenCalledWith(recentKey);
    });
  });

  describe('Error handling', () => {
    test('should handle CloudFront API errors', async () => {
      const paths = ['/test-path'];
      const error = new Error('CloudFront API Error');
      
      mockCloudFrontClient.send.mockRejectedValue(error);
      
      await expect(
        cacheService.invalidatePaths(paths, { immediate: true })
      ).rejects.toThrow('CloudFront API Error');
    });

    test('should handle batch invalidation errors gracefully', async () => {
      cacheService.invalidationQueue.add('/path1');
      
      const error = new Error('Batch processing error');
      mockCloudFrontClient.send.mockRejectedValue(error);
      mockRedis.get.mockResolvedValue(null);
      
      // Should not throw, but re-queue items
      await cacheService.processBatchInvalidation();
      
      expect(cacheService.invalidationQueue.size).toBe(1);
    });
  });

  describe('trackInvalidation', () => {
    test('should track invalidation in Redis', async () => {
      const invalidationId = 'test-invalidation-id';
      const paths = ['/test-path'];
      
      mockRedis.setex.mockResolvedValue('OK');
      mockRedis.lpush.mockResolvedValue(1);
      mockRedis.ltrim.mockResolvedValue('OK');
      
      await cacheService.trackInvalidation(invalidationId, paths);
      
      expect(mockRedis.setex).toHaveBeenCalledWith(
        `cloudfront:invalidation:${invalidationId}`,
        86400,
        expect.stringContaining(invalidationId)
      );
      expect(mockRedis.lpush).toHaveBeenCalledWith('cloudfront:recent_invalidations', invalidationId);
      expect(mockRedis.ltrim).toHaveBeenCalledWith('cloudfront:recent_invalidations', 0, 99);
    });
  });
});