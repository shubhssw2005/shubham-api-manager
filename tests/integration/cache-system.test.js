import { describe, it, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import CacheService from '../../lib/cache/index.js';

// Mock Redis for integration tests
vi.mock('ioredis', () => {
  const mockRedisData = new Map();
  
  const mockRedis = {
    get: vi.fn((key) => Promise.resolve(mockRedisData.get(key) || null)),
    set: vi.fn((key, value) => {
      mockRedisData.set(key, value);
      return Promise.resolve('OK');
    }),
    setex: vi.fn((key, ttl, value) => {
      mockRedisData.set(key, value);
      return Promise.resolve('OK');
    }),
    del: vi.fn((key) => {
      const existed = mockRedisData.has(key);
      mockRedisData.delete(key);
      return Promise.resolve(existed ? 1 : 0);
    }),
    keys: vi.fn((pattern) => {
      const regex = new RegExp(pattern.replace(/\*/g, '.*'));
      const matchingKeys = Array.from(mockRedisData.keys()).filter(key => regex.test(key));
      return Promise.resolve(matchingKeys);
    }),
    pipeline: vi.fn(() => {
      const delKeys = [];
      return {
        del: vi.fn((key) => {
          delKeys.push(key);
        }),
        exec: vi.fn(() => {
          delKeys.forEach(key => mockRedisData.delete(key));
          return Promise.resolve(delKeys.map(() => [null, 1]));
        })
      };
    }),
    on: vi.fn(),
    quit: vi.fn(() => Promise.resolve())
  };

  return {
    default: vi.fn(() => mockRedis),
    Cluster: vi.fn(() => mockRedis)
  };
});

describe('Cache System Integration', () => {
  let cacheService;

  beforeAll(() => {
    cacheService = new CacheService({
      memoryTTL: 300,
      memoryMaxKeys: 1000,
      redisTTL: 3600,
      redisKeyPrefix: 'test:',
      invalidationStrategies: {
        immediate: true,
        delayed: false,
        ttlBased: true,
        tagBased: true
      },
      warmingStrategies: {
        startup: false, // Disable for tests
        scheduled: false,
        predictive: false,
        onDemand: true
      }
    });
  });

  afterAll(async () => {
    await cacheService.close();
  });

  beforeEach(() => {
    // Clear any existing cache data
    vi.clearAllMocks();
  });

  describe('Multi-Layer Caching', () => {
    it('should demonstrate L1 -> L2 cache hierarchy', async () => {
      const key = 'integration:hierarchy';
      const value = { data: 'hierarchy test', timestamp: Date.now() };

      // Set value
      await cacheService.set(key, value, 300);

      // First get should hit L1 (memory)
      const result1 = await cacheService.get(key);
      expect(result1).toEqual(value);

      // Clear L1 cache by creating new instance (simulating memory pressure)
      cacheService.cacheManager.memoryCache.flushAll();

      // Second get should hit L2 (Redis) and populate L1
      const result2 = await cacheService.get(key);
      expect(result2).toEqual(value);

      // Third get should hit L1 again
      const result3 = await cacheService.get(key);
      expect(result3).toEqual(value);
    });

    it('should handle cache misses gracefully', async () => {
      const result = await cacheService.get('non:existent:key');
      expect(result).toBeNull();
    });
  });

  describe('Request Coalescing', () => {
    it('should coalesce concurrent requests to prevent thundering herd', async () => {
      const key = 'integration:coalesce';
      const value = { data: 'coalesced value', id: Math.random() };
      
      let fetchCallCount = 0;
      const fetchFunction = vi.fn(async () => {
        fetchCallCount++;
        // Simulate slow fetch
        await new Promise(resolve => setTimeout(resolve, 50));
        return value;
      });

      // Make 5 concurrent requests
      const promises = Array(5).fill().map(() => 
        cacheService.getOrSet(key, fetchFunction, 300)
      );

      const results = await Promise.all(promises);

      // All results should be identical
      results.forEach(result => {
        expect(result).toEqual(value);
      });

      // Fetch function should only be called once
      expect(fetchCallCount).toBe(1);
      expect(fetchFunction).toHaveBeenCalledTimes(1);
    });
  });

  describe('Event-Driven Invalidation', () => {
    it('should invalidate related cache entries on model events', async () => {
      const tenantId = 'tenant123';
      const postId = 'post456';
      
      // Set up some cached data
      await cacheService.set(`posts:${tenantId}:${postId}`, { id: postId, title: 'Test Post' }, 300);
      await cacheService.set(`posts:${tenantId}:list:recent`, [{ id: postId }], 300);
      await cacheService.set(`posts:${tenantId}:count`, 5, 300);

      // Verify data is cached
      expect(await cacheService.get(`posts:${tenantId}:${postId}`)).toBeTruthy();
      expect(await cacheService.get(`posts:${tenantId}:list:recent`)).toBeTruthy();
      expect(await cacheService.get(`posts:${tenantId}:count`)).toBeTruthy();

      // Emit post update event
      cacheService.emitModelEvent('post:updated', {
        id: postId,
        tenantId: tenantId,
        title: 'Updated Post',
        authorId: 'author789'
      });

      // Wait for invalidation to process
      await new Promise(resolve => setTimeout(resolve, 100));

      // Related cache entries should be invalidated
      expect(await cacheService.get(`posts:${tenantId}:${postId}`)).toBeNull();
      expect(await cacheService.get(`posts:${tenantId}:list:recent`)).toBeNull();
      expect(await cacheService.get(`posts:${tenantId}:count`)).toBeNull();
    });

    it('should handle media events and invalidate media cache', async () => {
      const tenantId = 'tenant123';
      const mediaId = 'media456';
      
      // Set up cached media data
      await cacheService.set(`media:${tenantId}:${mediaId}`, { id: mediaId, filename: 'test.jpg' }, 300);
      await cacheService.set(`media:${tenantId}:list:recent`, [{ id: mediaId }], 300);

      // Emit media processed event
      cacheService.emitModelEvent('media:processed', {
        id: mediaId,
        tenantId: tenantId,
        filename: 'test.jpg',
        mimeType: 'image/jpeg',
        folderId: 'folder123'
      });

      await new Promise(resolve => setTimeout(resolve, 50));

      // Media cache should be invalidated
      expect(await cacheService.get(`media:${tenantId}:${mediaId}`)).toBeNull();
      expect(await cacheService.get(`media:${tenantId}:list:recent`)).toBeNull();
    });
  });

  describe('Tag-Based Invalidation', () => {
    it('should support tag-based cache invalidation', async () => {
      const key1 = 'tagged:item1';
      const key2 = 'tagged:item2';
      const key3 = 'tagged:item3';
      
      // Set items with different tag combinations
      await cacheService.setWithTags(key1, { data: 'item1' }, 300, ['tag:a', 'tag:b']);
      await cacheService.setWithTags(key2, { data: 'item2' }, 300, ['tag:b', 'tag:c']);
      await cacheService.setWithTags(key3, { data: 'item3' }, 300, ['tag:c']);

      // Verify all items are cached
      expect(await cacheService.get(key1)).toBeTruthy();
      expect(await cacheService.get(key2)).toBeTruthy();
      expect(await cacheService.get(key3)).toBeTruthy();

      // Invalidate by tag:b
      await cacheService.invalidateByTags(['tag:b']);

      // Items with tag:b should be invalidated
      expect(await cacheService.get(key1)).toBeNull();
      expect(await cacheService.get(key2)).toBeNull();
      
      // Item with only tag:c should remain
      expect(await cacheService.get(key3)).toBeTruthy();
    });
  });

  describe('Cache Warming', () => {
    it('should warm tenant cache on demand', async () => {
      const tenantId = 'warm:tenant123';
      
      // Mock the warming service's fetch methods
      const originalFetchTenantById = cacheService.warmingService.fetchTenantById;
      const originalFetchTenantSettings = cacheService.warmingService.fetchTenantSettings;
      
      cacheService.warmingService.fetchTenantById = vi.fn().mockResolvedValue({
        id: tenantId,
        name: 'Test Tenant'
      });
      
      cacheService.warmingService.fetchTenantSettings = vi.fn().mockResolvedValue({
        theme: 'dark',
        language: 'en'
      });

      const warmedCount = await cacheService.warmTenant(tenantId, 'high');

      expect(warmedCount).toBeGreaterThan(0);
      
      // Verify cached data
      const cachedTenant = await cacheService.get(`tenant:${tenantId}`);
      expect(cachedTenant).toEqual({ id: tenantId, name: 'Test Tenant' });

      // Restore original methods
      cacheService.warmingService.fetchTenantById = originalFetchTenantById;
      cacheService.warmingService.fetchTenantSettings = originalFetchTenantSettings;
    });
  });

  describe('Pattern-Based Operations', () => {
    it('should invalidate cache entries by pattern', async () => {
      // Set up test data
      await cacheService.set('pattern:test:1', { data: 'test1' }, 300);
      await cacheService.set('pattern:test:2', { data: 'test2' }, 300);
      await cacheService.set('pattern:other:1', { data: 'other1' }, 300);

      // Verify data is cached
      expect(await cacheService.get('pattern:test:1')).toBeTruthy();
      expect(await cacheService.get('pattern:test:2')).toBeTruthy();
      expect(await cacheService.get('pattern:other:1')).toBeTruthy();

      // Invalidate by pattern
      const invalidatedCount = await cacheService.invalidate('pattern:test:*');

      expect(invalidatedCount).toBeGreaterThan(0);

      // Pattern-matched items should be invalidated
      expect(await cacheService.get('pattern:test:1')).toBeNull();
      expect(await cacheService.get('pattern:test:2')).toBeNull();
      
      // Non-matching item should remain
      expect(await cacheService.get('pattern:other:1')).toBeTruthy();
    });
  });

  describe('Health Check', () => {
    it('should perform health check successfully', async () => {
      const health = await cacheService.healthCheck();

      expect(health).toHaveProperty('healthy', true);
      expect(health).toHaveProperty('timestamp');
      expect(health).toHaveProperty('stats');
      expect(health.stats).toHaveProperty('cache');
      expect(health.stats).toHaveProperty('invalidation');
      expect(health.stats).toHaveProperty('warming');
    });
  });

  describe('Statistics', () => {
    it('should provide comprehensive cache statistics', async () => {
      const stats = cacheService.getStats();

      expect(stats).toHaveProperty('cache');
      expect(stats).toHaveProperty('invalidation');
      expect(stats).toHaveProperty('warming');

      // L1 cache stats
      expect(stats.cache.l1).toHaveProperty('hits');
      expect(stats.cache.l1).toHaveProperty('misses');
      expect(stats.cache.l1).toHaveProperty('sets');
      expect(stats.cache.l1).toHaveProperty('hitRate');

      // L2 cache stats
      expect(stats.cache.l2).toHaveProperty('hits');
      expect(stats.cache.l2).toHaveProperty('misses');
      expect(stats.cache.l2).toHaveProperty('sets');
      expect(stats.cache.l2).toHaveProperty('hitRate');

      // Verify structure is correct
      expect(typeof stats.cache.l1.sets).toBe('number');
      expect(typeof stats.cache.l1.hits).toBe('number');
    });
  });

  describe('Error Handling', () => {
    it('should handle Redis connection errors gracefully', async () => {
      // Mock Redis error
      const originalRedis = cacheService.cacheManager.redisCache;
      cacheService.cacheManager.redisCache.get = vi.fn().mockRejectedValue(new Error('Redis connection failed'));

      // Should not throw, should return null
      const result = await cacheService.get('error:test:key');
      expect(result).toBeNull();

      // Restore original Redis
      cacheService.cacheManager.redisCache = originalRedis;
    });

    it('should handle serialization errors gracefully', async () => {
      // Mock Redis to return invalid JSON
      const originalRedis = cacheService.cacheManager.redisCache;
      cacheService.cacheManager.redisCache.get = vi.fn().mockResolvedValue('invalid json {');

      const result = await cacheService.get('serialization:error:key');
      expect(result).toBeNull();

      // Restore original Redis
      cacheService.cacheManager.redisCache = originalRedis;
    });
  });
});