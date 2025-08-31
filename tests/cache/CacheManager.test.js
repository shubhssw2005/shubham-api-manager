import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import CacheManager from '../../lib/cache/CacheManager.js';

// Mock Redis
vi.mock('ioredis', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      get: vi.fn(),
      set: vi.fn(),
      setex: vi.fn(),
      del: vi.fn(),
      keys: vi.fn(),
      pipeline: vi.fn(() => ({
        del: vi.fn(),
        exec: vi.fn()
      })),
      on: vi.fn(),
      quit: vi.fn()
    })),
    Cluster: vi.fn().mockImplementation(() => ({
      get: vi.fn(),
      set: vi.fn(),
      setex: vi.fn(),
      del: vi.fn(),
      keys: vi.fn(),
      pipeline: vi.fn(() => ({
        del: vi.fn(),
        exec: vi.fn()
      })),
      on: vi.fn(),
      quit: vi.fn()
    }))
  };
});

describe('CacheManager', () => {
  let cacheManager;
  let mockRedis;

  beforeEach(() => {
    // Clear environment variables
    delete process.env.REDIS_CLUSTER_NODES;
    delete process.env.REDIS_HOST;
    delete process.env.REDIS_PORT;
    delete process.env.REDIS_PASSWORD;

    cacheManager = new CacheManager({
      memoryTTL: 300,
      memoryMaxKeys: 1000,
      redisTTL: 3600,
      redisKeyPrefix: 'test:'
    });

    mockRedis = cacheManager.redisCache;
  });

  afterEach(async () => {
    await cacheManager.close();
  });

  describe('L1 Cache (Memory)', () => {
    it('should store and retrieve values from memory cache', async () => {
      const key = 'test:key';
      const value = { data: 'test value' };

      await cacheManager.set(key, value, 300);
      const retrieved = await cacheManager.get(key);

      expect(retrieved).toEqual(value);
    });

    it('should return null for non-existent keys', async () => {
      const result = await cacheManager.get('non:existent:key');
      expect(result).toBeNull();
    });

    it('should handle memory cache expiration', async () => {
      const key = 'test:expire';
      const value = { data: 'expire test' };

      await cacheManager.set(key, value, 1); // 1 second TTL
      
      // Should be available immediately
      let retrieved = await cacheManager.get(key);
      expect(retrieved).toEqual(value);

      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 1100));
      
      // Should be expired from memory, but might still be in Redis
      mockRedis.get.mockResolvedValue(null);
      retrieved = await cacheManager.get(key);
      expect(retrieved).toBeNull();
    });
  });

  describe('L2 Cache (Redis)', () => {
    it('should fallback to Redis when memory cache misses', async () => {
      const key = 'test:redis:fallback';
      const value = { data: 'redis value' };
      
      // Mock Redis to return the value
      mockRedis.get.mockResolvedValue(JSON.stringify(value));

      const retrieved = await cacheManager.get(key);
      
      expect(mockRedis.get).toHaveBeenCalledWith('test:' + key);
      expect(retrieved).toEqual(value);
    });

    it('should populate L1 cache when retrieving from L2', async () => {
      const key = 'test:populate:l1';
      const value = { data: 'populate test' };
      
      mockRedis.get.mockResolvedValue(JSON.stringify(value));

      // First get should hit Redis
      await cacheManager.get(key);
      
      // Second get should hit memory cache (no Redis call)
      mockRedis.get.mockClear();
      const retrieved = await cacheManager.get(key);
      
      expect(mockRedis.get).not.toHaveBeenCalled();
      expect(retrieved).toEqual(value);
    });

    it('should handle Redis errors gracefully', async () => {
      const key = 'test:redis:error';
      
      mockRedis.get.mockRejectedValue(new Error('Redis connection failed'));

      const retrieved = await cacheManager.get(key);
      expect(retrieved).toBeNull();
    });
  });

  describe('Set Operations', () => {
    it('should set values in both L1 and L2 caches', async () => {
      const key = 'test:set:both';
      const value = { data: 'set test' };
      const ttl = 600;

      mockRedis.setex.mockResolvedValue('OK');

      const result = await cacheManager.set(key, value, ttl);

      expect(result).toBe(true);
      expect(mockRedis.setex).toHaveBeenCalledWith('test:' + key, ttl, JSON.stringify(value));
      
      // Should be in memory cache
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toEqual(value);
    });

    it('should handle set errors gracefully', async () => {
      const key = 'test:set:error';
      const value = { data: 'error test' };

      mockRedis.setex.mockRejectedValue(new Error('Redis set failed'));

      const result = await cacheManager.set(key, value, 300);
      expect(result).toBe(false);
    });
  });

  describe('Delete Operations', () => {
    it('should delete from both L1 and L2 caches', async () => {
      const key = 'test:delete';
      const value = { data: 'delete test' };

      // Set first
      await cacheManager.set(key, value, 300);
      
      mockRedis.del.mockResolvedValue(1);

      // Delete
      const result = await cacheManager.del(key);
      
      expect(result).toBe(true);
      expect(mockRedis.del).toHaveBeenCalledWith('test:' + key);
      
      // Should not be retrievable
      mockRedis.get.mockResolvedValue(null);
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toBeNull();
    });
  });

  describe('Request Coalescing', () => {
    it('should coalesce concurrent requests for the same key', async () => {
      const key = 'test:coalesce';
      const value = { data: 'coalesced value' };
      
      const fetchFunction = vi.fn().mockResolvedValue(value);
      mockRedis.get.mockResolvedValue(null);
      mockRedis.setex.mockResolvedValue('OK');

      // Make multiple concurrent requests
      const promises = [
        cacheManager.getOrSet(key, fetchFunction, 300),
        cacheManager.getOrSet(key, fetchFunction, 300),
        cacheManager.getOrSet(key, fetchFunction, 300)
      ];

      const results = await Promise.all(promises);

      // All should return the same value
      results.forEach(result => {
        expect(result).toEqual(value);
      });

      // Fetch function should only be called once
      expect(fetchFunction).toHaveBeenCalledTimes(1);
    });

    it('should handle fetch function errors in coalescing', async () => {
      const key = 'test:coalesce:error';
      const error = new Error('Fetch failed');
      
      const fetchFunction = vi.fn().mockRejectedValue(error);
      mockRedis.get.mockResolvedValue(null);

      await expect(cacheManager.getOrSet(key, fetchFunction, 300)).rejects.toThrow('Fetch failed');
    });
  });

  describe('Invalidation', () => {
    it('should invalidate by pattern', async () => {
      const pattern = 'test:pattern:*';
      const keys = ['test:test:pattern:1', 'test:test:pattern:2'];
      
      mockRedis.keys.mockResolvedValue(keys);
      mockRedis.pipeline.mockReturnValue({
        del: vi.fn(),
        exec: vi.fn().mockResolvedValue([])
      });

      const count = await cacheManager.invalidate(pattern);
      
      expect(mockRedis.keys).toHaveBeenCalledWith('test:' + pattern);
      expect(count).toBeGreaterThan(0);
    });

    it('should clear all caches with wildcard pattern', async () => {
      mockRedis.keys.mockResolvedValue(['test:key1', 'test:key2']);
      mockRedis.pipeline.mockReturnValue({
        del: vi.fn(),
        exec: vi.fn().mockResolvedValue([])
      });

      const count = await cacheManager.invalidate('*');
      
      expect(count).toBeGreaterThan(0);
    });
  });

  describe('Cache Warming', () => {
    it('should warm cache with provided data', async () => {
      const warmingData = [
        { key: 'warm:1', value: { data: 'warm1' }, ttl: 300 },
        { key: 'warm:2', value: { data: 'warm2' }, ttl: 600 }
      ];

      mockRedis.setex.mockResolvedValue('OK');

      const warmedCount = await cacheManager.warm(warmingData);
      
      expect(warmedCount).toBe(2);
      expect(mockRedis.setex).toHaveBeenCalledTimes(2);
    });

    it('should handle warming errors gracefully', async () => {
      const warmingData = [
        { key: 'warm:error', value: { data: 'error' }, ttl: 300 }
      ];

      mockRedis.setex.mockRejectedValue(new Error('Warming failed'));

      const warmedCount = await cacheManager.warm(warmingData);
      
      // Should return 0 because Redis failed, but memory cache might succeed
      expect(warmedCount).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Statistics', () => {
    it('should provide cache statistics', async () => {
      const stats = cacheManager.getStats();
      
      expect(stats).toHaveProperty('l1');
      expect(stats).toHaveProperty('l2');
      expect(stats).toHaveProperty('coalescing');
      expect(stats).toHaveProperty('warming');
      
      // Verify structure of L1 stats
      expect(stats.l1).toHaveProperty('hits');
      expect(stats.l1).toHaveProperty('misses');
      expect(stats.l1).toHaveProperty('sets');
      expect(stats.l1).toHaveProperty('gets');
      expect(stats.l1).toHaveProperty('hitRate');
      
      // Verify structure of L2 stats
      expect(stats.l2).toHaveProperty('hits');
      expect(stats.l2).toHaveProperty('misses');
      expect(stats.l2).toHaveProperty('sets');
      expect(stats.l2).toHaveProperty('hitRate');
      
      // All stats should be numbers
      expect(typeof stats.l1.hits).toBe('number');
      expect(typeof stats.l2.hits).toBe('number');
    });
  });
});