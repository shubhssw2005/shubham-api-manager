import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import CacheInvalidationService from '../../lib/cache/CacheInvalidationService.js';
import CacheManager from '../../lib/cache/CacheManager.js';

// Mock CacheManager
vi.mock('../../lib/cache/CacheManager.js');

describe('CacheInvalidationService', () => {
  let invalidationService;
  let mockCacheManager;

  beforeEach(() => {
    mockCacheManager = {
      invalidate: vi.fn(),
      del: vi.fn(),
      set: vi.fn(),
      on: vi.fn(),
      emit: vi.fn()
    };

    invalidationService = new CacheInvalidationService(mockCacheManager, {
      strategies: {
        immediate: true,
        delayed: false,
        ttlBased: true,
        tagBased: true
      }
    });
  });

  afterEach(() => {
    invalidationService.clear();
  });

  describe('Model Event Handling', () => {
    it('should invalidate post-related cache on post creation', async () => {
      const post = {
        id: 'post123',
        tenantId: 'tenant456',
        authorId: 'author789',
        categories: ['cat1', 'cat2'],
        tags: ['tag1', 'tag2']
      };

      mockCacheManager.invalidate.mockResolvedValue(5);

      invalidationService.emit('post:created', post);

      // Wait for async processing
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(
        expect.stringContaining(`posts:${post.tenantId}:`)
      );
    });

    it('should invalidate media-related cache on media upload', async () => {
      const media = {
        id: 'media123',
        tenantId: 'tenant456',
        folderId: 'folder789',
        mimeType: 'image/jpeg',
        uploadedBy: 'user123'
      };

      mockCacheManager.invalidate.mockResolvedValue(3);

      invalidationService.emit('media:uploaded', media);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(
        expect.stringContaining(`media:${media.tenantId}:`)
      );
    });

    it('should invalidate user-related cache on user update', async () => {
      const user = {
        id: 'user123',
        tenantId: 'tenant456',
        role: 'admin'
      };

      mockCacheManager.invalidate.mockResolvedValue(2);

      invalidationService.emit('user:updated', user);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(
        expect.stringContaining(`users:${user.tenantId}:`)
      );
    });

    it('should invalidate tenant-related cache on tenant update', async () => {
      const tenant = {
        id: 'tenant123'
      };

      mockCacheManager.invalidate.mockResolvedValue(4);

      invalidationService.emit('tenant:updated', tenant);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(
        expect.stringContaining(`tenant:${tenant.id}:`)
      );
    });
  });

  describe('Tag-based Invalidation', () => {
    it('should set cache with tags and maintain tag mappings', async () => {
      const key = 'test:key';
      const value = { data: 'test' };
      const tags = ['tag1', 'tag2'];

      mockCacheManager.set.mockResolvedValue(true);

      const result = await invalidationService.setWithTags(key, value, 300, tags);

      expect(result).toBe(true);
      expect(mockCacheManager.set).toHaveBeenCalledWith(key, value, 300);
      
      // Check tag mappings
      expect(invalidationService.tagMappings.get('tag1')).toContain(key);
      expect(invalidationService.tagMappings.get('tag2')).toContain(key);
      expect(invalidationService.keyTags.get(key)).toEqual(new Set(tags));
    });

    it('should invalidate cache entries by tags', async () => {
      const key1 = 'test:key1';
      const key2 = 'test:key2';

      // Set up tag mappings manually since we're mocking the cache manager
      invalidationService.tagMappings.set('tag1', new Set([key1, key2]));
      invalidationService.keyTags.set(key1, new Set(['tag1']));
      invalidationService.keyTags.set(key2, new Set(['tag1', 'tag2']));

      mockCacheManager.del.mockResolvedValue(true);

      const invalidatedCount = await invalidationService.invalidateByTags(['tag1']);

      expect(mockCacheManager.del).toHaveBeenCalledWith(key1);
      expect(mockCacheManager.del).toHaveBeenCalledWith(key2);
      expect(invalidatedCount).toBe(2);
    });

    it('should clean up tag mappings after invalidation', async () => {
      const key = 'test:cleanup';
      const tag = 'cleanup:tag';

      // Set up tag mappings manually
      invalidationService.tagMappings.set(tag, new Set([key]));
      invalidationService.keyTags.set(key, new Set([tag]));
      
      expect(invalidationService.tagMappings.get(tag).has(key)).toBe(true);

      mockCacheManager.del.mockResolvedValue(true);
      await invalidationService.invalidateByTags([tag]);

      expect(invalidationService.tagMappings.has(tag)).toBe(false);
      expect(invalidationService.keyTags.has(key)).toBe(false);
    });
  });

  describe('Pattern-based Invalidation', () => {
    it('should invalidate by pattern', async () => {
      const pattern = 'test:pattern:*';
      
      mockCacheManager.invalidate.mockResolvedValue(5);

      const count = await invalidationService.invalidateByPattern(pattern);

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(pattern);
      expect(count).toBe(5);
    });
  });

  describe('Key-based Invalidation', () => {
    it('should invalidate specific keys', async () => {
      const keys = ['key1', 'key2', 'key3'];
      
      mockCacheManager.del.mockResolvedValue(true);

      const count = await invalidationService.invalidateByKeys(keys);

      expect(mockCacheManager.del).toHaveBeenCalledTimes(3);
      expect(count).toBe(3);
    });

    it('should handle deletion failures', async () => {
      const keys = ['key1', 'key2'];
      
      mockCacheManager.del
        .mockResolvedValueOnce(true)
        .mockResolvedValueOnce(false);

      const count = await invalidationService.invalidateByKeys(keys);

      expect(count).toBe(1);
    });
  });

  describe('Custom Events', () => {
    it('should handle custom invalidation events', async () => {
      const pattern = 'custom:pattern:*';
      
      mockCacheManager.invalidate.mockResolvedValue(3);

      invalidationService.emit('cache:invalidate:pattern', pattern);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.invalidate).toHaveBeenCalledWith(pattern);
    });

    it('should handle tag invalidation events', async () => {
      const tags = ['custom:tag1', 'custom:tag2'];
      const key = 'tagged:key';
      
      // Set up tag mappings manually
      invalidationService.tagMappings.set('custom:tag1', new Set([key]));
      invalidationService.keyTags.set(key, new Set(['custom:tag1']));
      
      mockCacheManager.del.mockResolvedValue(true);

      invalidationService.emit('cache:invalidate:tags', tags);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.del).toHaveBeenCalled();
    });

    it('should handle key invalidation events', async () => {
      const keys = ['custom:key1', 'custom:key2'];
      
      mockCacheManager.del.mockResolvedValue(true);

      invalidationService.emit('cache:invalidate:keys', keys);

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(mockCacheManager.del).toHaveBeenCalledTimes(2);
    });
  });

  describe('Statistics', () => {
    it('should provide invalidation statistics', () => {
      // Add some data to queues and mappings
      invalidationService.invalidationQueue.push({ type: 'test' });
      invalidationService.tagMappings.set('test:tag', new Set(['key1', 'key2']));
      invalidationService.keyTags.set('key1', new Set(['test:tag']));

      const stats = invalidationService.getStats();

      expect(stats).toHaveProperty('queueSize', 1);
      expect(stats).toHaveProperty('tagMappings', 1);
      expect(stats).toHaveProperty('keyTags', 1);
      expect(stats).toHaveProperty('strategies');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalidation errors gracefully', async () => {
      const post = {
        id: 'error:post',
        tenantId: 'error:tenant',
        authorId: 'error:author'
      };

      mockCacheManager.invalidate.mockRejectedValue(new Error('Invalidation failed'));

      // Should not throw
      invalidationService.emit('post:created', post);

      await new Promise(resolve => setTimeout(resolve, 10));

      // Error should be handled internally
      expect(mockCacheManager.invalidate).toHaveBeenCalled();
    });

    it('should emit error events on invalidation failures', async () => {
      const errorSpy = vi.fn();
      invalidationService.on('invalidation:error', errorSpy);

      const request = {
        type: 'patterns',
        patterns: ['error:pattern:*']
      };

      mockCacheManager.invalidate.mockRejectedValue(new Error('Test error'));

      await invalidationService.processInvalidation(request);

      expect(errorSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          request,
          error: expect.any(Error)
        })
      );
    });
  });
});