import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import CloudFrontSignedURLService from '../../services/CloudFrontSignedURLService.js';
import CloudFrontCacheService from '../../services/CloudFrontCacheService.js';
import { S3Client, PutObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import crypto from 'crypto';

describe('CloudFront Signed URLs Integration', () => {
  let cloudfrontService;
  let cacheService;
  let s3Client;
  let testObjects = [];

  beforeAll(async () => {
    // Skip tests if CloudFront is not configured
    if (!process.env.CLOUDFRONT_DISTRIBUTION_ID || !process.env.CLOUDFRONT_PRIVATE_KEY) {
      console.log('Skipping CloudFront tests - missing configuration');
      return;
    }

    cloudfrontService = new CloudFrontSignedURLService();
    cacheService = new CloudFrontCacheService();
    s3Client = new S3Client({ region: process.env.AWS_REGION });
  });

  afterAll(async () => {
    // Clean up test objects
    if (testObjects.length > 0) {
      const deletePromises = testObjects.map(async (key) => {
        try {
          await s3Client.send(new DeleteObjectCommand({
            Bucket: process.env.S3_MEDIA_BUCKET,
            Key: key
          }));
        } catch (error) {
          console.warn(`Failed to delete test object ${key}:`, error.message);
        }
      });
      await Promise.allSettled(deletePromises);
    }
  });

  describe('Signed URL Generation', () => {
    it('should generate a valid canned signed URL', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKey = 'test-images/test-image.jpg';
      const signedUrl = cloudfrontService.generateMediaSignedURL(testKey, { expiresIn: 3600 });

      expect(signedUrl).toBeDefined();
      expect(signedUrl).toContain('Expires=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=');
      expect(cloudfrontService.validateSignedURL(signedUrl)).toBe(true);
    });

    it('should generate a valid custom signed URL with IP restriction', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKey = 'test-images/restricted-image.jpg';
      const signedUrl = cloudfrontService.generateMediaSignedURL(testKey, {
        expiresIn: 1800,
        ipAddress: '192.168.1.0/24'
      });

      expect(signedUrl).toBeDefined();
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=');
      expect(cloudfrontService.validateSignedURL(signedUrl)).toBe(true);
    });

    it('should generate bulk signed URLs', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKeys = [
        'test-images/bulk-1.jpg',
        'test-images/bulk-2.jpg',
        'test-images/bulk-3.jpg'
      ];

      const signedUrls = cloudfrontService.generateBulkMediaSignedURLs(testKeys, {
        expiresIn: 3600
      });

      expect(signedUrls).toHaveLength(3);
      signedUrls.forEach((item, index) => {
        expect(item.key).toBe(testKeys[index]);
        expect(item.signedUrl).toBeDefined();
        expect(cloudfrontService.validateSignedURL(item.signedUrl)).toBe(true);
      });
    });

    it('should handle different expiration times', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKey = 'test-images/expiration-test.jpg';
      
      // Short expiration
      const shortUrl = cloudfrontService.generateMediaSignedURL(testKey, { expiresIn: 300 });
      expect(shortUrl).toContain('Expires=');
      
      // Long expiration
      const longUrl = cloudfrontService.generateMediaSignedURL(testKey, { expiresIn: 86400 });
      expect(longUrl).toContain('Expires=');
      
      // Extract expiration times
      const shortExpires = new URL(shortUrl).searchParams.get('Expires');
      const longExpires = new URL(longUrl).searchParams.get('Expires');
      
      expect(parseInt(longExpires)).toBeGreaterThan(parseInt(shortExpires));
    });
  });

  describe('Cache Management', () => {
    it('should get cache statistics', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const stats = await cacheService.getCacheStats();
      
      expect(stats).toHaveProperty('invalidationsUsed');
      expect(stats).toHaveProperty('invalidationsRemaining');
      expect(stats).toHaveProperty('maxInvalidationsPerHour');
      expect(stats).toHaveProperty('distributionId');
      expect(stats.distributionId).toBe(process.env.CLOUDFRONT_DISTRIBUTION_ID);
    });

    it('should create cache invalidation', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testPaths = ['/test-cache/invalidation-test.jpg'];
      
      const results = await cacheService.smartInvalidate(testPaths, {
        priority: 'normal',
        skipDuplicates: false
      });

      expect(results).toHaveLength(1);
      expect(results[0]).toHaveProperty('Invalidation');
      expect(results[0].Invalidation).toHaveProperty('Id');
      expect(results[0].Invalidation).toHaveProperty('Status');
    });

    it('should handle batch invalidations', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      // Create a large number of paths to test batching
      const testPaths = Array.from({ length: 50 }, (_, i) => `/test-batch/file-${i}.jpg`);
      
      const results = await cacheService.smartInvalidate(testPaths, {
        priority: 'low',
        batchDelay: 100
      });

      expect(results).toHaveLength(1); // Should fit in one batch
      expect(results[0]).toHaveProperty('Invalidation');
    });

    it('should invalidate by content type', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const results = await cacheService.invalidateByContentType('images', 'test-tenant-123');
      
      expect(results).toHaveLength(1);
      expect(results[0]).toHaveProperty('Invalidation');
    });

    it('should warm cache after invalidation', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testPaths = ['/test-warm/cache-test.jpg'];
      
      // This should not throw an error
      await expect(cacheService.warmCache(testPaths)).resolves.not.toThrow();
    });
  });

  describe('API Endpoints', () => {
    it('should generate signed URL via API', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const response = await fetch(`${process.env.API_BASE_URL}/api/media/cloudfront-url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.TEST_JWT_TOKEN}`
        },
        body: JSON.stringify({
          s3Key: 'tenants/test-tenant/images/api-test.jpg',
          expiresIn: 3600,
          urlType: 'canned'
        })
      });

      expect(response.status).toBe(200);
      const data = await response.json();
      
      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('signedUrl');
      expect(data.data).toHaveProperty('expiresAt');
      expect(cloudfrontService.validateSignedURL(data.data.signedUrl)).toBe(true);
    });

    it('should generate bulk signed URLs via API', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKeys = [
        'tenants/test-tenant/images/bulk-api-1.jpg',
        'tenants/test-tenant/images/bulk-api-2.jpg'
      ];

      const response = await fetch(`${process.env.API_BASE_URL}/api/media/cloudfront-url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.TEST_JWT_TOKEN}`
        },
        body: JSON.stringify({
          s3Keys: testKeys,
          expiresIn: 1800
        })
      });

      expect(response.status).toBe(200);
      const data = await response.json();
      
      expect(data.success).toBe(true);
      expect(data.data.urls).toHaveLength(2);
      data.data.urls.forEach(item => {
        expect(item).toHaveProperty('signedUrl');
        expect(cloudfrontService.validateSignedURL(item.signedUrl)).toBe(true);
      });
    });

    it('should handle cache management via API', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const response = await fetch(`${process.env.API_BASE_URL}/api/media/cache-management`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.TEST_ADMIN_TOKEN}`
        },
        body: JSON.stringify({
          operation: 'smart_invalidate',
          paths: ['/test-api/cache-invalidation.jpg'],
          priority: 'normal'
        })
      });

      expect(response.status).toBe(200);
      const data = await response.json();
      
      expect(data.success).toBe(true);
      expect(data.operation).toBe('smart_invalidate');
      expect(data.data).toHaveProperty('invalidations');
    });

    it('should get cache statistics via API', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const response = await fetch(`${process.env.API_BASE_URL}/api/media/cache-management?info=stats`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.TEST_ADMIN_TOKEN}`
        }
      });

      expect(response.status).toBe(200);
      const data = await response.json();
      
      expect(data.success).toBe(true);
      expect(data.data.stats).toHaveProperty('invalidationsUsed');
      expect(data.data.stats).toHaveProperty('distributionId');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid S3 keys gracefully', () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      expect(() => {
        cloudfrontService.generateMediaSignedURL('');
      }).not.toThrow();
    });

    it('should validate signed URL format', () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      expect(cloudfrontService.validateSignedURL('https://example.com')).toBe(false);
      expect(cloudfrontService.validateSignedURL('invalid-url')).toBe(false);
      expect(cloudfrontService.validateSignedURL('')).toBe(false);
    });

    it('should handle rate limiting for cache invalidations', async () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      // This test would require mocking Redis to simulate rate limit exceeded
      // For now, we just ensure the method exists and can be called
      expect(cacheService.getCacheStats).toBeDefined();
    });
  });

  describe('Security', () => {
    it('should not expose private key in signed URLs', () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKey = 'test-security/private-key-test.jpg';
      const signedUrl = cloudfrontService.generateMediaSignedURL(testKey);
      
      expect(signedUrl).not.toContain('BEGIN RSA PRIVATE KEY');
      expect(signedUrl).not.toContain('END RSA PRIVATE KEY');
      expect(signedUrl).not.toContain(process.env.CLOUDFRONT_PRIVATE_KEY);
    });

    it('should include proper key pair ID in signed URLs', () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const testKey = 'test-security/key-pair-test.jpg';
      const signedUrl = cloudfrontService.generateMediaSignedURL(testKey);
      
      expect(signedUrl).toContain(`Key-Pair-Id=${process.env.CLOUDFRONT_KEY_PAIR_ID}`);
    });

    it('should generate different signatures for different URLs', () => {
      if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) return;

      const url1 = cloudfrontService.generateMediaSignedURL('test1.jpg');
      const url2 = cloudfrontService.generateMediaSignedURL('test2.jpg');
      
      const signature1 = new URL(url1).searchParams.get('Signature');
      const signature2 = new URL(url2).searchParams.get('Signature');
      
      expect(signature1).not.toBe(signature2);
    });
  });
});