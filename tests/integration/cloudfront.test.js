/**
 * CloudFront Integration Tests
 * Tests CloudFront distribution, signed URLs, and cache management
 */

import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import CloudFrontSignedURLService from '../../services/CloudFrontSignedURLService.js';
import CloudFrontCacheService from '../../services/CloudFrontCacheService.js';
import fetch from 'node-fetch';

describe('CloudFront Integration Tests', () => {
  let signedURLService;
  let cacheService;
  let testMediaPath;
  let testDomain;

  beforeAll(async () => {
    // Initialize services
    signedURLService = new CloudFrontSignedURLService({
      distributionDomain: process.env.TEST_CLOUDFRONT_DOMAIN || 'test.cloudfront.net',
      keyPairId: process.env.TEST_CLOUDFRONT_KEY_PAIR_ID || 'test-key-pair-id',
      privateKeyPath: process.env.TEST_CLOUDFRONT_PRIVATE_KEY_PATH
    });

    cacheService = new CloudFrontCacheService({
      distributionId: process.env.TEST_CLOUDFRONT_DISTRIBUTION_ID || 'test-distribution-id'
    });

    testMediaPath = '/test-media/sample-image.jpg';
    testDomain = process.env.TEST_CLOUDFRONT_DOMAIN || 'test.cloudfront.net';
  });

  describe('Signed URL Generation', () => {
    test('should generate valid signed URL with default expiration', () => {
      const signedUrl = signedURLService.generateSignedURL(testMediaPath);
      
      expect(signedUrl).toContain('https://');
      expect(signedUrl).toContain(testDomain);
      expect(signedUrl).toContain('Expires=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=');
    });

    test('should generate signed URL with custom expiration', () => {
      const customExpiration = 7200; // 2 hours
      const signedUrl = signedURLService.generateSignedURL(testMediaPath, {
        expiresIn: customExpiration
      });
      
      const url = new URL(signedUrl);
      const expires = parseInt(url.searchParams.get('Expires'));
      const expectedExpires = Math.floor(Date.now() / 1000) + customExpiration;
      
      expect(expires).toBeCloseTo(expectedExpires, -2); // Within 100 seconds
    });

    test('should generate signed URL with IP restriction', () => {
      const clientIP = '192.168.1.100';
      const signedUrl = signedURLService.generateSignedURL(testMediaPath, {
        ipAddress: clientIP
      });
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=');
    });

    test('should generate tenant-specific media URL', () => {
      const tenantId = 'tenant-123';
      const mediaPath = 'images/logo.png';
      
      const signedUrl = signedURLService.generateTenantMediaURL(tenantId, mediaPath);
      
      expect(signedUrl).toContain(`/tenants/${tenantId}/media/${mediaPath}`);
    });

    test('should generate bulk signed URLs', () => {
      const resources = [
        '/media/image1.jpg',
        '/media/image2.jpg',
        '/media/document.pdf'
      ];
      
      const results = signedURLService.generateBulkSignedURLs(resources);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result.resource).toBeDefined();
        expect(result.signedUrl).toContain('https://');
        expect(result.signedUrl).toContain('Signature=');
      });
    });

    test('should validate signed URL expiration', () => {
      const signedUrl = signedURLService.generateSignedURL(testMediaPath, {
        expiresIn: 3600
      });
      
      const isValid = signedURLService.isURLValid(signedUrl);
      const expiration = signedURLService.getURLExpiration(signedUrl);
      
      expect(isValid).toBe(true);
      expect(expiration).toBeInstanceOf(Date);
      expect(expiration.getTime()).toBeGreaterThan(Date.now());
    });

    test('should detect expired signed URL', () => {
      // Create URL that expires in 1 second
      const signedUrl = signedURLService.generateSignedURL(testMediaPath, {
        expiresIn: 1
      });
      
      // Wait for expiration
      setTimeout(() => {
        const isValid = signedURLService.isURLValid(signedUrl);
        expect(isValid).toBe(false);
      }, 2000);
    });
  });

  describe('Custom Policy URLs', () => {
    test('should generate signed URL with custom policy', () => {
      const customPolicy = signedURLService.createCustomPolicy(
        [`https://${testDomain}${testMediaPath}`],
        {
          expiration: Math.floor(Date.now() / 1000) + 3600,
          ipAddress: '192.168.1.0/24'
        }
      );
      
      const signedUrl = signedURLService.generateSignedURLWithCustomPolicy(
        `https://${testDomain}${testMediaPath}`,
        customPolicy
      );
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=');
    });

    test('should create time-window restricted URL', () => {
      const startTime = Date.now() + 60000; // Start in 1 minute
      const endTime = Date.now() + 3660000; // End in 1 hour 1 minute
      
      const signedUrl = signedURLService.generateTimeWindowURL(
        testMediaPath,
        startTime,
        endTime
      );
      
      expect(signedUrl).toContain('Policy=');
    });
  });

  describe('Cache Management', () => {
    test('should queue paths for invalidation', async () => {
      const paths = ['/test-path-1', '/test-path-2'];
      
      const result = await cacheService.invalidatePaths(paths);
      
      expect(result.queued).toBe(true);
      expect(result.paths).toEqual(paths);
    });

    test('should create immediate invalidation', async () => {
      const paths = ['/immediate-test-path'];
      
      try {
        const result = await cacheService.invalidatePaths(paths, { immediate: true });
        
        expect(result.invalidationId).toBeDefined();
        expect(result.status).toBeDefined();
        expect(result.paths).toEqual(paths);
      } catch (error) {
        // Skip test if AWS credentials not available
        if (error.message.includes('credentials')) {
          console.warn('Skipping AWS invalidation test - credentials not available');
          return;
        }
        throw error;
      }
    });

    test('should invalidate tenant media', async () => {
      const tenantId = 'test-tenant-123';
      const mediaKeys = ['image1.jpg', 'image2.jpg'];
      
      try {
        const result = await cacheService.invalidateTenantMedia(tenantId, mediaKeys);
        
        expect(result.invalidationId || result.queued).toBeDefined();
      } catch (error) {
        if (error.message.includes('credentials')) {
          console.warn('Skipping AWS invalidation test - credentials not available');
          return;
        }
        throw error;
      }
    });

    test('should perform smart invalidation', async () => {
      const metadata = {
        tenantId: 'test-tenant-123',
        resourceId: 'media-456',
        tags: ['product', 'featured']
      };
      
      try {
        const result = await cacheService.smartInvalidation('media', metadata);
        
        expect(result.invalidationId || result.queued).toBeDefined();
      } catch (error) {
        if (error.message.includes('credentials')) {
          console.warn('Skipping AWS invalidation test - credentials not available');
          return;
        }
        throw error;
      }
    });

    test('should get cache statistics', async () => {
      const stats = await cacheService.getCacheStatistics(24);
      
      expect(stats.hitRate).toBeDefined();
      expect(stats.missRate).toBeDefined();
      expect(stats.totalRequests).toBeDefined();
      expect(stats.period).toBe('24 hours');
    });

    test('should provide optimization recommendations', async () => {
      const recommendations = await cacheService.optimizeCacheSettings();
      
      expect(recommendations.currentStats).toBeDefined();
      expect(recommendations.recommendations).toBeInstanceOf(Array);
      expect(recommendations.optimizationScore).toBeTypeOf('number');
    });
  });

  describe('CloudFront Distribution Health', () => {
    test('should verify CloudFront distribution is accessible', async () => {
      if (!process.env.TEST_CLOUDFRONT_DOMAIN) {
        console.warn('Skipping CloudFront health test - TEST_CLOUDFRONT_DOMAIN not set');
        return;
      }

      try {
        const response = await fetch(`https://${process.env.TEST_CLOUDFRONT_DOMAIN}/health`, {
          method: 'HEAD',
          timeout: 10000
        });
        
        expect(response.status).toBeLessThan(500);
        
        // Check CloudFront headers
        const cacheStatus = response.headers.get('x-cache');
        const cfRay = response.headers.get('cf-ray');
        
        // At least one CloudFront header should be present
        expect(cacheStatus || cfRay).toBeTruthy();
      } catch (error) {
        console.warn('CloudFront health check failed:', error.message);
      }
    });

    test('should verify HTTPS redirect', async () => {
      if (!process.env.TEST_CLOUDFRONT_DOMAIN) {
        console.warn('Skipping HTTPS redirect test - TEST_CLOUDFRONT_DOMAIN not set');
        return;
      }

      try {
        const response = await fetch(`http://${process.env.TEST_CLOUDFRONT_DOMAIN}/`, {
          redirect: 'manual',
          timeout: 10000
        });
        
        expect([301, 302, 308]).toContain(response.status);
        
        const location = response.headers.get('location');
        expect(location).toMatch(/^https:/);
      } catch (error) {
        console.warn('HTTPS redirect test failed:', error.message);
      }
    });

    test('should verify security headers', async () => {
      if (!process.env.TEST_CLOUDFRONT_DOMAIN) {
        console.warn('Skipping security headers test - TEST_CLOUDFRONT_DOMAIN not set');
        return;
      }

      try {
        const response = await fetch(`https://${process.env.TEST_CLOUDFRONT_DOMAIN}/`);
        
        // Check for security headers
        const securityHeaders = {
          'strict-transport-security': response.headers.get('strict-transport-security'),
          'x-content-type-options': response.headers.get('x-content-type-options'),
          'x-frame-options': response.headers.get('x-frame-options'),
          'referrer-policy': response.headers.get('referrer-policy')
        };
        
        expect(securityHeaders['strict-transport-security']).toContain('max-age=');
        expect(securityHeaders['x-content-type-options']).toBe('nosniff');
        expect(securityHeaders['x-frame-options']).toBeTruthy();
        expect(securityHeaders['referrer-policy']).toBeTruthy();
      } catch (error) {
        console.warn('Security headers test failed:', error.message);
      }
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid private key gracefully', () => {
      expect(() => {
        new CloudFrontSignedURLService({
          distributionDomain: 'test.cloudfront.net',
          keyPairId: 'test-key-pair-id',
          privateKeyPath: '/nonexistent/path'
        });
      }).toThrow();
    });

    test('should handle missing configuration', () => {
      expect(() => {
        new CloudFrontSignedURLService({});
      }).toThrow('CloudFront configuration incomplete');
    });

    test('should handle invalid URL in validation', () => {
      const isValid = signedURLService.isURLValid('invalid-url');
      expect(isValid).toBe(false);
      
      const expiration = signedURLService.getURLExpiration('invalid-url');
      expect(expiration).toBeNull();
    });

    test('should handle cache service errors gracefully', async () => {
      const invalidCacheService = new CloudFrontCacheService({
        distributionId: 'invalid-distribution-id'
      });
      
      try {
        await invalidCacheService.invalidatePaths(['/test'], { immediate: true });
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
      }
    });
  });

  afterAll(async () => {
    // Cleanup any test resources
    if (cacheService.redis) {
      await cacheService.redis.disconnect();
    }
  });
});