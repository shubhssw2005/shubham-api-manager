/**
 * CloudFront Signed URL Service Unit Tests
 */

import { describe, test, expect, beforeEach, vi } from 'vitest';
import CloudFrontSignedURLService from '../../../services/CloudFrontSignedURLService.js';
import fs from 'fs';
import crypto from 'crypto';

// Mock fs module
vi.mock('fs');

describe('CloudFrontSignedURLService', () => {
  let service;
  let mockPrivateKey;

  beforeEach(() => {
    // Generate a test RSA key pair
    const keyPair = crypto.generateKeyPairSync('rsa', {
      modulusLength: 2048,
      publicKeyEncoding: { type: 'spki', format: 'pem' },
      privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
    });

    mockPrivateKey = keyPair.privateKey;

    // Mock fs.readFileSync
    vi.mocked(fs.readFileSync).mockReturnValue(mockPrivateKey);
    vi.mocked(fs.existsSync).mockReturnValue(true);

    service = new CloudFrontSignedURLService({
      distributionDomain: 'test.cloudfront.net',
      keyPairId: 'test-key-pair-id',
      privateKeyPath: '/test/private-key.pem',
      defaultExpiration: 3600
    });
  });

  describe('Constructor', () => {
    test('should initialize with valid configuration', () => {
      expect(service.distributionDomain).toBe('test.cloudfront.net');
      expect(service.keyPairId).toBe('test-key-pair-id');
      expect(service.defaultExpiration).toBe(3600);
      expect(service.privateKey).toBe(mockPrivateKey);
    });

    test('should throw error with missing configuration', () => {
      expect(() => {
        new CloudFrontSignedURLService({});
      }).toThrow('CloudFront configuration incomplete');
    });

    test('should load private key from environment variable', () => {
      process.env.CLOUDFRONT_PRIVATE_KEY = mockPrivateKey.replace(/\n/g, '\\n');
      
      const serviceFromEnv = new CloudFrontSignedURLService({
        distributionDomain: 'test.cloudfront.net',
        keyPairId: 'test-key-pair-id'
      });
      
      expect(serviceFromEnv.privateKey).toBe(mockPrivateKey);
      
      delete process.env.CLOUDFRONT_PRIVATE_KEY;
    });
  });

  describe('generateSignedURL', () => {
    test('should generate signed URL with default expiration', () => {
      const resourcePath = '/test-media/image.jpg';
      const signedUrl = service.generateSignedURL(resourcePath);
      
      expect(signedUrl).toContain('https://test.cloudfront.net/test-media/image.jpg');
      expect(signedUrl).toContain('Expires=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=test-key-pair-id');
    });

    test('should generate signed URL with custom expiration', () => {
      const resourcePath = '/test-media/image.jpg';
      const customExpiration = 7200;
      const signedUrl = service.generateSignedURL(resourcePath, {
        expiresIn: customExpiration
      });
      
      const url = new URL(signedUrl);
      const expires = parseInt(url.searchParams.get('Expires'));
      const expectedExpires = Math.floor(Date.now() / 1000) + customExpiration;
      
      expect(expires).toBeCloseTo(expectedExpires, -2);
    });

    test('should generate signed URL with IP restriction', () => {
      const resourcePath = '/test-media/image.jpg';
      const signedUrl = service.generateSignedURL(resourcePath, {
        ipAddress: '192.168.1.100'
      });
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=test-key-pair-id');
    });

    test('should generate signed URL with custom policy', () => {
      const resourcePath = '/test-media/image.jpg';
      const customPolicy = {
        Statement: [{
          Resource: `https://test.cloudfront.net${resourcePath}`,
          Condition: {
            DateLessThan: {
              'AWS:EpochTime': Math.floor(Date.now() / 1000) + 3600
            }
          }
        }]
      };
      
      const signedUrl = service.generateSignedURL(resourcePath, {
        policy: customPolicy
      });
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
      expect(signedUrl).toContain('Key-Pair-Id=test-key-pair-id');
    });
  });

  describe('generateTenantMediaURL', () => {
    test('should generate tenant-specific media URL', () => {
      const tenantId = 'tenant-123';
      const mediaPath = 'images/logo.png';
      
      const signedUrl = service.generateTenantMediaURL(tenantId, mediaPath);
      
      expect(signedUrl).toContain(`/tenants/${tenantId}/media/${mediaPath}`);
      expect(signedUrl).toContain('Signature=');
    });

    test('should generate tenant media URL with custom options', () => {
      const tenantId = 'tenant-456';
      const mediaPath = 'documents/report.pdf';
      const options = {
        expiresIn: 1800,
        ipAddress: '10.0.0.1'
      };
      
      const signedUrl = service.generateTenantMediaURL(tenantId, mediaPath, options);
      
      expect(signedUrl).toContain(`/tenants/${tenantId}/media/${mediaPath}`);
      expect(signedUrl).toContain('Policy=');
    });
  });

  describe('generateBulkSignedURLs', () => {
    test('should generate multiple signed URLs', () => {
      const resources = [
        '/media/image1.jpg',
        '/media/image2.jpg',
        '/media/document.pdf'
      ];
      
      const results = service.generateBulkSignedURLs(resources);
      
      expect(results).toHaveLength(3);
      results.forEach((result, index) => {
        expect(result.resource).toBe(resources[index]);
        expect(result.signedUrl).toContain('https://test.cloudfront.net');
        expect(result.signedUrl).toContain('Signature=');
      });
    });

    test('should generate bulk URLs with custom options', () => {
      const resources = ['/media/image1.jpg', '/media/image2.jpg'];
      const options = { expiresIn: 1800 };
      
      const results = service.generateBulkSignedURLs(resources, options);
      
      expect(results).toHaveLength(2);
      results.forEach(result => {
        const url = new URL(result.signedUrl);
        const expires = parseInt(url.searchParams.get('Expires'));
        const expectedExpires = Math.floor(Date.now() / 1000) + 1800;
        expect(expires).toBeCloseTo(expectedExpires, -2);
      });
    });
  });

  describe('createCustomPolicy', () => {
    test('should create policy with expiration only', () => {
      const resources = ['https://test.cloudfront.net/media/image.jpg'];
      const expiration = Math.floor(Date.now() / 1000) + 3600;
      
      const policy = service.createCustomPolicy(resources, { expiration });
      
      expect(policy.Statement).toHaveLength(1);
      expect(policy.Statement[0].Resource).toBe(resources[0]);
      expect(policy.Statement[0].Condition.DateLessThan['AWS:EpochTime']).toBe(expiration);
    });

    test('should create policy with IP restriction', () => {
      const resources = ['https://test.cloudfront.net/media/image.jpg'];
      const options = {
        expiration: Math.floor(Date.now() / 1000) + 3600,
        ipAddress: '192.168.1.0/24'
      };
      
      const policy = service.createCustomPolicy(resources, options);
      
      expect(policy.Statement[0].Condition.IpAddress['AWS:SourceIp']).toBe(options.ipAddress);
    });

    test('should create policy with time window', () => {
      const resources = ['https://test.cloudfront.net/media/image.jpg'];
      const options = {
        expiration: Math.floor(Date.now() / 1000) + 3600,
        dateGreaterThan: Math.floor(Date.now() / 1000) + 300
      };
      
      const policy = service.createCustomPolicy(resources, options);
      
      expect(policy.Statement[0].Condition.DateGreaterThan['AWS:EpochTime']).toBe(options.dateGreaterThan);
    });

    test('should create policy with user agent restriction', () => {
      const resources = ['https://test.cloudfront.net/media/image.jpg'];
      const options = {
        expiration: Math.floor(Date.now() / 1000) + 3600,
        userAgent: 'MyApp/1.0'
      };
      
      const policy = service.createCustomPolicy(resources, options);
      
      expect(policy.Statement[0].Condition.StringLike['AWS:UserAgent']).toBe(options.userAgent);
    });
  });

  describe('URL validation', () => {
    test('should validate unexpired signed URL', () => {
      const resourcePath = '/test-media/image.jpg';
      const signedUrl = service.generateSignedURL(resourcePath, {
        expiresIn: 3600
      });
      
      const isValid = service.isURLValid(signedUrl);
      expect(isValid).toBe(true);
    });

    test('should invalidate expired signed URL', () => {
      const resourcePath = '/test-media/image.jpg';
      const pastExpiration = Math.floor(Date.now() / 1000) - 3600; // 1 hour ago
      
      // Manually create an expired URL for testing
      const expiredUrl = `https://test.cloudfront.net${resourcePath}?Expires=${pastExpiration}&Signature=test&Key-Pair-Id=test`;
      
      const isValid = service.isURLValid(expiredUrl);
      expect(isValid).toBe(false);
    });

    test('should handle invalid URL format', () => {
      const isValid = service.isURLValid('invalid-url');
      expect(isValid).toBe(false);
    });

    test('should extract expiration from signed URL', () => {
      const resourcePath = '/test-media/image.jpg';
      const signedUrl = service.generateSignedURL(resourcePath, {
        expiresIn: 3600
      });
      
      const expiration = service.getURLExpiration(signedUrl);
      expect(expiration).toBeInstanceOf(Date);
      expect(expiration.getTime()).toBeGreaterThan(Date.now());
    });

    test('should return null for invalid URL expiration', () => {
      const expiration = service.getURLExpiration('invalid-url');
      expect(expiration).toBeNull();
    });
  });

  describe('base64UrlEncode', () => {
    test('should encode string for CloudFront URL safety', () => {
      const testString = 'test+string/with=padding';
      const encoded = service.base64UrlEncode(testString);
      
      expect(encoded).not.toContain('+');
      expect(encoded).not.toContain('/');
      expect(encoded).not.toContain('=');
    });
  });

  describe('signPolicy', () => {
    test('should sign policy string', () => {
      const policy = JSON.stringify({
        Statement: [{
          Resource: 'https://test.cloudfront.net/test',
          Condition: {
            DateLessThan: {
              'AWS:EpochTime': Math.floor(Date.now() / 1000) + 3600
            }
          }
        }]
      });
      
      const signature = service.signPolicy(policy);
      
      expect(signature).toBeTypeOf('string');
      expect(signature.length).toBeGreaterThan(0);
      expect(signature).not.toContain('+');
      expect(signature).not.toContain('/');
      expect(signature).not.toContain('=');
    });
  });

  describe('Time-based methods', () => {
    test('should generate IP restricted URL', () => {
      const resourcePath = '/test-media/image.jpg';
      const clientIP = '203.0.113.1';
      
      const signedUrl = service.generateIPRestrictedURL(resourcePath, clientIP);
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
    });

    test('should generate time window URL', () => {
      const resourcePath = '/test-media/image.jpg';
      const startTime = Date.now() + 60000; // 1 minute from now
      const endTime = Date.now() + 3660000; // 1 hour 1 minute from now
      
      const signedUrl = service.generateTimeWindowURL(resourcePath, startTime, endTime);
      
      expect(signedUrl).toContain('Policy=');
      expect(signedUrl).toContain('Signature=');
    });
  });
});