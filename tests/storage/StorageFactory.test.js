import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { StorageFactory, LocalStorageProvider, S3StorageProvider } from '../../lib/storage/index.js';

describe('StorageFactory', () => {
  describe('createProvider', () => {
    it('should create LocalStorageProvider', () => {
      const provider = StorageFactory.createProvider('local', {
        basePath: './test-storage-path',
        baseUrl: '/uploads'
      });
      
      expect(provider).toBeInstanceOf(LocalStorageProvider);
      expect(provider.basePath).toBe('./test-storage-path');
      expect(provider.baseUrl).toBe('/uploads');
    });

    it('should create S3StorageProvider', () => {
      const provider = StorageFactory.createProvider('s3', {
        bucket: 'test-bucket',
        region: 'us-west-2'
      });
      
      expect(provider).toBeInstanceOf(S3StorageProvider);
      expect(provider.bucket).toBe('test-bucket');
      expect(provider.region).toBe('us-west-2');
    });

    it('should throw error for unknown provider type', () => {
      expect(() => {
        StorageFactory.createProvider('unknown');
      }).toThrow('Unknown storage provider type: unknown');
    });
  });

  describe('registerProvider', () => {
    it('should register custom provider', () => {
      class CustomProvider {
        constructor(config) {
          this.config = config;
        }
      }
      
      StorageFactory.registerProvider('custom', CustomProvider);
      
      const provider = StorageFactory.createProvider('custom', { test: 'config' });
      expect(provider).toBeInstanceOf(CustomProvider);
      expect(provider.config).toEqual({ test: 'config' });
    });
  });

  describe('getAvailableProviders', () => {
    it('should return available provider types', () => {
      const providers = StorageFactory.getAvailableProviders();
      expect(providers).toContain('local');
      expect(providers).toContain('s3');
    });
  });

  describe('createFromEnv', () => {
    let originalEnv;

    beforeEach(() => {
      originalEnv = { ...process.env };
    });

    afterEach(() => {
      process.env = originalEnv;
    });

    it('should create local provider from environment', () => {
      process.env.STORAGE_TYPE = 'local';
      process.env.LOCAL_STORAGE_PATH = './custom-storage-path';
      process.env.LOCAL_STORAGE_URL = '/custom/url';
      
      const provider = StorageFactory.createFromEnv();
      
      expect(provider).toBeInstanceOf(LocalStorageProvider);
      expect(provider.basePath).toBe('./custom-storage-path');
      expect(provider.baseUrl).toBe('/custom/url');
    });

    it('should create S3 provider from environment', () => {
      process.env.STORAGE_TYPE = 's3';
      process.env.S3_BUCKET = 'env-test-bucket';
      process.env.S3_REGION = 'eu-west-1';
      process.env.AWS_ACCESS_KEY_ID = 'test-key';
      process.env.AWS_SECRET_ACCESS_KEY = 'test-secret';
      
      const provider = StorageFactory.createFromEnv();
      
      expect(provider).toBeInstanceOf(S3StorageProvider);
      expect(provider.bucket).toBe('env-test-bucket');
      expect(provider.region).toBe('eu-west-1');
    });

    it('should default to local provider when no STORAGE_TYPE is set', () => {
      delete process.env.STORAGE_TYPE;
      
      const provider = StorageFactory.createFromEnv();
      
      expect(provider).toBeInstanceOf(LocalStorageProvider);
    });

    it('should throw error for unsupported storage type', () => {
      process.env.STORAGE_TYPE = 'unsupported';
      
      expect(() => {
        StorageFactory.createFromEnv();
      }).toThrow('Unsupported storage type: unsupported');
    });
  });
});