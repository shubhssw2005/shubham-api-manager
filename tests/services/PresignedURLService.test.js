import { describe, it, expect, beforeEach, vi } from 'vitest';
import PresignedURLService from '../../services/PresignedURLService.js';

// Mock AWS SDK
vi.mock('@aws-sdk/client-s3', () => ({
  S3Client: vi.fn(() => ({
    send: vi.fn()
  })),
  PutObjectCommand: vi.fn(),
  CreateMultipartUploadCommand: vi.fn(),
  UploadPartCommand: vi.fn(),
  CompleteMultipartUploadCommand: vi.fn(),
  AbortMultipartUploadCommand: vi.fn()
}));

vi.mock('@aws-sdk/s3-request-presigner', () => ({
  getSignedUrl: vi.fn(() => Promise.resolve('https://mock-presigned-url.com'))
}));

describe('PresignedURLService', () => {
  let service;
  let mockS3Client;

  beforeEach(() => {
    // Reset environment variables
    process.env.AWS_REGION = 'us-east-1';
    process.env.AWS_ACCESS_KEY_ID = 'test-key';
    process.env.AWS_SECRET_ACCESS_KEY = 'test-secret';
    process.env.MEDIA_BUCKET = 'test-bucket';

    service = new PresignedURLService();
    mockS3Client = service.s3Client;
  });

  describe('generateUploadURL', () => {
    const validParams = {
      tenantId: 'tenant-123',
      originalName: 'test-image.jpg',
      contentType: 'image/jpeg',
      size: 1024 * 1024, // 1MB
      userId: 'user-456',
      metadata: { description: 'Test image' }
    };

    it('should generate single upload URL for small files', async () => {
      const result = await service.generateUploadURL(validParams);

      expect(result).toMatchObject({
        uploadType: 'single',
        uploadUrl: expect.stringContaining('https://'),
        s3Key: expect.stringContaining('tenants/tenant-123/users/user-456/media/'),
        bucket: 'test-bucket',
        expiresIn: 300
      });
    });

    it('should generate multipart upload URLs for large files', async () => {
      const largeFileParams = {
        ...validParams,
        size: 200 * 1024 * 1024 // 200MB
      };

      // Mock multipart upload creation
      mockS3Client.send.mockResolvedValueOnce({
        UploadId: 'test-upload-id'
      });

      const result = await service.generateUploadURL(largeFileParams);

      expect(result).toMatchObject({
        uploadType: 'multipart',
        uploadId: 'test-upload-id',
        s3Key: expect.stringContaining('tenants/tenant-123/users/user-456/media/'),
        bucket: 'test-bucket',
        partUrls: expect.arrayContaining([
          expect.objectContaining({
            partNumber: expect.any(Number),
            uploadUrl: expect.stringContaining('https://')
          })
        ])
      });
    });

    it('should validate required parameters', async () => {
      const invalidParams = { ...validParams };
      delete invalidParams.tenantId;

      await expect(service.generateUploadURL(invalidParams))
        .rejects.toThrow('Tenant ID is required');
    });

    it('should validate file size limits', async () => {
      const oversizedParams = {
        ...validParams,
        size: 6 * 1024 * 1024 * 1024 // 6GB (exceeds 5GB limit)
      };

      await expect(service.generateUploadURL(oversizedParams))
        .rejects.toThrow('File size exceeds maximum allowed size');
    });

    it('should validate MIME types', async () => {
      const invalidMimeParams = {
        ...validParams,
        contentType: 'application/x-executable'
      };

      await expect(service.generateUploadURL(invalidMimeParams))
        .rejects.toThrow('File type application/x-executable is not allowed');
    });

    it('should detect malicious filenames', async () => {
      const maliciousParams = {
        ...validParams,
        originalName: '../../../etc/passwd'
      };

      await expect(service.generateUploadURL(maliciousParams))
        .rejects.toThrow('Invalid filename detected');
    });
  });

  describe('completeMultipartUpload', () => {
    it('should complete multipart upload successfully', async () => {
      const params = {
        s3Key: 'tenants/tenant-123/users/user-456/media/test-file.jpg',
        uploadId: 'test-upload-id',
        parts: [
          { PartNumber: 1, ETag: 'etag1' },
          { PartNumber: 2, ETag: 'etag2' }
        ],
        tenantId: 'tenant-123'
      };

      mockS3Client.send.mockResolvedValueOnce({
        Location: 'https://test-bucket.s3.amazonaws.com/test-key',
        ETag: 'final-etag',
        VersionId: 'version-123'
      });

      const result = await service.completeMultipartUpload(params);

      expect(result).toMatchObject({
        success: true,
        s3Key: params.s3Key,
        bucket: 'test-bucket',
        location: 'https://test-bucket.s3.amazonaws.com/test-key',
        etag: 'final-etag',
        versionId: 'version-123'
      });
    });

    it('should validate tenant access', async () => {
      const params = {
        s3Key: 'tenants/other-tenant/users/user-456/media/test-file.jpg',
        uploadId: 'test-upload-id',
        parts: [{ PartNumber: 1, ETag: 'etag1' }],
        tenantId: 'tenant-123'
      };

      await expect(service.completeMultipartUpload(params))
        .rejects.toThrow('Unauthorized access to upload');
    });
  });

  describe('abortMultipartUpload', () => {
    it('should abort multipart upload successfully', async () => {
      const params = {
        s3Key: 'tenants/tenant-123/users/user-456/media/test-file.jpg',
        uploadId: 'test-upload-id',
        tenantId: 'tenant-123'
      };

      mockS3Client.send.mockResolvedValueOnce({});

      const result = await service.abortMultipartUpload(params);

      expect(result).toMatchObject({
        success: true,
        message: 'Multipart upload aborted successfully'
      });
    });
  });

  describe('createResumableSession', () => {
    it('should create resumable session successfully', async () => {
      const params = {
        tenantId: 'tenant-123',
        originalName: 'large-video.mp4',
        contentType: 'video/mp4',
        size: 500 * 1024 * 1024, // 500MB
        userId: 'user-456',
        metadata: { description: 'Large video file' }
      };

      const result = await service.createResumableSession(params);

      expect(result).toMatchObject({
        sessionId: expect.any(String),
        s3Key: expect.stringContaining('tenants/tenant-123/users/user-456/media/'),
        resumable: true,
        chunkSize: expect.any(Number),
        totalChunks: expect.any(Number),
        expiresIn: 24 * 60 * 60,
        uploadUrl: expect.stringContaining('/api/media/resumable-upload/'),
        statusUrl: expect.stringContaining('/api/media/resumable-status/')
      });
    });
  });

  describe('utility methods', () => {
    it('should generate valid S3 keys with tenant isolation', () => {
      const key = service.generateS3Key('tenant-123', 'test file.jpg', 'user-456');
      
      expect(key).toMatch(/^tenants\/tenant-123\/users\/user-456\/media\/\d+-[a-f0-9]+-test-file\.jpg$/);
    });

    it('should sanitize filenames properly', () => {
      const sanitized = service.sanitizeFilename('Test File With Spaces & Special!@#$%^&*()_+Characters.jpg');
      
      expect(sanitized).toBe('test-file-with-spaces-special-characters');
      expect(sanitized).not.toContain(' ');
      expect(sanitized).not.toContain('&');
      expect(sanitized).not.toContain('!');
    });

    it('should detect malicious filenames through validation', () => {
      const maliciousFiles = [
        { name: '../../../etc/passwd', shouldThrow: true },
        { name: 'file<script>alert("xss")</script>.jpg', shouldThrow: true },
        { name: 'CON.txt', shouldThrow: true },
        { name: '.hidden-file', shouldThrow: true },
        { name: 'virus.exe', shouldThrow: true },
        { name: 'normal-file.jpg', shouldThrow: false }
      ];

      maliciousFiles.forEach(({ name, shouldThrow }) => {
        const params = {
          tenantId: 'test',
          originalName: name,
          contentType: 'text/plain',
          size: 1024,
          userId: 'user-123'
        };
        
        if (shouldThrow) {
          expect(() => service.validateUploadParams(params)).toThrow('Invalid filename detected');
        } else {
          expect(() => service.validateUploadParams(params)).not.toThrow();
        }
      });
    });

    it('should validate tenant access through multipart operations', async () => {
      // Test valid tenant access
      const validParams = {
        s3Key: 'tenants/tenant-123/users/user-456/media/file.jpg',
        uploadId: 'test-upload-id',
        parts: [{ PartNumber: 1, ETag: 'etag1' }],
        tenantId: 'tenant-123'
      };

      mockS3Client.send.mockResolvedValueOnce({
        Location: 'https://test-bucket.s3.amazonaws.com/test-key',
        ETag: 'final-etag'
      });

      await expect(service.completeMultipartUpload(validParams)).resolves.toBeDefined();

      // Test invalid tenant access
      const invalidParams = {
        ...validParams,
        s3Key: 'tenants/tenant-456/users/user-456/media/file.jpg'
      };

      await expect(service.completeMultipartUpload(invalidParams))
        .rejects.toThrow('Unauthorized access to upload');
    });

    it('should format bytes correctly', () => {
      expect(service.formatBytes(0)).toBe('0 Bytes');
      expect(service.formatBytes(1024)).toBe('1 KB');
      expect(service.formatBytes(1024 * 1024)).toBe('1 MB');
      expect(service.formatBytes(1024 * 1024 * 1024)).toBe('1 GB');
    });
  });

  describe('validation', () => {
    it('should validate upload parameters comprehensively', () => {
      const invalidCases = [
        { params: {}, expectedError: 'Tenant ID is required' },
        { params: { tenantId: 'test' }, expectedError: 'Original filename is required' },
        { params: { tenantId: 'test', originalName: 'file.jpg' }, expectedError: 'Content type is required' },
        { params: { tenantId: 'test', originalName: 'file.jpg', contentType: 'image/jpeg' }, expectedError: 'Valid file size is required' },
        { params: { tenantId: 'test', originalName: 'file.jpg', contentType: 'image/jpeg', size: 0 }, expectedError: 'Valid file size is required' },
        { params: { tenantId: 'test', originalName: 'file.jpg', contentType: 'image/jpeg', size: 1024 }, expectedError: 'User ID is required' }
      ];

      invalidCases.forEach(({ params, expectedError }) => {
        expect(() => service.validateUploadParams(params)).toThrow(expectedError);
      });
    });

    it('should accept valid MIME types', () => {
      const validMimeTypes = [
        'image/jpeg',
        'image/png',
        'video/mp4',
        'audio/mpeg',
        'application/pdf',
        'text/plain'
      ];

      validMimeTypes.forEach(mimeType => {
        const params = {
          tenantId: 'test',
          originalName: 'file.ext',
          contentType: mimeType,
          size: 1024,
          userId: 'user-123'
        };
        
        expect(() => service.validateUploadParams(params)).not.toThrow();
      });
    });
  });
});