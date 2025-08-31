import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createMocks } from 'node-mocks-http';
import handler from '../../pages/api/media/presigned-url.js';

// Mock the PresignedURLService
vi.mock('../../services/PresignedURLService.js', () => {
  return {
    default: vi.fn(() => ({
      generateUploadURL: vi.fn()
    }))
  };
});

// Mock JWT verification
vi.mock('../../lib/jwt.js', () => ({
  verifyToken: vi.fn()
}));

import PresignedURLService from '../../services/PresignedURLService.js';
import { verifyToken } from '../../lib/jwt.js';

describe('/api/media/presigned-url', () => {
  let mockService;

  beforeEach(() => {
    mockService = {
      generateUploadURL: vi.fn()
    };
    PresignedURLService.mockImplementation(() => mockService);
    
    verifyToken.mockReturnValue({
      userId: 'user-123',
      tenantId: 'tenant-456'
    });
  });

  it('should generate presigned URL successfully', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer valid-token'
      },
      body: {
        originalName: 'test-image.jpg',
        contentType: 'image/jpeg',
        size: 1024 * 1024,
        metadata: { description: 'Test image' }
      }
    });

    const mockResult = {
      uploadType: 'single',
      uploadUrl: 'https://mock-presigned-url.com',
      s3Key: 'tenants/tenant-456/users/user-123/media/test-image.jpg',
      bucket: 'test-bucket',
      expiresIn: 300
    };

    mockService.generateUploadURL.mockResolvedValue(mockResult);

    await handler(req, res);

    expect(res._getStatusCode()).toBe(200);
    expect(JSON.parse(res._getData())).toEqual({
      success: true,
      data: mockResult
    });

    expect(mockService.generateUploadURL).toHaveBeenCalledWith({
      tenantId: 'tenant-456',
      originalName: 'test-image.jpg',
      contentType: 'image/jpeg',
      size: 1024 * 1024,
      userId: 'user-123',
      metadata: { description: 'Test image' }
    });
  });

  it('should return 405 for non-POST requests', async () => {
    const { req, res } = createMocks({
      method: 'GET'
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(405);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Method not allowed',
      message: 'Only POST requests are supported'
    });
  });

  it('should return 401 for missing authorization', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      body: {
        originalName: 'test-image.jpg',
        contentType: 'image/jpeg',
        size: 1024 * 1024
      }
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(401);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Unauthorized',
      message: 'Authentication token is required'
    });
  });

  it('should return 400 for missing required fields', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer valid-token'
      },
      body: {
        originalName: 'test-image.jpg'
        // Missing contentType and size
      }
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(400);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Bad Request',
      message: 'originalName, contentType, and size are required'
    });
  });

  it('should handle service validation errors', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer valid-token'
      },
      body: {
        originalName: 'test-image.jpg',
        contentType: 'image/jpeg',
        size: 1024 * 1024
      }
    });

    mockService.generateUploadURL.mockRejectedValue(
      new Error('File type image/jpeg is not allowed')
    );

    await handler(req, res);

    expect(res._getStatusCode()).toBe(400);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Bad Request',
      message: 'File type image/jpeg is not allowed'
    });
  });

  it('should handle JWT verification errors', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer invalid-token'
      },
      body: {
        originalName: 'test-image.jpg',
        contentType: 'image/jpeg',
        size: 1024 * 1024
      }
    });

    verifyToken.mockImplementation(() => {
      throw new Error('Invalid token');
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(401);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Unauthorized',
      message: 'Invalid authentication token'
    });
  });

  it('should handle service internal errors', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer valid-token'
      },
      body: {
        originalName: 'test-image.jpg',
        contentType: 'image/jpeg',
        size: 1024 * 1024
      }
    });

    mockService.generateUploadURL.mockRejectedValue(
      new Error('AWS S3 service unavailable')
    );

    await handler(req, res);

    expect(res._getStatusCode()).toBe(500);
    expect(JSON.parse(res._getData())).toEqual({
      error: 'Internal Server Error',
      message: 'Failed to generate presigned URL'
    });
  });

  it('should handle multipart upload response', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      headers: {
        authorization: 'Bearer valid-token'
      },
      body: {
        originalName: 'large-video.mp4',
        contentType: 'video/mp4',
        size: 200 * 1024 * 1024 // 200MB
      }
    });

    const mockMultipartResult = {
      uploadType: 'multipart',
      uploadId: 'test-upload-id',
      s3Key: 'tenants/tenant-456/users/user-123/media/large-video.mp4',
      bucket: 'test-bucket',
      partSize: 5 * 1024 * 1024,
      partCount: 40,
      partUrls: [
        {
          partNumber: 1,
          uploadUrl: 'https://mock-part-1-url.com',
          minSize: 5 * 1024 * 1024,
          maxSize: 5 * 1024 * 1024
        }
      ],
      expiresIn: 300,
      completeUrl: '/api/media/complete-multipart',
      abortUrl: '/api/media/abort-multipart'
    };

    mockService.generateUploadURL.mockResolvedValue(mockMultipartResult);

    await handler(req, res);

    expect(res._getStatusCode()).toBe(200);
    expect(JSON.parse(res._getData())).toEqual({
      success: true,
      data: mockMultipartResult
    });
  });
});