/**
 * Request Validation Middleware Tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createRequestValidationMiddleware, RequestValidationService } from '../../middleware/requestValidation.js';

describe('Request Validation Middleware', () => {
  let req, res, next;
  let validationService;

  beforeEach(() => {
    req = {
      method: 'GET',
      headers: {},
      path: '/api/test',
      originalUrl: '/api/test',
      protocol: 'https',
      get: vi.fn((header) => {
        const headers = { host: 'api.example.com' };
        return headers[header.toLowerCase()];
      }),
      route: { path: '/api/test' },
      query: {},
      body: null,
      user: { tenantId: 'test-tenant' }
    };

    res = {
      status: vi.fn().mockReturnThis(),
      json: vi.fn()
    };

    next = vi.fn();

    validationService = new RequestValidationService();
  });

  describe('Request Size Validation', () => {
    it('should allow requests within size limit', () => {
      req.headers['content-length'] = '1024'; // 1KB

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should reject requests exceeding size limit', () => {
      req.headers['content-length'] = '11000000'; // 11MB (exceeds 10MB default)

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).not.toHaveBeenCalled();
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: expect.stringContaining('Request size'),
          details: expect.objectContaining({
            currentSize: 11000000,
            maxSize: 10485760
          })
        }
      });
    });

    it('should validate body size for string bodies', () => {
      req.body = 'x'.repeat(11000000); // 11MB string

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });
  });

  describe('Content Type Validation', () => {
    it('should allow valid content types for POST requests', () => {
      req.method = 'POST';
      req.headers['content-type'] = 'application/json';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should reject invalid content types', () => {
      req.method = 'POST';
      req.headers['content-type'] = 'application/x-malicious';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: expect.stringContaining('Content type'),
          details: expect.objectContaining({
            contentType: 'application/x-malicious'
          })
        }
      });
    });

    it('should require content type for POST requests', () => {
      req.method = 'POST';
      // No content-type header

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should skip content type validation for GET requests', () => {
      req.method = 'GET';
      // No content-type header

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  describe('Header Validation', () => {
    it('should validate authorization header format', () => {
      req.headers.authorization = 'InvalidFormat token123';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: 'Invalid Authorization header format',
          details: expect.objectContaining({
            field: 'authorization'
          })
        }
      });
    });

    it('should accept valid Bearer token format', () => {
      req.headers.authorization = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should validate tenant ID header format', () => {
      req.headers['x-tenant-id'] = 'invalid@tenant#id';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should reject oversized headers', () => {
      req.headers['user-agent'] = 'x'.repeat(10000); // Very long user agent

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });
  });

  describe('URL Length Validation', () => {
    it('should reject URLs exceeding maximum length', () => {
      req.originalUrl = '/api/test?' + 'x'.repeat(3000);

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: expect.stringContaining('URL length'),
          details: expect.objectContaining({
            field: 'url'
          })
        }
      });
    });
  });

  describe('SQL Injection Detection', () => {
    it('should detect SQL injection in query parameters', () => {
      req.query = { search: "'; DROP TABLE users; --" };

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: 'Potential SQL injection detected',
          details: expect.objectContaining({
            field: 'query.search'
          })
        }
      });
    });

    it('should detect SQL injection in request body', () => {
      req.body = { 
        title: 'Normal title',
        content: 'SELECT * FROM users WHERE id = 1 OR 1=1'
      };

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should allow legitimate SQL-like content when disabled', () => {
      req.body = { content: 'This is about SELECT statements in SQL' };

      const middleware = validationService.createMiddleware({
        enableSqlInjectionCheck: false
      });
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  describe('XSS Detection', () => {
    it('should detect XSS attempts in query parameters', () => {
      req.query = { message: '<script>alert("xss")</script>' };

      const middleware = validationService.createMiddleware({
        enableSqlInjectionCheck: false // Disable SQL injection to test XSS specifically
      });
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: 'Potential XSS attempt detected',
          details: expect.objectContaining({
            field: 'query.message'
          })
        }
      });
    });

    it('should detect XSS in request body', () => {
      req.body = { 
        comment: '<img src="x" onerror="alert(1)">'
      };

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should detect iframe injection', () => {
      req.body = { 
        content: '<iframe src="javascript:alert(1)"></iframe>'
      };

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });
  });

  describe('File Upload Validation', () => {
    it('should validate file sizes', () => {
      req.files = [
        {
          originalname: 'test.jpg',
          size: 60 * 1024 * 1024 // 60MB (exceeds 50MB default)
        }
      ];

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: expect.stringContaining('File 1 size exceeds'),
          details: expect.objectContaining({
            fileIndex: 0,
            currentSize: 60 * 1024 * 1024
          })
        }
      });
    });

    it('should validate file extensions', () => {
      req.files = [
        {
          originalname: 'malicious.exe',
          size: 1024
        }
      ];

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: expect.stringContaining('File 1 type not allowed'),
          details: expect.objectContaining({
            extension: '.exe'
          })
        }
      });
    });

    it('should allow valid file uploads', () => {
      req.files = [
        {
          originalname: 'document.pdf',
          size: 1024 * 1024 // 1MB
        }
      ];

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  describe('API Version Validation', () => {
    it('should validate API version header', () => {
      req.headers['api-version'] = 'v99';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: 'Invalid API version',
          details: expect.objectContaining({
            version: 'v99',
            supportedVersions: ['v1', 'v2', '1.0', '2.0']
          })
        }
      });
    });

    it('should accept valid API versions', () => {
      req.headers['api-version'] = 'v1';

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  describe('Tenant Context Validation', () => {
    it('should require tenant context for non-public endpoints', () => {
      req.user = null; // No authenticated user
      req.headers['x-tenant-id'] = undefined; // No tenant header
      req.method = 'GET'; // Use GET to avoid content-type validation
      req.path = '/api/private'; // Non-public endpoint

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: {
          code: 'REQUEST_VALIDATION_ERROR',
          message: 'Tenant context is required',
          details: expect.objectContaining({
            field: 'x-tenant-id'
          })
        }
      });
    });

    it('should skip tenant validation for public endpoints', () => {
      req.path = '/public/health';
      req.user = null;

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should skip tenant validation for OPTIONS requests', () => {
      req.method = 'OPTIONS';
      req.user = null;

      const middleware = validationService.createMiddleware();
      middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  describe('Custom Validation Configurations', () => {
    it('should apply custom size limits', () => {
      req.headers['content-length'] = '2000000'; // 2MB

      const middleware = validationService.createMiddleware({
        maxRequestSize: 1024 * 1024 // 1MB limit
      });
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should apply custom content type restrictions', () => {
      req.method = 'POST';
      req.headers['content-type'] = 'application/json';

      const middleware = validationService.createMiddleware({
        allowedContentTypes: ['text/plain'] // Only allow text/plain
      });
      middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });
  });
});