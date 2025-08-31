/**
 * Request Validation Middleware
 * Implements request size limits, content validation, and security checks
 */

import { ValidationError } from '../lib/errors/index.js';

class RequestValidationService {
  constructor(options = {}) {
    this.maxRequestSize = options.maxRequestSize || 10 * 1024 * 1024; // 10MB
    this.allowedContentTypes = options.allowedContentTypes || [
      'application/json',
      'application/x-www-form-urlencoded',
      'multipart/form-data',
      'text/plain',
      'application/xml',
      'text/xml'
    ];
    this.maxHeaderSize = options.maxHeaderSize || 8192; // 8KB
    this.maxUrlLength = options.maxUrlLength || 2048; // 2KB
    this.enableSqlInjectionCheck = options.enableSqlInjectionCheck !== false;
    this.enableXssCheck = options.enableXssCheck !== false;
  }

  /**
   * Create request validation middleware
   */
  createMiddleware(options = {}) {
    return (req, res, next) => {
      try {
        // Merge options with instance defaults
        const config = { ...this, ...options };

        // Validate request size
        this.validateRequestSize(req, config);

        // Validate content type
        this.validateContentType(req, config);

        // Validate headers
        this.validateHeaders(req, config);

        // Validate URL length
        this.validateUrlLength(req, config);

        // Security validations
        if (config.enableSqlInjectionCheck) {
          this.checkSqlInjection(req);
        }

        if (config.enableXssCheck) {
          this.checkXssAttempts(req);
        }

        // Validate specific request patterns
        this.validateRequestPatterns(req, config);

        next();

      } catch (error) {
        if (error instanceof ValidationError) {
          res.status(400).json({
            error: {
              code: 'REQUEST_VALIDATION_ERROR',
              message: error.message,
              details: error.details
            }
          });
        } else {
          console.error('Request validation error:', error);
          res.status(500).json({
            error: {
              code: 'INTERNAL_SERVER_ERROR',
              message: 'Request validation failed'
            }
          });
        }
      }
    };
  }

  /**
   * Validate request size
   */
  validateRequestSize(req, config) {
    const contentLength = parseInt(req.headers['content-length'] || '0');
    
    if (contentLength > config.maxRequestSize) {
      throw new ValidationError(
        `Request size ${contentLength} bytes exceeds maximum allowed size ${config.maxRequestSize} bytes`,
        {
          currentSize: contentLength,
          maxSize: config.maxRequestSize,
          field: 'content-length'
        }
      );
    }

    // Check if body exists and validate its size
    if (req.body && typeof req.body === 'string') {
      const bodySize = Buffer.byteLength(req.body, 'utf8');
      if (bodySize > config.maxRequestSize) {
        throw new ValidationError(
          `Request body size ${bodySize} bytes exceeds maximum allowed size ${config.maxRequestSize} bytes`,
          {
            currentSize: bodySize,
            maxSize: config.maxRequestSize,
            field: 'body'
          }
        );
      }
    }
  }

  /**
   * Validate content type
   */
  validateContentType(req, config) {
    const method = req.method.toUpperCase();
    const contentType = req.headers['content-type'] || '';

    // Skip validation for GET, HEAD, DELETE requests
    if (['GET', 'HEAD', 'DELETE', 'OPTIONS'].includes(method)) {
      return;
    }

    // Check if content type is provided for requests with body
    if (['POST', 'PUT', 'PATCH'].includes(method) && !contentType) {
      throw new ValidationError(
        'Content-Type header is required for requests with body',
        {
          method,
          field: 'content-type'
        }
      );
    }

    // Validate against allowed content types
    if (contentType) {
      const baseContentType = contentType.split(';')[0].trim().toLowerCase();
      const isAllowed = config.allowedContentTypes.some(allowed => 
        baseContentType === allowed.toLowerCase()
      );

      if (!isAllowed) {
        throw new ValidationError(
          `Content type '${baseContentType}' is not allowed`,
          {
            contentType: baseContentType,
            allowedTypes: config.allowedContentTypes,
            field: 'content-type'
          }
        );
      }
    }
  }

  /**
   * Validate headers
   */
  validateHeaders(req, config) {
    const headers = req.headers;
    
    // Check total header size
    const headerString = Object.entries(headers)
      .map(([key, value]) => `${key}: ${value}`)
      .join('\r\n');
    
    const headerSize = Buffer.byteLength(headerString, 'utf8');
    
    if (headerSize > config.maxHeaderSize) {
      throw new ValidationError(
        `Total header size ${headerSize} bytes exceeds maximum allowed size ${config.maxHeaderSize} bytes`,
        {
          currentSize: headerSize,
          maxSize: config.maxHeaderSize,
          field: 'headers'
        }
      );
    }

    // Validate specific headers
    this.validateSpecificHeaders(headers);
  }

  /**
   * Validate specific headers
   */
  validateSpecificHeaders(headers) {
    // Validate Authorization header format
    if (headers.authorization) {
      const authHeader = headers.authorization;
      if (!authHeader.startsWith('Bearer ') && !authHeader.startsWith('Basic ')) {
        throw new ValidationError(
          'Invalid Authorization header format',
          {
            field: 'authorization',
            expected: 'Bearer <token> or Basic <credentials>'
          }
        );
      }
    }

    // Validate X-Tenant-ID header
    if (headers['x-tenant-id']) {
      const tenantId = headers['x-tenant-id'];
      if (!/^[a-zA-Z0-9-_]{1,64}$/.test(tenantId)) {
        throw new ValidationError(
          'Invalid X-Tenant-ID header format',
          {
            field: 'x-tenant-id',
            pattern: '^[a-zA-Z0-9-_]{1,64}$'
          }
        );
      }
    }

    // Validate User-Agent header (basic check)
    if (headers['user-agent']) {
      const userAgent = headers['user-agent'];
      if (userAgent.length > 512) {
        throw new ValidationError(
          'User-Agent header too long',
          {
            field: 'user-agent',
            maxLength: 512
          }
        );
      }
    }
  }

  /**
   * Validate URL length
   */
  validateUrlLength(req, config) {
    const fullUrl = req.protocol + '://' + req.get('host') + req.originalUrl;
    
    if (fullUrl.length > config.maxUrlLength) {
      throw new ValidationError(
        `URL length ${fullUrl.length} exceeds maximum allowed length ${config.maxUrlLength}`,
        {
          currentLength: fullUrl.length,
          maxLength: config.maxUrlLength,
          field: 'url'
        }
      );
    }
  }

  /**
   * Check for SQL injection attempts
   */
  checkSqlInjection(req) {
    const sqlPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/i,
      /(;|\-\-|\/\*|\*\/)/,
      /(\b(OR|AND)\b.*=.*)/i,
      /'.*(\bOR\b|\bAND\b).*'/i,
      /\b(WAITFOR|DELAY)\b/i
    ];

    const checkString = (str, field) => {
      if (typeof str !== 'string') return;
      
      for (const pattern of sqlPatterns) {
        if (pattern.test(str)) {
          throw new ValidationError(
            'Potential SQL injection detected',
            {
              field,
              pattern: pattern.toString(),
              suspiciousContent: str.substring(0, 100)
            }
          );
        }
      }
    };

    // Check query parameters
    Object.entries(req.query || {}).forEach(([key, value]) => {
      checkString(value, `query.${key}`);
    });

    // Check body parameters (if JSON)
    if (req.body && typeof req.body === 'object') {
      this.checkObjectForSqlInjection(req.body, 'body');
    }

    // Check URL path
    checkString(req.path, 'path');
  }

  /**
   * Recursively check object for SQL injection
   */
  checkObjectForSqlInjection(obj, prefix = '') {
    Object.entries(obj).forEach(([key, value]) => {
      const fieldPath = prefix ? `${prefix}.${key}` : key;
      
      if (typeof value === 'string') {
        this.checkSqlInjection({ query: { [key]: value } });
      } else if (typeof value === 'object' && value !== null) {
        this.checkObjectForSqlInjection(value, fieldPath);
      }
    });
  }

  /**
   * Check for XSS attempts
   */
  checkXssAttempts(req) {
    const xssPatterns = [
      /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
      /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<img[^>]+src[^>]*>/gi,
      /<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/gi
    ];

    const checkString = (str, field) => {
      if (typeof str !== 'string') return;
      
      for (const pattern of xssPatterns) {
        if (pattern.test(str)) {
          throw new ValidationError(
            'Potential XSS attempt detected',
            {
              field,
              pattern: pattern.toString(),
              suspiciousContent: str.substring(0, 100)
            }
          );
        }
      }
    };

    // Check query parameters
    Object.entries(req.query || {}).forEach(([key, value]) => {
      checkString(value, `query.${key}`);
    });

    // Check body parameters (if JSON)
    if (req.body && typeof req.body === 'object') {
      this.checkObjectForXss(req.body, 'body');
    }
  }

  /**
   * Recursively check object for XSS
   */
  checkObjectForXss(obj, prefix = '') {
    Object.entries(obj).forEach(([key, value]) => {
      const fieldPath = prefix ? `${prefix}.${key}` : key;
      
      if (typeof value === 'string') {
        this.checkXssAttempts({ query: { [key]: value } });
      } else if (typeof value === 'object' && value !== null) {
        this.checkObjectForXss(value, fieldPath);
      }
    });
  }

  /**
   * Validate request patterns
   */
  validateRequestPatterns(req, config) {
    // Validate file upload patterns
    if (req.files || (req.body && req.body.files)) {
      this.validateFileUploads(req, config);
    }

    // Validate API versioning
    this.validateApiVersion(req);

    // Validate tenant context
    this.validateTenantContext(req);
  }

  /**
   * Validate file uploads
   */
  validateFileUploads(req, config) {
    const files = req.files || req.body.files || [];
    const maxFileSize = config.maxFileSize || 50 * 1024 * 1024; // 50MB
    const allowedExtensions = config.allowedFileExtensions || [
      '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.txt'
    ];

    if (Array.isArray(files)) {
      files.forEach((file, index) => {
        if (file.size > maxFileSize) {
          throw new ValidationError(
            `File ${index + 1} size exceeds maximum allowed size`,
            {
              fileIndex: index,
              currentSize: file.size,
              maxSize: maxFileSize,
              field: 'files'
            }
          );
        }

        const extension = file.originalname?.toLowerCase().match(/\.[^.]+$/)?.[0];
        if (extension && !allowedExtensions.includes(extension)) {
          throw new ValidationError(
            `File ${index + 1} type not allowed`,
            {
              fileIndex: index,
              extension,
              allowedExtensions,
              field: 'files'
            }
          );
        }
      });
    }
  }

  /**
   * Validate API version
   */
  validateApiVersion(req) {
    const apiVersion = req.headers['api-version'] || req.query.version;
    
    if (apiVersion) {
      const validVersions = ['v1', 'v2', '1.0', '2.0'];
      
      if (!validVersions.includes(apiVersion)) {
        throw new ValidationError(
          'Invalid API version',
          {
            version: apiVersion,
            supportedVersions: validVersions,
            field: 'api-version'
          }
        );
      }
    }
  }

  /**
   * Validate tenant context
   */
  validateTenantContext(req) {
    // Skip validation for public endpoints
    if (req.path.startsWith('/public') || req.path.startsWith('/health')) {
      return;
    }

    const tenantId = req.headers['x-tenant-id'] || req.user?.tenantId;
    
    if (!tenantId && req.method !== 'OPTIONS') {
      throw new ValidationError(
        'Tenant context is required',
        {
          field: 'x-tenant-id',
          message: 'X-Tenant-ID header or authenticated user context required'
        }
      );
    }
  }
}

// Export middleware factory
export function createRequestValidationMiddleware(options = {}) {
  const validationService = new RequestValidationService(options);
  return validationService.createMiddleware(options);
}

// Export service class
export { RequestValidationService };

// Export specific validation configurations
export const validationConfigs = {
  // Strict validation for API endpoints
  api: {
    maxRequestSize: 1024 * 1024, // 1MB
    enableSqlInjectionCheck: true,
    enableXssCheck: true
  },
  
  // Lenient validation for media uploads
  media: {
    maxRequestSize: 100 * 1024 * 1024, // 100MB
    maxFileSize: 50 * 1024 * 1024, // 50MB per file
    allowedFileExtensions: ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.pdf', '.doc', '.docx'],
    enableSqlInjectionCheck: false,
    enableXssCheck: false
  },
  
  // Very strict validation for admin endpoints
  admin: {
    maxRequestSize: 512 * 1024, // 512KB
    enableSqlInjectionCheck: true,
    enableXssCheck: true,
    maxUrlLength: 1024
  },
  
  // Basic validation for public endpoints
  public: {
    maxRequestSize: 10 * 1024, // 10KB
    enableSqlInjectionCheck: true,
    enableXssCheck: true,
    allowedContentTypes: ['application/json', 'text/plain']
  }
};