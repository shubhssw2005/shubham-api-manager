/**
 * CloudFront Integration Middleware
 * Handles CloudFront-specific headers, signed URLs, and cache control
 */

import CloudFrontSignedURLService from '../services/CloudFrontSignedURLService.js';
import CloudFrontCacheService from '../services/CloudFrontCacheService.js';

class CloudFrontMiddleware {
  constructor() {
    this.signedURLService = new CloudFrontSignedURLService();
    this.cacheService = new CloudFrontCacheService();
    this.trustedHeaders = [
      'cloudfront-viewer-country',
      'cloudfront-viewer-country-name',
      'cloudfront-viewer-country-region',
      'cloudfront-viewer-country-region-name',
      'cloudfront-is-desktop-viewer',
      'cloudfront-is-mobile-viewer',
      'cloudfront-is-smarttv-viewer',
      'cloudfront-is-tablet-viewer'
    ];
  }

  /**
   * Extract CloudFront viewer information
   */
  extractViewerInfo() {
    return (req, res, next) => {
      req.cloudfront = {
        viewerCountry: req.headers['cloudfront-viewer-country'],
        viewerCountryName: req.headers['cloudfront-viewer-country-name'],
        viewerRegion: req.headers['cloudfront-viewer-country-region'],
        viewerRegionName: req.headers['cloudfront-viewer-country-region-name'],
        isDesktop: req.headers['cloudfront-is-desktop-viewer'] === 'true',
        isMobile: req.headers['cloudfront-is-mobile-viewer'] === 'true',
        isSmartTV: req.headers['cloudfront-is-smarttv-viewer'] === 'true',
        isTablet: req.headers['cloudfront-is-tablet-viewer'] === 'true',
        forwardedFor: req.headers['x-forwarded-for'],
        realIP: req.headers['x-real-ip'] || req.connection.remoteAddress
      };

      next();
    };
  }

  /**
   * Validate CloudFront headers to prevent spoofing
   */
  validateCloudFrontHeaders() {
    return (req, res, next) => {
      // Check if request is coming through CloudFront
      const cfHeaders = Object.keys(req.headers).filter(header => 
        header.startsWith('cloudfront-')
      );

      if (cfHeaders.length === 0 && process.env.NODE_ENV === 'production') {
        // In production, all requests should come through CloudFront
        return res.status(403).json({
          error: 'Direct access not allowed'
        });
      }

      // Validate trusted headers
      for (const header of this.trustedHeaders) {
        if (req.headers[header] && !this.isValidHeaderValue(header, req.headers[header])) {
          console.warn(`Invalid CloudFront header value: ${header}=${req.headers[header]}`);
          delete req.headers[header];
        }
      }

      next();
    };
  }

  /**
   * Set appropriate cache headers based on content type
   */
  setCacheHeaders() {
    return (req, res, next) => {
      // Store original res.json to modify cache headers
      const originalJson = res.json;
      const originalSend = res.send;

      res.json = function(data) {
        setCacheHeadersForResponse(this, req, 'application/json');
        return originalJson.call(this, data);
      };

      res.send = function(data) {
        setCacheHeadersForResponse(this, req, 'text/html');
        return originalSend.call(this, data);
      };

      next();
    };

    function setCacheHeadersForResponse(res, req, contentType) {
      const path = req.path;
      const method = req.method;

      // Don't cache POST, PUT, DELETE requests
      if (method !== 'GET' && method !== 'HEAD') {
        res.set({
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        });
        return;
      }

      // API endpoints - short cache
      if (path.startsWith('/api/')) {
        res.set({
          'Cache-Control': 'public, max-age=300, s-maxage=300', // 5 minutes
          'Vary': 'Authorization, X-Tenant-ID'
        });
        return;
      }

      // Media files - long cache
      if (path.includes('/media/') || path.match(/\.(jpg|jpeg|png|gif|webp|mp4|pdf)$/i)) {
        res.set({
          'Cache-Control': 'public, max-age=31536000, s-maxage=31536000', // 1 year
          'Vary': 'Accept-Encoding'
        });
        return;
      }

      // Static assets - medium cache
      if (path.match(/\.(css|js|ico|svg)$/i)) {
        res.set({
          'Cache-Control': 'public, max-age=86400, s-maxage=86400', // 1 day
          'Vary': 'Accept-Encoding'
        });
        return;
      }

      // Default - short cache
      res.set({
        'Cache-Control': 'public, max-age=300, s-maxage=300', // 5 minutes
        'Vary': 'Accept-Encoding'
      });
    }
  }

  /**
   * Generate signed URLs for media responses
   */
  generateSignedURLs() {
    return async (req, res, next) => {
      const originalJson = res.json;

      res.json = function(data) {
        // Process media objects to add signed URLs
        if (data && typeof data === 'object') {
          processMediaObjects(data, req.user?.tenantId);
        }
        return originalJson.call(this, data);
      };

      next();
    };

    const processMediaObjects = (obj, tenantId) => {
      if (Array.isArray(obj)) {
        obj.forEach(item => processMediaObjects(item, tenantId));
        return;
      }

      if (obj && typeof obj === 'object') {
        // Check if this is a media object
        if (obj.s3Key && obj.mimeType && tenantId) {
          try {
            obj.signedUrl = this.signedURLService.generateTenantMediaURL(
              tenantId,
              obj.s3Key,
              { expiresIn: 3600 } // 1 hour
            );
          } catch (error) {
            console.error('Failed to generate signed URL:', error);
          }
        }

        // Process nested objects
        Object.values(obj).forEach(value => {
          if (typeof value === 'object') {
            processMediaObjects(value, tenantId);
          }
        });
      }
    };
  }

  /**
   * Handle cache invalidation after data modifications
   */
  handleCacheInvalidation() {
    return async (req, res, next) => {
      const originalJson = res.json;

      res.json = function(data) {
        // Trigger cache invalidation for successful modifications
        if (res.statusCode >= 200 && res.statusCode < 300) {
          triggerInvalidation(req, data);
        }
        return originalJson.call(this, data);
      };

      next();
    };

    const triggerInvalidation = async (req, data) => {
      const method = req.method;
      const path = req.path;
      const tenantId = req.user?.tenantId;

      if (!tenantId || method === 'GET') return;

      try {
        // Media invalidation
        if (path.includes('/media/')) {
          if (data && data.id) {
            await this.cacheService.invalidateTenantMedia(tenantId, [data.id]);
          } else {
            await this.cacheService.invalidateTenantMedia(tenantId);
          }
        }

        // API invalidation
        if (path.startsWith('/api/')) {
          const endpoints = [path.replace('/api/', '')];
          await this.cacheService.invalidateTenantAPI(tenantId, endpoints);
        }
      } catch (error) {
        console.error('Cache invalidation failed:', error);
        // Don't fail the request if invalidation fails
      }
    };
  }

  /**
   * Security headers for CloudFront
   */
  setSecurityHeaders() {
    return (req, res, next) => {
      // Security headers that work well with CloudFront
      res.set({
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
      });

      // HSTS header (CloudFront will add this too, but good to have)
      if (req.secure || req.headers['x-forwarded-proto'] === 'https') {
        res.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
      }

      next();
    };
  }

  /**
   * Rate limiting based on CloudFront viewer info
   */
  geoBasedRateLimit() {
    return (req, res, next) => {
      const country = req.cloudfront?.viewerCountry;
      const isHighRiskCountry = process.env.HIGH_RISK_COUNTRIES?.split(',').includes(country);

      if (isHighRiskCountry) {
        // Apply stricter rate limiting for high-risk countries
        req.rateLimitMultiplier = 0.5; // Half the normal rate limit
      }

      next();
    };
  }

  /**
   * Device-specific optimizations
   */
  deviceOptimization() {
    return (req, res, next) => {
      const cloudfront = req.cloudfront;

      if (cloudfront?.isMobile) {
        // Mobile optimizations
        res.set('Vary', 'CloudFront-Is-Mobile-Viewer');
        req.deviceType = 'mobile';
      } else if (cloudfront?.isTablet) {
        req.deviceType = 'tablet';
      } else if (cloudfront?.isSmartTV) {
        req.deviceType = 'tv';
      } else {
        req.deviceType = 'desktop';
      }

      next();
    };
  }

  /**
   * Validate header values
   */
  isValidHeaderValue(header, value) {
    switch (header) {
      case 'cloudfront-viewer-country':
        return /^[A-Z]{2}$/.test(value);
      case 'cloudfront-is-desktop-viewer':
      case 'cloudfront-is-mobile-viewer':
      case 'cloudfront-is-smarttv-viewer':
      case 'cloudfront-is-tablet-viewer':
        return value === 'true' || value === 'false';
      default:
        return true;
    }
  }

  /**
   * Create complete middleware stack
   */
  createMiddlewareStack() {
    return [
      this.validateCloudFrontHeaders(),
      this.extractViewerInfo(),
      this.setSecurityHeaders(),
      this.setCacheHeaders(),
      this.deviceOptimization(),
      this.geoBasedRateLimit(),
      this.generateSignedURLs(),
      this.handleCacheInvalidation()
    ];
  }
}

// Export singleton instance
const cloudFrontMiddleware = new CloudFrontMiddleware();

export default cloudFrontMiddleware;
export { CloudFrontMiddleware };