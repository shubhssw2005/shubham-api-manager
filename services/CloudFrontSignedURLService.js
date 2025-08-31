/**
 * CloudFront Signed URL Service
 * Provides secure access to media files through CloudFront signed URLs
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class CloudFrontSignedURLService {
  constructor(options = {}) {
    this.distributionDomain = options.distributionDomain || process.env.CLOUDFRONT_DOMAIN;
    this.keyPairId = options.keyPairId || process.env.CLOUDFRONT_KEY_PAIR_ID;
    this.privateKeyPath = options.privateKeyPath || process.env.CLOUDFRONT_PRIVATE_KEY_PATH;
    this.defaultExpiration = options.defaultExpiration || 3600; // 1 hour
    
    // Load private key
    this.privateKey = this.loadPrivateKey();
    
    if (!this.distributionDomain || !this.keyPairId || !this.privateKey) {
      throw new Error('CloudFront configuration incomplete. Missing domain, key pair ID, or private key.');
    }
  }

  /**
   * Load private key from file or environment variable
   */
  loadPrivateKey() {
    try {
      if (this.privateKeyPath && fs.existsSync(this.privateKeyPath)) {
        return fs.readFileSync(this.privateKeyPath, 'utf8');
      }
      
      if (process.env.CLOUDFRONT_PRIVATE_KEY) {
        return process.env.CLOUDFRONT_PRIVATE_KEY.replace(/\\n/g, '\n');
      }
      
      throw new Error('Private key not found');
    } catch (error) {
      console.error('Error loading CloudFront private key:', error);
      throw error;
    }
  }

  /**
   * Generate a signed URL for a specific resource
   * @param {string} resourcePath - Path to the resource (e.g., '/media/image.jpg')
   * @param {Object} options - Configuration options
   * @returns {string} Signed URL
   */
  generateSignedURL(resourcePath, options = {}) {
    const {
      expiresIn = this.defaultExpiration,
      ipAddress = null,
      dateGreaterThan = null,
      policy = null
    } = options;

    const url = `https://${this.distributionDomain}${resourcePath}`;
    const expiration = Math.floor(Date.now() / 1000) + expiresIn;

    if (policy) {
      return this.generateSignedURLWithCustomPolicy(url, policy);
    }

    return this.generateSignedURLWithCannedPolicy(url, expiration, ipAddress, dateGreaterThan);
  }

  /**
   * Generate signed URL with canned policy (simpler, most common use case)
   */
  generateSignedURLWithCannedPolicy(url, expiration, ipAddress = null, dateGreaterThan = null) {
    const policy = this.createCannedPolicy(url, expiration, ipAddress, dateGreaterThan);
    const signature = this.signPolicy(policy);
    
    const queryParams = new URLSearchParams({
      'Expires': expiration.toString(),
      'Signature': signature,
      'Key-Pair-Id': this.keyPairId
    });

    return `${url}?${queryParams.toString()}`;
  }

  /**
   * Generate signed URL with custom policy (more flexible)
   */
  generateSignedURLWithCustomPolicy(url, customPolicy) {
    const encodedPolicy = this.base64UrlEncode(JSON.stringify(customPolicy));
    const signature = this.signPolicy(JSON.stringify(customPolicy));
    
    const queryParams = new URLSearchParams({
      'Policy': encodedPolicy,
      'Signature': signature,
      'Key-Pair-Id': this.keyPairId
    });

    return `${url}?${queryParams.toString()}`;
  }

  /**
   * Create a canned policy for CloudFront signed URLs
   */
  createCannedPolicy(url, expiration, ipAddress = null, dateGreaterThan = null) {
    const condition = {
      DateLessThan: {
        'AWS:EpochTime': expiration
      }
    };

    if (ipAddress) {
      condition.IpAddress = {
        'AWS:SourceIp': ipAddress
      };
    }

    if (dateGreaterThan) {
      condition.DateGreaterThan = {
        'AWS:EpochTime': dateGreaterThan
      };
    }

    return {
      Statement: [{
        Resource: url,
        Condition: condition
      }]
    };
  }

  /**
   * Create a custom policy for more complex access control
   */
  createCustomPolicy(resources, options = {}) {
    const {
      expiration,
      ipAddress,
      dateGreaterThan,
      userAgent,
      referer
    } = options;

    const condition = {};

    if (expiration) {
      condition.DateLessThan = {
        'AWS:EpochTime': expiration
      };
    }

    if (dateGreaterThan) {
      condition.DateGreaterThan = {
        'AWS:EpochTime': dateGreaterThan
      };
    }

    if (ipAddress) {
      condition.IpAddress = {
        'AWS:SourceIp': ipAddress
      };
    }

    if (userAgent) {
      condition.StringLike = {
        'AWS:UserAgent': userAgent
      };
    }

    if (referer) {
      condition.StringLike = {
        ...condition.StringLike,
        'AWS:Referer': referer
      };
    }

    return {
      Statement: resources.map(resource => ({
        Resource: resource,
        Condition: condition
      }))
    };
  }

  /**
   * Sign the policy using RSA-SHA1
   */
  signPolicy(policy) {
    const sign = crypto.createSign('RSA-SHA1');
    sign.update(policy);
    const signature = sign.sign(this.privateKey, 'base64');
    return this.base64UrlEncode(signature);
  }

  /**
   * Base64 URL encode (CloudFront safe)
   */
  base64UrlEncode(str) {
    return str
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
  }

  /**
   * Generate signed URLs for multiple resources
   */
  generateBulkSignedURLs(resources, options = {}) {
    return resources.map(resource => ({
      resource,
      signedUrl: this.generateSignedURL(resource, options)
    }));
  }

  /**
   * Generate signed URL for media with tenant isolation
   */
  generateTenantMediaURL(tenantId, mediaPath, options = {}) {
    const resourcePath = `/tenants/${tenantId}/media/${mediaPath}`;
    return this.generateSignedURL(resourcePath, options);
  }

  /**
   * Generate signed URL with IP restriction
   */
  generateIPRestrictedURL(resourcePath, clientIP, options = {}) {
    return this.generateSignedURL(resourcePath, {
      ...options,
      ipAddress: clientIP
    });
  }

  /**
   * Generate time-window restricted URL
   */
  generateTimeWindowURL(resourcePath, startTime, endTime, options = {}) {
    const expiration = Math.floor(endTime / 1000);
    const dateGreaterThan = Math.floor(startTime / 1000);
    
    return this.generateSignedURL(resourcePath, {
      ...options,
      expiresIn: expiration - Math.floor(Date.now() / 1000),
      dateGreaterThan
    });
  }

  /**
   * Validate if a signed URL is still valid
   */
  isURLValid(signedUrl) {
    try {
      const url = new URL(signedUrl);
      const expires = url.searchParams.get('Expires');
      
      if (!expires) {
        return false;
      }
      
      const expirationTime = parseInt(expires) * 1000;
      return Date.now() < expirationTime;
    } catch (error) {
      return false;
    }
  }

  /**
   * Extract expiration time from signed URL
   */
  getURLExpiration(signedUrl) {
    try {
      const url = new URL(signedUrl);
      const expires = url.searchParams.get('Expires');
      return expires ? new Date(parseInt(expires) * 1000) : null;
    } catch (error) {
      return null;
    }
  }
}

export default CloudFrontSignedURLService;