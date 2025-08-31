/**
 * CloudFront CDN Usage Examples
 * Demonstrates how to use CloudFront signed URLs and cache management
 */

import CloudFrontSignedURLService from '../services/CloudFrontSignedURLService.js';
import CloudFrontCacheService from '../services/CloudFrontCacheService.js';
import { createCloudFrontMiddleware, addCloudFrontHelpers } from '../middleware/cloudfront.js';
import express from 'express';

// Initialize services
const cloudfrontService = new CloudFrontSignedURLService();
const cacheService = new CloudFrontCacheService();

/**
 * Example 1: Basic Signed URL Generation
 */
async function basicSignedUrlExample() {
  console.log('=== Basic Signed URL Generation ===');
  
  try {
    // Generate a simple signed URL
    const s3Key = 'tenants/demo-tenant/images/sample-image.jpg';
    const signedUrl = cloudfrontService.generateMediaSignedURL(s3Key, {
      expiresIn: 3600 // 1 hour
    });
    
    console.log('Generated signed URL:', signedUrl);
    console.log('URL is valid:', cloudfrontService.validateSignedURL(signedUrl));
    
    // Generate signed URL with IP restriction
    const restrictedUrl = cloudfrontService.generateSignedURL(
      `https://${process.env.CLOUDFRONT_DOMAIN_NAME}/${s3Key}`,
      {
        expiresIn: 1800, // 30 minutes
        ipAddress: '192.168.1.0/24'
      }
    );
    
    console.log('IP-restricted signed URL:', restrictedUrl);
    
  } catch (error) {
    console.error('Error generating signed URLs:', error);
  }
}

/**
 * Example 2: Bulk Signed URL Generation
 */
async function bulkSignedUrlExample() {
  console.log('\n=== Bulk Signed URL Generation ===');
  
  try {
    const mediaFiles = [
      'tenants/demo-tenant/images/photo1.jpg',
      'tenants/demo-tenant/images/photo2.jpg',
      'tenants/demo-tenant/videos/video1.mp4',
      'tenants/demo-tenant/documents/document1.pdf'
    ];
    
    const signedUrls = cloudfrontService.generateBulkMediaSignedURLs(mediaFiles, {
      expiresIn: 7200 // 2 hours
    });
    
    console.log('Generated bulk signed URLs:');
    signedUrls.forEach((item, index) => {
      console.log(`${index + 1}. ${item.key} -> ${item.signedUrl.substring(0, 100)}...`);
    });
    
  } catch (error) {
    console.error('Error generating bulk signed URLs:', error);
  }
}

/**
 * Example 3: Cache Management
 */
async function cacheManagementExample() {
  console.log('\n=== Cache Management ===');
  
  try {
    // Get cache statistics
    const stats = await cacheService.getCacheStats();
    console.log('Cache statistics:', stats);
    
    // Smart invalidation with batching
    const pathsToInvalidate = [
      '/tenants/demo-tenant/images/updated-image.jpg',
      '/tenants/demo-tenant/thumbnails/updated-image-thumb.jpg'
    ];
    
    const invalidationResults = await cacheService.smartInvalidate(pathsToInvalidate, {
      priority: 'normal',
      skipDuplicates: true
    });
    
    console.log('Invalidation results:', invalidationResults);
    
    // Invalidate by content type
    const contentTypeResults = await cacheService.invalidateByContentType('images', 'demo-tenant');
    console.log('Content type invalidation:', contentTypeResults);
    
    // Schedule cache warming
    await cacheService.scheduleWarmup(['/popular-content/trending-video.mp4'], 30000);
    console.log('Cache warming scheduled');
    
  } catch (error) {
    console.error('Error with cache management:', error);
  }
}

/**
 * Example 4: Express.js Integration
 */
function expressIntegrationExample() {
  console.log('\n=== Express.js Integration ===');
  
  const app = express();
  
  // Add CloudFront middleware
  app.use(createCloudFrontMiddleware({
    enabled: true,
    defaultExpiration: 3600,
    autoConvert: true,
    urlType: 'canned'
  }));
  
  // Add CloudFront helpers to request object
  app.use(addCloudFrontHelpers({
    defaultExpiration: 3600
  }));
  
  // Example route that automatically converts S3 URLs
  app.get('/api/media/:id', async (req, res) => {
    try {
      // Simulate fetching media data from database
      const mediaData = {
        id: req.params.id,
        title: 'Sample Media',
        url: `https://my-bucket.s3.us-east-1.amazonaws.com/tenants/demo/media/${req.params.id}.jpg`,
        thumbnail: `https://my-bucket.s3.us-east-1.amazonaws.com/tenants/demo/thumbnails/${req.params.id}-thumb.jpg`,
        metadata: {
          size: 1024000,
          type: 'image/jpeg'
        }
      };
      
      // URLs will be automatically converted to signed CloudFront URLs
      res.json({
        success: true,
        data: mediaData
      });
      
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  // Example route using CloudFront helpers
  app.get('/api/media/:id/signed-url', async (req, res) => {
    try {
      const s3Key = `tenants/demo/media/${req.params.id}.jpg`;
      
      // Use CloudFront helper from request object
      const signedUrl = req.cloudfront.generateSignedUrl(s3Key, {
        expiresIn: 1800, // 30 minutes
        ipRestriction: true // Will use client IP
      });
      
      res.json({
        success: true,
        data: {
          signedUrl,
          expiresIn: 1800
        }
      });
      
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  // Example route for bulk signed URLs
  app.post('/api/media/bulk-signed-urls', async (req, res) => {
    try {
      const { s3Keys, expiresIn = 3600 } = req.body;
      
      if (!s3Keys || !Array.isArray(s3Keys)) {
        return res.status(400).json({ error: 's3Keys array is required' });
      }
      
      const signedUrls = req.cloudfront.generateBulkSignedUrls(s3Keys, {
        expiresIn
      });
      
      res.json({
        success: true,
        data: {
          urls: signedUrls,
          count: signedUrls.length
        }
      });
      
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  console.log('Express app configured with CloudFront middleware');
  return app;
}

/**
 * Example 5: Advanced Cache Strategies
 */
async function advancedCacheStrategiesExample() {
  console.log('\n=== Advanced Cache Strategies ===');
  
  try {
    // Tenant-specific cache invalidation
    await cacheService.invalidateTenantContent('demo-tenant', ['images', 'videos']);
    console.log('Tenant cache invalidated');
    
    // Emergency cache purge for critical content
    const criticalPaths = ['/api/critical-data.json', '/system/config.js'];
    await cacheService.emergencyPurge(criticalPaths);
    console.log('Emergency purge completed');
    
    // Pattern-based invalidation
    await cacheService.invalidateCacheByPattern(['/api/v1/users/*', '/thumbnails/large/*']);
    console.log('Pattern-based invalidation completed');
    
    // Get invalidation status
    const invalidationId = 'I1234567890ABCDEF'; // Example ID
    try {
      const status = await cacheService.getInvalidationStatus(invalidationId);
      console.log('Invalidation status:', status.Status);
    } catch (error) {
      console.log('Invalidation not found or error:', error.message);
    }
    
  } catch (error) {
    console.error('Error with advanced cache strategies:', error);
  }
}

/**
 * Example 6: Security Best Practices
 */
async function securityBestPracticesExample() {
  console.log('\n=== Security Best Practices ===');
  
  try {
    // Generate signed URL with time-based restrictions
    const now = Math.floor(Date.now() / 1000);
    const oneHourFromNow = now + 3600;
    const twoHoursFromNow = now + 7200;
    
    const timeRestrictedUrl = cloudfrontService.generateSignedURL(
      'https://cdn.example.com/sensitive-content.pdf',
      {
        expiresIn: 3600,
        dateGreaterThan: oneHourFromNow // Can't access until 1 hour from now
      }
    );
    
    console.log('Time-restricted URL generated');
    
    // Generate signed URL with IP and time restrictions
    const fullyRestrictedUrl = cloudfrontService.generateSignedURL(
      'https://cdn.example.com/private-video.mp4',
      {
        expiresIn: 1800, // 30 minutes
        ipAddress: '203.0.113.0/24', // Specific IP range
        dateGreaterThan: now + 300 // Can't access for 5 minutes
      }
    );
    
    console.log('Fully restricted URL generated');
    
    // Validate URL format (security check)
    const isValid = cloudfrontService.validateSignedURL(fullyRestrictedUrl);
    console.log('URL validation passed:', isValid);
    
    // Get distribution info (for monitoring)
    const distributionInfo = cloudfrontService.getDistributionInfo();
    console.log('Distribution info:', distributionInfo);
    
  } catch (error) {
    console.error('Error with security examples:', error);
  }
}

/**
 * Example 7: Error Handling and Monitoring
 */
async function errorHandlingExample() {
  console.log('\n=== Error Handling and Monitoring ===');
  
  try {
    // Handle missing configuration gracefully
    if (!process.env.CLOUDFRONT_DISTRIBUTION_ID) {
      console.log('CloudFront not configured - falling back to direct S3 URLs');
      return;
    }
    
    // Monitor cache statistics
    const stats = await cacheService.getCacheStats();
    if (stats.invalidationsRemaining < 10) {
      console.warn('Warning: Low invalidation quota remaining:', stats.invalidationsRemaining);
    }
    
    // Test signed URL generation with error handling
    try {
      const testUrl = cloudfrontService.generateMediaSignedURL('test/invalid-key.jpg');
      console.log('Test URL generated successfully');
    } catch (error) {
      console.error('Failed to generate test URL:', error.message);
    }
    
    // Test cache invalidation with error handling
    try {
      await cacheService.smartInvalidate(['/test/path.jpg'], {
        priority: 'low'
      });
      console.log('Test invalidation successful');
    } catch (error) {
      console.error('Failed to invalidate cache:', error.message);
    }
    
  } catch (error) {
    console.error('Error in error handling example:', error);
  }
}

/**
 * Run all examples
 */
async function runAllExamples() {
  console.log('CloudFront CDN Usage Examples');
  console.log('============================');
  
  await basicSignedUrlExample();
  await bulkSignedUrlExample();
  await cacheManagementExample();
  expressIntegrationExample();
  await advancedCacheStrategiesExample();
  await securityBestPracticesExample();
  await errorHandlingExample();
  
  console.log('\n=== All Examples Completed ===');
}

// Export examples for individual use
export {
  basicSignedUrlExample,
  bulkSignedUrlExample,
  cacheManagementExample,
  expressIntegrationExample,
  advancedCacheStrategiesExample,
  securityBestPracticesExample,
  errorHandlingExample,
  runAllExamples
};

// Run examples if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples().catch(console.error);
}