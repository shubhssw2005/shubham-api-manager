/**
 * CloudFront Cache Management API
 * Handles cache invalidation and optimization
 */

import { getServerSession } from 'next-auth/next';
import { authOptions } from '../auth/[...nextauth].js';
import CloudFrontCacheService from '../../../services/CloudFrontCacheService.js';
import { APIError, ValidationError, AuthorizationError } from '../../../lib/errors/index.js';

const cacheService = new CloudFrontCacheService();

export default async function handler(req, res) {
  try {
    const session = await getServerSession(req, res, authOptions);
    if (!session?.user) {
      throw new AuthorizationError('Authentication required');
    }

    // Check if user has cache management permissions
    if (!session.user.permissions?.includes('cache:manage')) {
      throw new AuthorizationError('Insufficient permissions for cache management');
    }

    switch (req.method) {
      case 'POST':
        return await handleInvalidation(req, res, session);
      case 'GET':
        return await handleCacheStatus(req, res, session);
      case 'DELETE':
        return await handleCacheCleanup(req, res, session);
      default:
        return res.status(405).json({ error: 'Method not allowed' });
    }

  } catch (error) {
    console.error('Cache management error:', error);
    
    if (error instanceof APIError) {
      return res.status(error.statusCode).json({ error: error.message });
    }
    
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Handle cache invalidation requests
 */
async function handleInvalidation(req, res, session) {
  const { 
    action, 
    paths, 
    mediaIds, 
    contentType, 
    immediate = false,
    tenantId 
  } = req.body;

  const targetTenantId = tenantId || session.user.tenantId;

  // Validate tenant access
  if (tenantId && tenantId !== session.user.tenantId && !session.user.permissions?.includes('admin:all')) {
    throw new AuthorizationError('Cannot manage cache for other tenants');
  }

  switch (action) {
    case 'invalidate_paths':
      if (!paths || !Array.isArray(paths)) {
        throw new ValidationError('paths array is required');
      }
      
      const result = await cacheService.invalidatePaths(paths, { immediate });
      return res.status(200).json({
        success: true,
        invalidation: result,
        message: `${paths.length} paths queued for invalidation`
      });

    case 'invalidate_tenant_media':
      const mediaResult = await cacheService.invalidateTenantMedia(targetTenantId, mediaIds);
      return res.status(200).json({
        success: true,
        invalidation: mediaResult,
        message: `Media cache invalidated for tenant ${targetTenantId}`
      });

    case 'invalidate_tenant_api':
      const { endpoints } = req.body;
      const apiResult = await cacheService.invalidateTenantAPI(targetTenantId, endpoints);
      return res.status(200).json({
        success: true,
        invalidation: apiResult,
        message: `API cache invalidated for tenant ${targetTenantId}`
      });

    case 'smart_invalidation':
      if (!contentType) {
        throw new ValidationError('contentType is required for smart invalidation');
      }
      
      const { resourceId, tags } = req.body;
      const smartResult = await cacheService.smartInvalidation(contentType, {
        tenantId: targetTenantId,
        resourceId,
        tags
      });
      
      return res.status(200).json({
        success: true,
        invalidation: smartResult,
        message: `Smart invalidation completed for ${contentType}`
      });

    default:
      throw new ValidationError('Invalid action. Supported actions: invalidate_paths, invalidate_tenant_media, invalidate_tenant_api, smart_invalidation');
  }
}

/**
 * Handle cache status requests
 */
async function handleCacheStatus(req, res, session) {
  const { action, invalidationId } = req.query;

  switch (action) {
    case 'invalidation_status':
      if (!invalidationId) {
        throw new ValidationError('invalidationId is required');
      }
      
      const status = await cacheService.getInvalidationStatus(invalidationId);
      return res.status(200).json({ status });

    case 'list_invalidations':
      const maxItems = parseInt(req.query.maxItems) || 50;
      const invalidations = await cacheService.listInvalidations(maxItems);
      return res.status(200).json({ invalidations });

    case 'cache_statistics':
      const hours = parseInt(req.query.hours) || 24;
      const stats = await cacheService.getCacheStatistics(hours);
      return res.status(200).json({ statistics: stats });

    case 'optimization_recommendations':
      const recommendations = await cacheService.optimizeCacheSettings();
      return res.status(200).json({ recommendations });

    default:
      throw new ValidationError('Invalid action. Supported actions: invalidation_status, list_invalidations, cache_statistics, optimization_recommendations');
  }
}

/**
 * Handle cache cleanup requests
 */
async function handleCacheCleanup(req, res, session) {
  // Only admins can perform cleanup
  if (!session.user.permissions?.includes('admin:all')) {
    throw new AuthorizationError('Admin permissions required for cache cleanup');
  }

  await cacheService.cleanup();
  
  return res.status(200).json({
    success: true,
    message: 'Cache cleanup completed'
  });
}

/**
 * Batch invalidation endpoint
 */
export async function batchInvalidation(req, res) {
  try {
    const session = await getServerSession(req, res, authOptions);
    if (!session?.user?.permissions?.includes('cache:manage')) {
      throw new AuthorizationError('Insufficient permissions');
    }

    const { operations } = req.body;
    
    if (!operations || !Array.isArray(operations)) {
      throw new ValidationError('operations array is required');
    }

    if (operations.length > 10) {
      throw new ValidationError('Maximum 10 operations per batch');
    }

    const results = [];
    
    for (const operation of operations) {
      try {
        let result;
        
        switch (operation.type) {
          case 'paths':
            result = await cacheService.invalidatePaths(operation.paths, operation.options);
            break;
          case 'tenant_media':
            result = await cacheService.invalidateTenantMedia(operation.tenantId, operation.mediaIds);
            break;
          case 'tenant_api':
            result = await cacheService.invalidateTenantAPI(operation.tenantId, operation.endpoints);
            break;
          default:
            throw new Error(`Unknown operation type: ${operation.type}`);
        }
        
        results.push({
          operation: operation.type,
          success: true,
          result
        });
      } catch (error) {
        results.push({
          operation: operation.type,
          success: false,
          error: error.message
        });
      }
    }

    return res.status(200).json({
      success: true,
      results,
      summary: {
        total: operations.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      }
    });

  } catch (error) {
    console.error('Batch invalidation error:', error);
    
    if (error instanceof APIError) {
      return res.status(error.statusCode).json({ error: error.message });
    }
    
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Cache warming endpoint
 */
export async function warmCache(req, res) {
  try {
    const session = await getServerSession(req, res, authOptions);
    if (!session?.user?.permissions?.includes('cache:manage')) {
      throw new AuthorizationError('Insufficient permissions');
    }

    const { urls, priority = 'normal' } = req.body;
    
    if (!urls || !Array.isArray(urls)) {
      throw new ValidationError('urls array is required');
    }

    if (urls.length > 100) {
      throw new ValidationError('Maximum 100 URLs can be warmed at once');
    }

    // Simulate cache warming by making requests to the URLs
    const results = [];
    
    for (const url of urls) {
      try {
        const response = await fetch(url, {
          method: 'HEAD',
          headers: {
            'User-Agent': 'CloudFront-Cache-Warmer/1.0'
          }
        });
        
        results.push({
          url,
          success: response.ok,
          status: response.status,
          cacheStatus: response.headers.get('x-cache') || 'unknown'
        });
      } catch (error) {
        results.push({
          url,
          success: false,
          error: error.message
        });
      }
    }

    return res.status(200).json({
      success: true,
      results,
      summary: {
        total: urls.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      }
    });

  } catch (error) {
    console.error('Cache warming error:', error);
    
    if (error instanceof APIError) {
      return res.status(error.statusCode).json({ error: error.message });
    }
    
    return res.status(500).json({ error: 'Internal server error' });
  }
}