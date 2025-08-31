import { jwtAuth, optionalJwtAuth, tenantIsolation, tenantRateLimit } from './jwtAuth.js';
import PermissionService from '../lib/auth/PermissionService.js';

const permissionService = new PermissionService();

/**
 * Enhanced authentication middleware that combines JWT auth with permission checking
 */
export function requireAuth(options = {}) {
  const { 
    permissions = [], 
    checkType = 'any', 
    requireTenant = false,
    rateLimit = false 
  } = options;

  return async (req, res, next) => {
    try {
      // Apply JWT authentication
      await new Promise((resolve, reject) => {
        jwtAuth(req, res, (error) => {
          if (error) reject(error);
          else resolve();
        });
      });

      // Apply tenant isolation if required
      if (requireTenant) {
        await new Promise((resolve, reject) => {
          tenantIsolation(req, res, (error) => {
            if (error) reject(error);
            else resolve();
          });
        });
      }

      // Apply rate limiting if enabled
      if (rateLimit) {
        const rateLimitOptions = typeof rateLimit === 'object' ? rateLimit : {};
        await new Promise((resolve, reject) => {
          tenantRateLimit(rateLimitOptions)(req, res, (error) => {
            if (error) reject(error);
            else resolve();
          });
        });
      }

      // Check permissions if specified
      if (permissions.length > 0) {
        let hasPermission = false;

        if (checkType === 'all') {
          hasPermission = await permissionService.hasAllPermissions(req.user, permissions);
        } else if (checkType === 'any') {
          hasPermission = await permissionService.hasAnyPermission(req.user, permissions);
        }

        if (!hasPermission) {
          return res.status(403).json({
            error: 'Insufficient permissions',
            code: 'INSUFFICIENT_PERMISSIONS',
            requiredPermissions: permissions
          });
        }
      }

      next();
    } catch (error) {
      console.error('Enhanced auth error:', error);
      
      // Handle specific error types
      if (error.message?.includes('token')) {
        return res.status(401).json({
          error: error.message,
          code: 'AUTH_ERROR'
        });
      }

      return res.status(500).json({
        error: 'Authentication error',
        code: 'AUTH_ERROR'
      });
    }
  };
}

/**
 * Role-based authentication middleware
 */
export function requireRole(roles, options = {}) {
  if (typeof roles === 'string') {
    roles = [roles];
  }

  return requireAuth({
    ...options,
    permissions: roles.map(role => `role:${role}`)
  });
}

/**
 * Admin authentication middleware
 */
export function requireAdmin(options = {}) {
  return async (req, res, next) => {
    try {
      // Apply JWT authentication first
      await new Promise((resolve, reject) => {
        jwtAuth(req, res, (error) => {
          if (error) reject(error);
          else resolve();
        });
      });

      // Check if user is admin or superadmin
      if (req.user.role !== 'admin' && req.user.role !== 'superadmin') {
        return res.status(403).json({
          error: 'Admin access required',
          code: 'ADMIN_REQUIRED'
        });
      }

      // Apply additional options if provided
      if (options.requireTenant) {
        await new Promise((resolve, reject) => {
          tenantIsolation(req, res, (error) => {
            if (error) reject(error);
            else resolve();
          });
        });
      }

      next();
    } catch (error) {
      console.error('Admin auth error:', error);
      return res.status(500).json({
        error: 'Authentication error',
        code: 'AUTH_ERROR'
      });
    }
  };
}

/**
 * Superadmin authentication middleware
 */
export function requireSuperAdmin(options = {}) {
  return async (req, res, next) => {
    try {
      // Apply JWT authentication first
      await new Promise((resolve, reject) => {
        jwtAuth(req, res, (error) => {
          if (error) reject(error);
          else resolve();
        });
      });

      // Check if user is superadmin
      if (req.user.role !== 'superadmin') {
        return res.status(403).json({
          error: 'Superadmin access required',
          code: 'SUPERADMIN_REQUIRED'
        });
      }

      next();
    } catch (error) {
      console.error('Superadmin auth error:', error);
      return res.status(500).json({
        error: 'Authentication error',
        code: 'AUTH_ERROR'
      });
    }
  };
}

/**
 * Optional authentication middleware
 */
export function optionalAuth(options = {}) {
  return async (req, res, next) => {
    try {
      // Apply optional JWT authentication
      await new Promise((resolve, reject) => {
        optionalJwtAuth(req, res, (error) => {
          if (error) reject(error);
          else resolve();
        });
      });

      // Apply tenant isolation if user is authenticated and tenant is required
      if (req.user && options.requireTenant) {
        await new Promise((resolve, reject) => {
          tenantIsolation(req, res, (error) => {
            if (error) reject(error);
            else resolve();
          });
        });
      }

      next();
    } catch (error) {
      console.error('Optional auth error:', error);
      // Don't fail on optional auth errors
      req.user = null;
      req.tenant = null;
      next();
    }
  };
}

/**
 * Resource ownership middleware - ensures user can only access their own resources
 */
export function requireOwnership(options = {}) {
  const { resourceIdParam = 'id', userIdField = 'userId' } = options;

  return async (req, res, next) => {
    try {
      // Must be authenticated first
      if (!req.user) {
        return res.status(401).json({
          error: 'Authentication required',
          code: 'AUTH_REQUIRED'
        });
      }

      // Superadmin can access any resource
      if (req.user.role === 'superadmin') {
        return next();
      }

      // Get resource ID from request
      const resourceId = req.params[resourceIdParam] || req.body[resourceIdParam] || req.query[resourceIdParam];
      
      if (!resourceId) {
        return res.status(400).json({
          error: 'Resource ID required',
          code: 'MISSING_RESOURCE_ID'
        });
      }

      // For now, we'll add the ownership check to the request context
      // The actual resource ownership check should be done in the route handler
      req.ownershipCheck = {
        resourceId,
        userIdField,
        userId: req.user.id
      };

      next();
    } catch (error) {
      console.error('Ownership check error:', error);
      return res.status(500).json({
        error: 'Ownership check error',
        code: 'OWNERSHIP_ERROR'
      });
    }
  };
}

// Export permission service for direct use
export { permissionService };