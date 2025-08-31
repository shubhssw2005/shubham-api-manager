/**
 * Example usage of the JWT Authentication and Authorization Service
 */

import { requireAuth, requireAdmin, requirePermission } from '../middleware/enhancedAuth.js';
import { tokenService, permissionService } from '../lib/auth/index.js';

// Example 1: Basic authentication middleware
export function basicAuthExample(req, res, next) {
  // Use the enhanced auth middleware with basic authentication
  return requireAuth()(req, res, next);
}

// Example 2: Role-based authentication
export function adminOnlyExample(req, res, next) {
  // Require admin role
  return requireAdmin()(req, res, next);
}

// Example 3: Permission-based authentication
export function contentManagementExample(req, res, next) {
  // Require specific permissions
  return requireAuth({
    permissions: ['content.create', 'content.update'],
    checkType: 'any', // User needs ANY of these permissions
    requireTenant: true, // Enforce tenant isolation
    rateLimit: { maxRequests: 100, windowMs: 60000 } // 100 requests per minute
  })(req, res, next);
}

// Example 4: Multiple permissions required
export function adminContentExample(req, res, next) {
  // Require ALL specified permissions
  return requireAuth({
    permissions: ['content.delete', 'users.manage'],
    checkType: 'all', // User needs ALL of these permissions
    requireTenant: false // No tenant isolation for admin operations
  })(req, res, next);
}

// Example 5: Manual permission checking in route handler
export async function manualPermissionCheck(req, res) {
  try {
    // Check if user has permission to perform action
    const hasPermission = await permissionService.hasPermission(
      req.user, 
      'media.delete',
      { tenantId: req.params.tenantId } // Resource context
    );

    if (!hasPermission) {
      return res.status(403).json({
        error: 'Insufficient permissions to delete media',
        code: 'PERMISSION_DENIED'
      });
    }

    // Proceed with the operation
    res.json({ message: 'Media deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Permission check failed' });
  }
}

// Example 6: Token management
export async function tokenManagementExample(req, res) {
  try {
    const userId = req.user.id;

    // Get user's active sessions
    const sessions = await tokenService.getUserSessions(userId);

    // Revoke a specific session
    if (req.body.revokeSessionId) {
      await tokenService.revokeRefreshToken(userId, req.body.revokeSessionId);
    }

    // Revoke all sessions (logout from all devices)
    if (req.body.revokeAll) {
      await tokenService.revokeAllUserTokens(userId);
    }

    res.json({
      sessions,
      message: 'Token management completed'
    });
  } catch (error) {
    res.status(500).json({ error: 'Token management failed' });
  }
}

// Example API route using the new authentication system
export default async function protectedApiRoute(req, res) {
  // Apply authentication and authorization
  await new Promise((resolve, reject) => {
    requireAuth({
      permissions: ['api.access'],
      requireTenant: true,
      rateLimit: true
    })(req, res, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });

  // Route logic here - user is authenticated and authorized
  res.json({
    message: 'Access granted',
    user: {
      id: req.user.id,
      email: req.user.email,
      role: req.user.role,
      permissions: req.user.permissions,
      tenantId: req.user.tenantId
    },
    tenant: req.tenant
  });
}

/**
 * Example of how to use in Next.js API routes
 */

// pages/api/protected-endpoint.js
/*
import { requireAuth } from '../../middleware/enhancedAuth.js';

export default async function handler(req, res) {
  // Apply authentication middleware
  await new Promise((resolve, reject) => {
    requireAuth({
      permissions: ['content.read'],
      requireTenant: true
    })(req, res, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });

  // Your protected route logic here
  if (req.method === 'GET') {
    // Handle GET request
    res.json({ data: 'Protected data', user: req.user });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
*/

/**
 * Example of permission definitions for different roles
 */
export const ROLE_EXAMPLES = {
  // Superadmin - all permissions
  superadmin: ['*'],
  
  // Admin - most permissions except system-critical ones
  admin: [
    'content.*',
    'media.*',
    'users.read',
    'users.update',
    'users.manage_roles',
    'api.tokens',
    'system.settings'
  ],
  
  // Content Manager - content and media management
  contentmanager: [
    'content.*',
    'media.upload',
    'media.read',
    'media.update',
    'users.read'
  ],
  
  // Regular User - basic operations
  user: [
    'content.create',
    'content.read',
    'content.update',
    'media.upload',
    'media.read'
  ],
  
  // Viewer - read-only access
  viewer: [
    'content.read',
    'media.read'
  ]
};

/**
 * Example of tenant-specific permissions
 */
export function checkTenantPermission(user, resource, action) {
  // Superadmin can access any tenant
  if (user.role === 'superadmin') {
    return true;
  }
  
  // Users can only access their own tenant's resources
  if (resource.tenantId !== user.tenantId) {
    return false;
  }
  
  // Check if user has the required permission
  return permissionService.hasPermission(user, action);
}