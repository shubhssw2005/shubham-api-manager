import { jwtAuth } from '../../../middleware/jwtAuth.js';
import PermissionService from '../../../lib/auth/PermissionService.js';

const permissionService = new PermissionService();

export default async function handler(req, res) {
  // Apply JWT authentication
  await new Promise((resolve, reject) => {
    jwtAuth(req, res, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });

  if (req.method === 'GET') {
    return await handleGetPermissions(req, res);
  } else if (req.method === 'POST') {
    return await handleCheckPermissions(req, res);
  } else {
    return res.status(405).json({ error: 'Method not allowed' });
  }
}

/**
 * Get user's permissions
 */
async function handleGetPermissions(req, res) {
  try {
    const permissions = await permissionService.getUserPermissions(req.user);
    
    res.status(200).json({
      success: true,
      data: {
        permissions,
        role: req.user.role,
        userId: req.user.id,
        tenantId: req.user.tenantId
      }
    });
  } catch (error) {
    console.error('Get permissions error:', error);
    res.status(500).json({
      error: 'Failed to get permissions',
      code: 'GET_PERMISSIONS_ERROR'
    });
  }
}

/**
 * Check specific permissions
 */
async function handleCheckPermissions(req, res) {
  try {
    const { permissions, checkType = 'any', resource } = req.body;
    
    if (!permissions || !Array.isArray(permissions)) {
      return res.status(400).json({
        error: 'Permissions array required',
        code: 'MISSING_PERMISSIONS'
      });
    }

    let hasPermission = false;
    const results = {};

    if (checkType === 'all') {
      hasPermission = await permissionService.hasAllPermissions(req.user, permissions, resource);
    } else if (checkType === 'any') {
      hasPermission = await permissionService.hasAnyPermission(req.user, permissions, resource);
    } else {
      // Check each permission individually
      for (const permission of permissions) {
        results[permission] = await permissionService.hasPermission(req.user, permission, resource);
      }
      hasPermission = Object.values(results).some(result => result);
    }

    res.status(200).json({
      success: true,
      data: {
        hasPermission,
        checkType,
        permissions,
        results: Object.keys(results).length > 0 ? results : undefined,
        user: {
          id: req.user.id,
          role: req.user.role,
          tenantId: req.user.tenantId
        }
      }
    });
  } catch (error) {
    console.error('Check permissions error:', error);
    res.status(500).json({
      error: 'Failed to check permissions',
      code: 'CHECK_PERMISSIONS_ERROR'
    });
  }
}