import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import APIToken from '../../../models/APIToken';
import AuditLog from '../../../models/AuditLog';

export default async function handler(req, res) {
  try {
    await dbConnect();

    const token = getTokenFromRequest(req);
    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const decoded = verifyToken(token);
    if (!decoded) {
      return res.status(401).json({ error: 'Invalid or expired token' });
    }

    const user = await User.findById(decoded.userId);
    if (!user || !user.isApproved()) {
      return res.status(401).json({ error: 'User not authorized' });
    }

    // Only admins can manage API tokens
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    switch (req.method) {
      case 'GET':
        return handleGet(req, res, user);
      case 'POST':
        return handlePost(req, res, user);
      case 'PUT':
        return handlePut(req, res, user);
      case 'DELETE':
        return handleDelete(req, res, user);
      default:
        return res.status(405).json({ error: 'Method not allowed' });
    }
  } catch (error) {
    console.error('API Tokens error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req, res, user) {
  try {
    const { includeInactive } = req.query;
    
    const query = includeInactive === 'true' ? {} : { isActive: true };
    const tokens = await APIToken.find(query)
      .populate('createdBy', 'name email')
      .sort({ createdAt: -1 });

    return res.status(200).json({ tokens });
  } catch (error) {
    console.error('Get tokens error:', error);
    return res.status(500).json({ error: 'Failed to fetch tokens' });
  }
}

async function handlePost(req, res, user) {
  try {
    const { 
      name, 
      permissions, 
      rateLimit, 
      expiresAt, 
      description 
    } = req.body;

    if (!name || !permissions || !Array.isArray(permissions)) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Validate permissions structure
    for (const perm of permissions) {
      if (!perm.model || !perm.actions || !Array.isArray(perm.actions)) {
        return res.status(400).json({ error: 'Invalid permissions structure' });
      }
    }

    // Generate token
    const tokenValue = APIToken.generateToken();
    
    const apiToken = new APIToken({
      name,
      token: tokenValue,
      permissions,
      rateLimit: rateLimit || { requests: 1000, window: 3600 },
      expiresAt: expiresAt ? new Date(expiresAt) : undefined,
      description,
      createdBy: user._id
    });

    await apiToken.save();

    // Log the action
    await AuditLog.logAction({
      action: 'token_create',
      resource: 'api_token',
      resourceId: apiToken._id,
      userId: user._id,
      details: { name, permissions: permissions.length },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    // Return the token value only once (for the user to copy)
    const response = {
      ...apiToken.toJSON(),
      token: tokenValue // This will be the only time the token is returned
    };

    return res.status(201).json({ token: response });
  } catch (error) {
    console.error('Create token error:', error);
    return res.status(500).json({ error: 'Failed to create token' });
  }
}

async function handlePut(req, res, user) {
  try {
    const { id, name, permissions, rateLimit, expiresAt, description, isActive } = req.body;

    if (!id) {
      return res.status(400).json({ error: 'Missing token ID' });
    }

    const apiToken = await APIToken.findById(id);
    if (!apiToken) {
      return res.status(404).json({ error: 'Token not found' });
    }

    // Store old values for audit log
    const oldValues = {
      name: apiToken.name,
      permissions: apiToken.permissions,
      rateLimit: apiToken.rateLimit,
      isActive: apiToken.isActive
    };

    // Update fields
    if (name !== undefined) apiToken.name = name;
    if (permissions !== undefined) {
      // Validate permissions structure
      for (const perm of permissions) {
        if (!perm.model || !perm.actions || !Array.isArray(perm.actions)) {
          return res.status(400).json({ error: 'Invalid permissions structure' });
        }
      }
      apiToken.permissions = permissions;
    }
    if (rateLimit !== undefined) apiToken.rateLimit = rateLimit;
    if (expiresAt !== undefined) apiToken.expiresAt = expiresAt ? new Date(expiresAt) : null;
    if (description !== undefined) apiToken.description = description;
    if (isActive !== undefined) apiToken.isActive = isActive;

    await apiToken.save();

    // Log the action
    await AuditLog.logAction({
      action: 'update',
      resource: 'api_token',
      resourceId: apiToken._id,
      userId: user._id,
      details: { name: apiToken.name },
      changes: { 
        before: oldValues, 
        after: {
          name: apiToken.name,
          permissions: apiToken.permissions,
          rateLimit: apiToken.rateLimit,
          isActive: apiToken.isActive
        }
      },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ token: apiToken });
  } catch (error) {
    console.error('Update token error:', error);
    return res.status(500).json({ error: 'Failed to update token' });
  }
}

async function handleDelete(req, res, user) {
  try {
    const { id } = req.body;

    if (!id) {
      return res.status(400).json({ error: 'Missing token ID' });
    }

    const apiToken = await APIToken.findById(id);
    if (!apiToken) {
      return res.status(404).json({ error: 'Token not found' });
    }

    // Revoke the token instead of deleting it (for audit purposes)
    await apiToken.revoke();

    // Log the action
    await AuditLog.logAction({
      action: 'token_revoke',
      resource: 'api_token',
      resourceId: apiToken._id,
      userId: user._id,
      details: { name: apiToken.name },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ message: 'Token revoked successfully' });
  } catch (error) {
    console.error('Delete token error:', error);
    return res.status(500).json({ error: 'Failed to revoke token' });
  }
}