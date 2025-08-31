import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import Settings from '../../../models/Settings';
import AuditLog from '../../../models/AuditLog';
import { checkPermission } from '../../../lib/rbac';

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
    console.error('Settings API error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req, res, user) {
  try {
    const { category, includePrivate } = req.query;
    
    // Check if user can view private settings
    const canViewPrivate = user.role === 'admin' || includePrivate === 'false';
    
    let settings;
    if (category) {
      settings = await Settings.getByCategory(category, canViewPrivate);
    } else {
      const query = canViewPrivate ? {} : { isPublic: true };
      settings = await Settings.find(query).sort({ category: 1, key: 1 });
    }

    return res.status(200).json({ settings });
  } catch (error) {
    console.error('Get settings error:', error);
    return res.status(500).json({ error: 'Failed to fetch settings' });
  }
}

async function handlePost(req, res, user) {
  try {
    // Only admins can create new settings
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    const { key, value, type, category, description, isPublic, isEditable, validation } = req.body;

    if (!key || value === undefined || !type || !category) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Validate the value against the type
    if (!Settings.validateValue(value, type, validation)) {
      return res.status(400).json({ error: 'Invalid value for the specified type' });
    }

    const setting = new Settings({
      key,
      value,
      type,
      category,
      description,
      isPublic: isPublic || false,
      isEditable: isEditable !== false, // default to true
      validation,
      lastModifiedBy: user._id
    });

    await setting.save();

    // Log the action
    await AuditLog.logAction({
      action: 'create',
      resource: 'settings',
      resourceId: setting._id,
      userId: user._id,
      details: { key, category },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(201).json({ setting });
  } catch (error) {
    if (error.code === 11000) {
      return res.status(409).json({ error: 'Setting key already exists' });
    }
    console.error('Create setting error:', error);
    return res.status(500).json({ error: 'Failed to create setting' });
  }
}

async function handlePut(req, res, user) {
  try {
    const { key, value } = req.body;

    if (!key || value === undefined) {
      return res.status(400).json({ error: 'Missing key or value' });
    }

    const setting = await Settings.findOne({ key });
    if (!setting) {
      return res.status(404).json({ error: 'Setting not found' });
    }

    if (!setting.isEditable) {
      return res.status(403).json({ error: 'Setting is not editable' });
    }

    // Check permissions - admins can edit all, users can only edit public settings
    if (user.role !== 'admin' && !setting.isPublic) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    // Store old value for audit log
    const oldValue = setting.value;

    // Validate the new value
    if (!Settings.validateValue(value, setting.type, setting.validation)) {
      return res.status(400).json({ error: 'Invalid value for the setting type' });
    }

    setting.value = value;
    setting.lastModifiedBy = user._id;
    await setting.save();

    // Log the action
    await AuditLog.logAction({
      action: 'settings_change',
      resource: 'settings',
      resourceId: setting._id,
      userId: user._id,
      details: { key },
      changes: { before: oldValue, after: value },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ setting });
  } catch (error) {
    console.error('Update setting error:', error);
    return res.status(500).json({ error: 'Failed to update setting' });
  }
}

async function handleDelete(req, res, user) {
  try {
    // Only admins can delete settings
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    const { key } = req.body;
    if (!key) {
      return res.status(400).json({ error: 'Missing setting key' });
    }

    const setting = await Settings.findOne({ key });
    if (!setting) {
      return res.status(404).json({ error: 'Setting not found' });
    }

    if (!setting.isEditable) {
      return res.status(403).json({ error: 'Setting cannot be deleted' });
    }

    await Settings.deleteOne({ key });

    // Log the action
    await AuditLog.logAction({
      action: 'delete',
      resource: 'settings',
      resourceId: setting._id,
      userId: user._id,
      details: { key, category: setting.category },
      changes: { before: setting.value, after: null },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ message: 'Setting deleted successfully' });
  } catch (error) {
    console.error('Delete setting error:', error);
    return res.status(500).json({ error: 'Failed to delete setting' });
  }
}