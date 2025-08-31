import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import Settings from '../../../models/Settings';
import AuditLog from '../../../models/AuditLog';
import fs from 'fs';
import path from 'path';

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
    if (!user || !user.isApproved() || user.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required' });
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
    console.error('Environment API error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req, res, user) {
  try {
    const { action } = req.query;

    if (action === 'backup') {
      return handleBackup(req, res, user);
    }

    // Get environment settings from database
    const envSettings = await Settings.find({ 
      category: 'system',
      key: { $regex: /^env_/ }
    }).sort({ key: 1 });

    // Get .env file variables (masked for security)
    const envFilePath = path.join(process.cwd(), '.env');
    let envFileVars = {};
    
    if (fs.existsSync(envFilePath)) {
      const envContent = fs.readFileSync(envFilePath, 'utf8');
      const lines = envContent.split('\n');
      
      lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
          const [key, ...valueParts] = trimmed.split('=');
          if (key && valueParts.length > 0) {
            const value = valueParts.join('=');
            // Mask sensitive values
            envFileVars[key.trim()] = maskSensitiveValue(key.trim(), value);
          }
        }
      });
    }

    return res.status(200).json({
      dbSettings: envSettings,
      fileSettings: envFileVars,
      backups: await getBackupList()
    });
  } catch (error) {
    console.error('Get environment error:', error);
    return res.status(500).json({ error: 'Failed to fetch environment settings' });
  }
}

async function handlePost(req, res, user) {
  try {
    const { action, key, value, type, description, target } = req.body;

    if (action === 'restore') {
      return handleRestore(req, res, user);
    }

    if (!key || value === undefined) {
      return res.status(400).json({ error: 'Key and value are required' });
    }

    // Validate environment variable key format
    if (!/^[A-Z][A-Z0-9_]*$/.test(key)) {
      return res.status(400).json({ 
        error: 'Environment variable keys must be uppercase with underscores only' 
      });
    }

    if (target === 'file') {
      return handleFileUpdate(req, res, user, key, value, 'create');
    } else {
      // Store in database as setting
      const settingKey = `env_${key}`;
      const setting = new Settings({
        key: settingKey,
        value,
        type: type || 'string',
        category: 'system',
        description: description || `Environment variable: ${key}`,
        isPublic: false,
        isEditable: true,
        lastModifiedBy: user._id
      });

      await setting.save();

      // Log the action
      await AuditLog.logAction({
        action: 'create',
        resource: 'environment',
        resourceId: setting._id,
        userId: user._id,
        details: { key, target: 'database' },
        ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
        userAgent: req.headers['user-agent']
      });

      return res.status(201).json({ setting });
    }
  } catch (error) {
    if (error.code === 11000) {
      return res.status(409).json({ error: 'Environment variable already exists' });
    }
    console.error('Create environment variable error:', error);
    return res.status(500).json({ error: 'Failed to create environment variable' });
  }
}

async function handlePut(req, res, user) {
  try {
    const { key, value, target } = req.body;

    if (!key || value === undefined) {
      return res.status(400).json({ error: 'Key and value are required' });
    }

    if (target === 'file') {
      return handleFileUpdate(req, res, user, key, value, 'update');
    } else {
      // Update database setting
      const settingKey = `env_${key}`;
      const setting = await Settings.findOne({ key: settingKey });
      
      if (!setting) {
        return res.status(404).json({ error: 'Environment variable not found' });
      }

      const oldValue = setting.value;
      setting.value = value;
      setting.lastModifiedBy = user._id;
      await setting.save();

      // Log the action
      await AuditLog.logAction({
        action: 'update',
        resource: 'environment',
        resourceId: setting._id,
        userId: user._id,
        details: { key, target: 'database' },
        changes: { before: oldValue, after: value },
        ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
        userAgent: req.headers['user-agent']
      });

      return res.status(200).json({ setting });
    }
  } catch (error) {
    console.error('Update environment variable error:', error);
    return res.status(500).json({ error: 'Failed to update environment variable' });
  }
}

async function handleDelete(req, res, user) {
  try {
    const { key, target } = req.body;

    if (!key) {
      return res.status(400).json({ error: 'Key is required' });
    }

    if (target === 'file') {
      return handleFileDelete(req, res, user, key);
    } else {
      // Delete from database
      const settingKey = `env_${key}`;
      const setting = await Settings.findOne({ key: settingKey });
      
      if (!setting) {
        return res.status(404).json({ error: 'Environment variable not found' });
      }

      await Settings.deleteOne({ key: settingKey });

      // Log the action
      await AuditLog.logAction({
        action: 'delete',
        resource: 'environment',
        resourceId: setting._id,
        userId: user._id,
        details: { key, target: 'database' },
        changes: { before: setting.value, after: null },
        ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
        userAgent: req.headers['user-agent']
      });

      return res.status(200).json({ message: 'Environment variable deleted successfully' });
    }
  } catch (error) {
    console.error('Delete environment variable error:', error);
    return res.status(500).json({ error: 'Failed to delete environment variable' });
  }
}

async function handleFileUpdate(req, res, user, key, value, action) {
  try {
    const envFilePath = path.join(process.cwd(), '.env');
    let envContent = '';
    
    if (fs.existsSync(envFilePath)) {
      envContent = fs.readFileSync(envFilePath, 'utf8');
    }

    const lines = envContent.split('\n');
    let keyFound = false;
    let updatedLines = [];

    // Process existing lines
    lines.forEach(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const [existingKey] = trimmed.split('=');
        if (existingKey && existingKey.trim() === key) {
          if (action === 'update') {
            updatedLines.push(`${key}=${value}`);
            keyFound = true;
          }
          // Skip line if deleting
        } else {
          updatedLines.push(line);
        }
      } else {
        updatedLines.push(line);
      }
    });

    // Add new key if creating and not found
    if (action === 'create' && !keyFound) {
      updatedLines.push(`${key}=${value}`);
    } else if (action === 'update' && !keyFound) {
      return res.status(404).json({ error: 'Environment variable not found in file' });
    }

    // Create backup before modifying
    await createBackup(user._id);

    // Write updated content
    fs.writeFileSync(envFilePath, updatedLines.join('\n'));

    // Log the action
    await AuditLog.logAction({
      action: action,
      resource: 'environment',
      userId: user._id,
      details: { key, target: 'file' },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ 
      message: `Environment variable ${action}d successfully in file`,
      requiresRestart: true
    });
  } catch (error) {
    console.error('File update error:', error);
    return res.status(500).json({ error: 'Failed to update environment file' });
  }
}

async function handleFileDelete(req, res, user, key) {
  try {
    const envFilePath = path.join(process.cwd(), '.env');
    
    if (!fs.existsSync(envFilePath)) {
      return res.status(404).json({ error: 'Environment file not found' });
    }

    const envContent = fs.readFileSync(envFilePath, 'utf8');
    const lines = envContent.split('\n');
    let keyFound = false;
    
    const updatedLines = lines.filter(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const [existingKey] = trimmed.split('=');
        if (existingKey && existingKey.trim() === key) {
          keyFound = true;
          return false; // Remove this line
        }
      }
      return true; // Keep this line
    });

    if (!keyFound) {
      return res.status(404).json({ error: 'Environment variable not found in file' });
    }

    // Create backup before modifying
    await createBackup(user._id);

    // Write updated content
    fs.writeFileSync(envFilePath, updatedLines.join('\n'));

    // Log the action
    await AuditLog.logAction({
      action: 'delete',
      resource: 'environment',
      userId: user._id,
      details: { key, target: 'file' },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ 
      message: 'Environment variable deleted successfully from file',
      requiresRestart: true
    });
  } catch (error) {
    console.error('File delete error:', error);
    return res.status(500).json({ error: 'Failed to delete from environment file' });
  }
}

async function handleBackup(req, res, user) {
  try {
    const backupId = await createBackup(user._id);
    return res.status(200).json({ 
      message: 'Backup created successfully',
      backupId,
      backups: await getBackupList()
    });
  } catch (error) {
    console.error('Backup error:', error);
    return res.status(500).json({ error: 'Failed to create backup' });
  }
}

async function handleRestore(req, res, user) {
  try {
    const { backupId } = req.body;
    
    if (!backupId) {
      return res.status(400).json({ error: 'Backup ID is required' });
    }

    const backupDir = path.join(process.cwd(), '.backups');
    const backupPath = path.join(backupDir, `${backupId}.json`);
    
    if (!fs.existsSync(backupPath)) {
      return res.status(404).json({ error: 'Backup not found' });
    }

    const backup = JSON.parse(fs.readFileSync(backupPath, 'utf8'));
    
    // Restore .env file if it exists in backup
    if (backup.envFile) {
      const envFilePath = path.join(process.cwd(), '.env');
      fs.writeFileSync(envFilePath, backup.envFile);
    }

    // Restore database settings
    if (backup.dbSettings && backup.dbSettings.length > 0) {
      // Remove existing environment settings
      await Settings.deleteMany({ 
        category: 'system',
        key: { $regex: /^env_/ }
      });

      // Restore settings
      for (const setting of backup.dbSettings) {
        const newSetting = new Settings({
          ...setting,
          _id: undefined, // Let MongoDB generate new ID
          lastModifiedBy: user._id
        });
        await newSetting.save();
      }
    }

    // Log the action
    await AuditLog.logAction({
      action: 'restore',
      resource: 'environment',
      userId: user._id,
      details: { backupId },
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    });

    return res.status(200).json({ 
      message: 'Configuration restored successfully',
      requiresRestart: true
    });
  } catch (error) {
    console.error('Restore error:', error);
    return res.status(500).json({ error: 'Failed to restore configuration' });
  }
}

async function createBackup(userId) {
  try {
    const backupDir = path.join(process.cwd(), '.backups');
    
    // Create backup directory if it doesn't exist
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupId = `env-backup-${timestamp}`;
    
    // Get current environment settings from database
    const dbSettings = await Settings.find({ 
      category: 'system',
      key: { $regex: /^env_/ }
    }).lean();

    // Get current .env file content
    const envFilePath = path.join(process.cwd(), '.env');
    let envFile = null;
    if (fs.existsSync(envFilePath)) {
      envFile = fs.readFileSync(envFilePath, 'utf8');
    }

    const backup = {
      id: backupId,
      timestamp: new Date().toISOString(),
      createdBy: userId,
      dbSettings,
      envFile
    };

    const backupPath = path.join(backupDir, `${backupId}.json`);
    fs.writeFileSync(backupPath, JSON.stringify(backup, null, 2));

    return backupId;
  } catch (error) {
    console.error('Create backup error:', error);
    throw error;
  }
}

async function getBackupList() {
  try {
    const backupDir = path.join(process.cwd(), '.backups');
    
    if (!fs.existsSync(backupDir)) {
      return [];
    }

    const files = fs.readdirSync(backupDir);
    const backups = [];

    for (const file of files) {
      if (file.endsWith('.json') && file.startsWith('env-backup-')) {
        try {
          const backupPath = path.join(backupDir, file);
          const backup = JSON.parse(fs.readFileSync(backupPath, 'utf8'));
          backups.push({
            id: backup.id,
            timestamp: backup.timestamp,
            createdBy: backup.createdBy,
            size: fs.statSync(backupPath).size
          });
        } catch (error) {
          console.error(`Error reading backup file ${file}:`, error);
        }
      }
    }

    return backups.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  } catch (error) {
    console.error('Get backup list error:', error);
    return [];
  }
}

function maskSensitiveValue(key, value) {
  const sensitiveKeys = [
    'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'API_KEY', 
    'PRIVATE', 'CREDENTIAL', 'AUTH', 'JWT'
  ];
  
  const isSensitive = sensitiveKeys.some(sensitive => 
    key.toUpperCase().includes(sensitive)
  );
  
  if (isSensitive && value && value.length > 4) {
    return value.substring(0, 4) + '*'.repeat(value.length - 4);
  }
  
  return value;
}