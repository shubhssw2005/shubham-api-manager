import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import Settings from '../../../models/Settings';
import APIToken from '../../../models/APIToken';
import AuditLog from '../../../models/AuditLog';
import { Role } from '../../../models/Role';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

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

    // Only admins can view system stats
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    // Get various system statistics
    const [
      userStats,
      tokenStats,
      settingsStats,
      recentActivity,
      systemHealth
    ] = await Promise.all([
      getUserStats(),
      getTokenStats(),
      getSettingsStats(),
      getRecentActivity(),
      getSystemHealth()
    ]);

    return res.status(200).json({
      users: userStats,
      tokens: tokenStats,
      settings: settingsStats,
      recentActivity,
      systemHealth,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('System stats error:', error);
    return res.status(500).json({ error: 'Failed to fetch system statistics' });
  }
}

async function getUserStats() {
  const [total, approved, pending, admins] = await Promise.all([
    User.countDocuments(),
    User.countDocuments({ status: 'approved' }),
    User.countDocuments({ status: 'pending' }),
    User.countDocuments({ role: 'admin' })
  ]);

  // Get recent registrations (last 7 days)
  const weekAgo = new Date();
  weekAgo.setDate(weekAgo.getDate() - 7);
  const recentRegistrations = await User.countDocuments({ 
    createdAt: { $gte: weekAgo } 
  });

  return {
    total,
    approved,
    pending,
    admins,
    recentRegistrations
  };
}

async function getTokenStats() {
  const [total, active, expired] = await Promise.all([
    APIToken.countDocuments(),
    APIToken.countDocuments({ isActive: true }),
    APIToken.countDocuments({ 
      $or: [
        { isActive: false },
        { expiresAt: { $lt: new Date() } }
      ]
    })
  ]);

  // Get most used tokens (top 5)
  const mostUsed = await APIToken.find({ isActive: true })
    .sort({ 'usage.totalRequests': -1 })
    .limit(5)
    .select('name usage.totalRequests usage.lastUsed')
    .populate('createdBy', 'name');

  return {
    total,
    active,
    expired,
    mostUsed
  };
}

async function getSettingsStats() {
  const [total, byCategory] = await Promise.all([
    Settings.countDocuments(),
    Settings.aggregate([
      {
        $group: {
          _id: '$category',
          count: { $sum: 1 },
          public: { $sum: { $cond: ['$isPublic', 1, 0] } },
          editable: { $sum: { $cond: ['$isEditable', 1, 0] } }
        }
      }
    ])
  ]);

  return {
    total,
    byCategory
  };
}

async function getRecentActivity() {
  const activities = await AuditLog.getRecentActivity(10);
  
  // Get activity counts by action for the last 24 hours
  const dayAgo = new Date();
  dayAgo.setDate(dayAgo.getDate() - 1);
  
  const activityCounts = await AuditLog.aggregate([
    { $match: { createdAt: { $gte: dayAgo } } },
    {
      $group: {
        _id: '$action',
        count: { $sum: 1 }
      }
    },
    { $sort: { count: -1 } }
  ]);

  return {
    recent: activities,
    counts: activityCounts
  };
}

async function getSystemHealth() {
  try {
    // Check database connection
    const dbStatus = await checkDatabaseHealth();
    
    // Check if critical settings exist
    const criticalSettings = await checkCriticalSettings();
    
    // Check for failed operations in the last hour
    const hourAgo = new Date();
    hourAgo.setHours(hourAgo.getHours() - 1);
    const recentErrors = await AuditLog.countDocuments({
      success: false,
      createdAt: { $gte: hourAgo }
    });

    return {
      database: dbStatus,
      settings: criticalSettings,
      recentErrors,
      status: dbStatus && criticalSettings && recentErrors < 10 ? 'healthy' : 'warning'
    };
  } catch (error) {
    return {
      database: false,
      settings: false,
      recentErrors: -1,
      status: 'error',
      error: error.message
    };
  }
}

async function checkDatabaseHealth() {
  try {
    // Simple database connectivity check
    await User.findOne().limit(1);
    return true;
  } catch (error) {
    return false;
  }
}

async function checkCriticalSettings() {
  try {
    // Check if basic system settings exist
    const criticalKeys = ['system.name', 'system.version', 'api.rate_limit'];
    const existingSettings = await Settings.find({ 
      key: { $in: criticalKeys } 
    }).countDocuments();
    
    // Return true if at least some critical settings exist
    return existingSettings > 0;
  } catch (error) {
    return false;
  }
}