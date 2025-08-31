import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import AuditLog from '../../../models/AuditLog';
import APIToken from '../../../models/APIToken';

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

    // Only admins can view system notifications
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    // Generate system notifications
    const notifications = await generateSystemNotifications();

    return res.status(200).json({
      notifications,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Notifications error:', error);
    return res.status(500).json({ error: 'Failed to fetch notifications' });
  }
}

async function generateSystemNotifications() {
  const notifications = [];

  try {
    // Check for recent failed login attempts
    const recentFailures = await AuditLog.countDocuments({
      action: 'login',
      success: false,
      createdAt: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
    });

    if (recentFailures > 5) {
      notifications.push({
        type: 'warning',
        title: 'Multiple Failed Login Attempts',
        message: `${recentFailures} failed login attempts in the last 24 hours`,
        createdAt: new Date().toISOString()
      });
    }

    // Check for expiring API tokens
    const expiringTokens = await APIToken.countDocuments({
      isActive: true,
      expiresAt: {
        $gte: new Date(),
        $lte: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // Next 7 days
      }
    });

    if (expiringTokens > 0) {
      notifications.push({
        type: 'warning',
        title: 'API Tokens Expiring Soon',
        message: `${expiringTokens} API token(s) will expire within the next 7 days`,
        createdAt: new Date().toISOString()
      });
    }

    // Check for pending user approvals
    const pendingUsers = await User.countDocuments({ status: 'pending' });
    if (pendingUsers > 0) {
      notifications.push({
        type: 'info',
        title: 'Pending User Approvals',
        message: `${pendingUsers} user(s) are waiting for approval`,
        createdAt: new Date().toISOString()
      });
    }

    // Check system health
    const memoryUsage = getMemoryUsagePercentage();
    if (memoryUsage > 80) {
      notifications.push({
        type: 'error',
        title: 'High Memory Usage',
        message: `System memory usage is at ${memoryUsage.toFixed(1)}%`,
        createdAt: new Date().toISOString()
      });
    }

    // Check for recent errors
    const recentErrors = await AuditLog.countDocuments({
      success: false,
      createdAt: { $gte: new Date(Date.now() - 60 * 60 * 1000) } // Last hour
    });

    if (recentErrors > 10) {
      notifications.push({
        type: 'error',
        title: 'High Error Rate',
        message: `${recentErrors} errors occurred in the last hour`,
        createdAt: new Date().toISOString()
      });
    }

    // Add a success notification if system is healthy
    if (notifications.length === 0) {
      notifications.push({
        type: 'success',
        title: 'System Running Smoothly',
        message: 'All systems are operating normally',
        createdAt: new Date().toISOString()
      });
    }

  } catch (error) {
    console.error('Error generating notifications:', error);
    notifications.push({
      type: 'error',
      title: 'Notification System Error',
      message: 'Unable to generate system notifications',
      createdAt: new Date().toISOString()
    });
  }

  return notifications.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
}

function getMemoryUsagePercentage() {
  const used = process.memoryUsage();
  const totalMemory = used.heapTotal;
  const usedMemory = used.heapUsed;
  
  return (usedMemory / totalMemory) * 100;
}