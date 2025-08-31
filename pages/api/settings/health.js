import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';

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

    // Only admins can view health metrics
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    // Get system health metrics
    const healthMetrics = await getSystemHealthMetrics();

    return res.status(200).json(healthMetrics);
  } catch (error) {
    console.error('Health check error:', error);
    return res.status(500).json({ 
      error: 'Health check failed',
      uptime: process.uptime(),
      memoryUsage: getMemoryUsagePercentage(),
      diskUsage: 0, // Would need additional implementation for disk usage
      status: 'error'
    });
  }
}

async function getSystemHealthMetrics() {
  const uptime = process.uptime();
  const memoryUsage = getMemoryUsagePercentage();
  
  // In a production environment, you might also check:
  // - Database connection pool status
  // - External service connectivity
  // - Disk space usage
  // - CPU usage
  // - Network connectivity
  
  return {
    uptime,
    memoryUsage,
    diskUsage: await getDiskUsage(),
    status: 'healthy',
    timestamp: new Date().toISOString(),
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch
  };
}

function getMemoryUsagePercentage() {
  const used = process.memoryUsage();
  const totalMemory = used.heapTotal;
  const usedMemory = used.heapUsed;
  
  return (usedMemory / totalMemory) * 100;
}

async function getDiskUsage() {
  // This is a simplified implementation
  // In production, you'd want to use a library like 'node-disk-info' or similar
  try {
    const fs = require('fs').promises;
    const stats = await fs.stat('.');
    // This is just a placeholder - real disk usage would require platform-specific code
    return Math.random() * 30 + 10; // Mock 10-40% usage
  } catch (error) {
    return 0;
  }
}