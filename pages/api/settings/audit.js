import { verifyToken, getTokenFromRequest } from '../../../lib/jwt';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import AuditLog from '../../../models/AuditLog';

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

    // Only admins can view audit logs
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    const { 
      limit = 50, 
      resource, 
      action, 
      userId: filterUserId, 
      success,
      days = 30 
    } = req.query;

    // Build filter query
    const filters = {};
    if (resource) filters.resource = resource;
    if (action) filters.action = action;
    if (filterUserId) filters.userId = filterUserId;
    if (success !== undefined) filters.success = success === 'true';

    // Add date filter
    if (days) {
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - parseInt(days));
      filters.createdAt = { $gte: startDate };
    }

    const logs = await AuditLog.find(filters)
      .populate('userId', 'name email')
      .sort({ createdAt: -1 })
      .limit(parseInt(limit));

    // Get statistics for the same period
    const stats = await AuditLog.getStatistics(parseInt(days));

    return res.status(200).json({ 
      logs, 
      stats,
      total: await AuditLog.countDocuments(filters)
    });
  } catch (error) {
    console.error('Audit log error:', error);
    return res.status(500).json({ error: 'Failed to fetch audit logs' });
  }
}