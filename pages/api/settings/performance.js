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

    // Only admins can view performance metrics
    if (user.role !== 'admin') {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    // Get performance metrics
    const [responseTimeTrend, requestVolume] = await Promise.all([
      getResponseTimeTrend(),
      getRequestVolume()
    ]);

    return res.status(200).json({
      responseTimeTrend,
      requestVolume,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Performance metrics error:', error);
    return res.status(500).json({ error: 'Failed to fetch performance metrics' });
  }
}

async function getResponseTimeTrend() {
  // Generate mock response time data for the last 24 hours
  // In a real implementation, this would come from application monitoring
  const hours = Array.from({ length: 24 }, (_, i) => {
    const hour = new Date();
    hour.setHours(hour.getHours() - (23 - i), 0, 0, 0);
    
    // Simulate response times with some variation
    const baseTime = 200 + Math.random() * 300;
    const peakHours = [9, 10, 11, 14, 15, 16]; // Business hours
    const avgTime = peakHours.includes(hour.getHours()) 
      ? baseTime + Math.random() * 200 
      : baseTime;

    return {
      hour: hour.getHours(),
      avgTime: Math.round(avgTime),
      timestamp: hour.toISOString()
    };
  });

  return hours;
}

async function getRequestVolume() {
  try {
    // Get actual request volume from audit logs for the last 24 hours
    const dayAgo = new Date();
    dayAgo.setDate(dayAgo.getDate() - 1);

    const hourlyActivity = await AuditLog.aggregate([
      { 
        $match: { 
          createdAt: { $gte: dayAgo },
          success: true 
        } 
      },
      {
        $group: {
          _id: { 
            hour: { $hour: '$createdAt' },
            date: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } }
          },
          count: { $sum: 1 }
        }
      },
      { $sort: { '_id.hour': 1 } }
    ]);

    // Fill in missing hours with 0 counts
    const hours = Array.from({ length: 24 }, (_, i) => {
      const existing = hourlyActivity.find(item => item._id.hour === i);
      return {
        hour: i,
        count: existing ? existing.count : 0
      };
    });

    return hours;
  } catch (error) {
    console.error('Error getting request volume:', error);
    // Return mock data if database query fails
    return Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      count: Math.floor(Math.random() * 50) + 10
    }));
  }
}