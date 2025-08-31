import { requireAdmin } from '../../../middleware/auth';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';

export default async function handler(req, res) {
  try {
    const admin = await requireAdmin(req, res);
    if (!admin) return; // Error already handled by middleware

    await dbConnect();

    if (req.method === 'GET') {
      // Get all users with pagination
      const page = parseInt(req.query.page) || 1;
      const limit = parseInt(req.query.limit) || 10;
      const status = req.query.status; // Filter by status if provided
      
      const skip = (page - 1) * limit;
      
      let query = {};
      if (status) {
        query.status = status;
      }

      const users = await User.find(query)
        .select('-password')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .populate('approvedBy', 'name email');

      const total = await User.countDocuments(query);

      res.status(200).json({
        users,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit)
        }
      });

    } else {
      res.status(405).json({ error: 'Method not allowed' });
    }

  } catch (error) {
    console.error('Admin users error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}