import { requireAdmin } from '../../../middleware/auth';
import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const admin = await requireAdmin(req, res);
    if (!admin) return; // Error already handled by middleware

    await dbConnect();

    const { userId, action } = req.body; // action: 'approve' or 'reject'

    if (!userId || !action) {
      return res.status(400).json({ error: 'User ID and action are required' });
    }

    if (!['approve', 'reject'].includes(action)) {
      return res.status(400).json({ error: 'Action must be approve or reject' });
    }

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (user.status !== 'pending') {
      return res.status(400).json({ error: 'User is not in pending status' });
    }

    // Update user status
    user.status = action === 'approve' ? 'approved' : 'rejected';
    user.approvedBy = admin._id;
    user.approvedAt = new Date();

    await user.save();

    const userResponse = {
      id: user._id,
      email: user.email,
      name: user.name,
      status: user.status,
      role: user.role,
      createdAt: user.createdAt,
      approvedAt: user.approvedAt
    };

    res.status(200).json({
      message: `User ${action}d successfully`,
      user: userResponse
    });

  } catch (error) {
    console.error('Approve user error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}