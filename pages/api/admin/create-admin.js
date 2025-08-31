import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    await dbConnect();

    // Check if any admin exists
    const adminExists = await User.findOne({ role: 'admin' });
    if (adminExists) {
      return res.status(400).json({ error: 'Admin already exists. Use regular signup.' });
    }

    const { email, password, name } = req.body;

    // Validation
    if (!email || !password || !name) {
      return res.status(400).json({ error: 'All fields are required' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists with this email' });
    }

    // Create first admin user
    const admin = new User({
      email,
      password,
      name,
      role: 'admin',
      status: 'approved' // Admin is auto-approved
    });

    await admin.save();

    const adminResponse = {
      id: admin._id,
      email: admin.email,
      name: admin.name,
      role: admin.role,
      status: admin.status,
      createdAt: admin.createdAt
    };

    res.status(201).json({
      message: 'Admin account created successfully',
      user: adminResponse
    });

  } catch (error) {
    console.error('Create admin error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}