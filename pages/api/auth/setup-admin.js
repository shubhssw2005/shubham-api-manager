import User from '../../../models/User';
import bcryptjs from 'bcryptjs';
import { ROLES } from '../../../lib/rbac';
import mongoose from 'mongoose';
import dbConnect from '../../../lib/dbConnect';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

    try {
    console.log('Setting up admin user in environment:', process.env.NODE_ENV);
    await dbConnect();

    // Check if any user exists
    const userCount = await User.countDocuments();
    if (userCount > 0) {
      return res.status(400).json({ error: 'Admin already exists. Cannot create first admin again.' });
    }

    const { email, password, name } = req.body;

    // Validation
    if (!email || !password || !name) {
      return res.status(400).json({ error: 'All fields are required' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    // Hash password
    const hashedPassword = await bcryptjs.hash(password, 12);

    // Create superadmin user
    const user = await User.create({
      email,
      password: hashedPassword,
      name,
      role: 'superadmin',
      status: 'approved'
    });

    res.status(201).json({
      success: true,
      message: 'Superadmin account created successfully',
      data: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        status: user.status
      }
    });

  } catch (error) {
    console.error('Setup admin error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
}
