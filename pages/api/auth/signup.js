import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    await dbConnect();

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

    // Create new user
    const user = new User({
      email,
      password,
      name,
      status: 'pending' // User needs admin approval
    });

    await user.save();

    // Return user without password
    const userResponse = {
      id: user._id,
      email: user.email,
      name: user.name,
      status: user.status,
      role: user.role,
      createdAt: user.createdAt
    };

    res.status(201).json({
      message: 'Account created successfully. Please wait for admin approval.',
      user: userResponse
    });

  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}