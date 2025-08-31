import dbConnect from '../../../lib/dbConnect';
import User from '../../../models/User';
import { signToken } from '../../../lib/jwt';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    console.log('Starting signin process...');

    console.log('Connecting to database in environment:', process.env.NODE_ENV);
    await dbConnect().catch(err => {
      console.error('Database connection error:', err);
      throw new Error('Database connection failed');
    });
    console.log('Database connected successfully');

    const { email, password } = req.body;
    console.log('Attempting login for:', email);

    // Validation
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Find user
    console.log('Finding user...');
    const user = await User.findOne({ email }).catch(err => {
      console.error('Error finding user:', err);
      return null;
    });

    if (!user) {
      console.log('User not found');
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    console.log('User found, checking password...');
    // Check password
    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Check if user is approved
    if (user.status === 'pending') {
      return res.status(403).json({ error: 'Account pending admin approval' });
    }

    if (user.status === 'rejected') {
      return res.status(403).json({ error: 'Account has been rejected' });
    }

    // Generate JWT token
    const token = signToken({
      userId: user._id,
      email: user.email,
      role: user.role
    });

    // Set cookie
    res.setHeader('Set-Cookie', `token=${token}; HttpOnly; Path=/; Max-Age=604800; SameSite=Strict`);

    // Return user data and token
    const userResponse = {
      id: user._id,
      email: user.email,
      name: user.name,
      role: user.role,
      status: user.status
    };

    res.status(200).json({
      message: 'Login successful',
      user: userResponse,
      token
    });

  } catch (error) {
    console.error('Error in signin handler:', error);
    return res.status(500).json({
      error: 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? {
        message: error.message,
        stack: error.stack,
        name: error.name
      } : undefined
    });
  }
}