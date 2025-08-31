import TokenService from '../../../lib/auth/TokenService.js';
import dbConnect from '../../../lib/dbConnect.js';
import User from '../../../models/User.js';

const tokenService = new TokenService();

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    await dbConnect();

    const { email, password, tenantId } = req.body;

    if (!email || !password) {
      return res.status(400).json({ 
        error: 'Email and password are required',
        code: 'MISSING_CREDENTIALS'
      });
    }

    // Find user by email
    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) {
      return res.status(401).json({ 
        error: 'Invalid credentials',
        code: 'INVALID_CREDENTIALS'
      });
    }

    // Check password
    const isValidPassword = await user.comparePassword(password);
    if (!isValidPassword) {
      return res.status(401).json({ 
        error: 'Invalid credentials',
        code: 'INVALID_CREDENTIALS'
      });
    }

    // Check if user is approved
    if (user.status !== 'active') {
      return res.status(403).json({ 
        error: user.status === 'pending' ? 'Account pending approval' : 'Account has been deactivated',
        code: 'ACCOUNT_INACTIVE',
        status: user.status
      });
    }

    // Add request context for token generation
    const userWithContext = {
      ...user.toObject(),
      userAgent: req.headers['user-agent'],
      ipAddress: req.headers['x-forwarded-for'] || req.connection.remoteAddress
    };

    // Generate JWT token pair
    const tokenPair = await tokenService.generateTokenPair(userWithContext, tenantId);

    // Update last login
    user.lastLoginAt = new Date();
    await user.save();

    // Set refresh token as httpOnly cookie
    res.setHeader('Set-Cookie', [
      `refreshToken=${tokenPair.refreshToken}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=${7 * 24 * 60 * 60}`, // 7 days
      `accessToken=${tokenPair.accessToken}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=${15 * 60}` // 15 minutes
    ]);

    res.status(200).json({
      success: true,
      data: {
        accessToken: tokenPair.accessToken,
        refreshToken: tokenPair.refreshToken,
        expiresIn: tokenPair.expiresIn,
        tokenType: tokenPair.tokenType,
        user: {
          id: user._id,
          email: user.email,
          name: user.name,
          role: user.role,
          status: user.status,
          tenantId: tenantId || user.tenantId,
          isGrootUser: user.isGrootUser
        }
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      code: 'LOGIN_ERROR'
    });
  }
}
