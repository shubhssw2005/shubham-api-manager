import { verifyToken, getTokenFromRequest } from '../lib/jwt';
import dbConnect from '../lib/dbConnect';
import User from '../models/User';

export async function requireAuth(req, res, next) {
  try {
    const token = getTokenFromRequest(req);
    
    if (!token) {
      if (res) {
        return res.status(401).json({ error: 'Access token required' });
      }
      return null;
    }

    const decoded = verifyToken(token);
    if (!decoded) {
      if (res) {
        return res.status(401).json({ error: 'Invalid or expired token' });
      }
      return null;
    }

    await dbConnect();
    const user = await User.findById(decoded.userId).select('-password');
    
    if (!user) {
      if (res) {
        return res.status(401).json({ error: 'User not found' });
      }
      return null;
    }

    // For general auth, we allow pending users but mark their status
    req.user = user;
    if (next) next();
    return user;
  } catch (error) {
    console.error('Auth error:', error);
    if (res) {
      return res.status(500).json({ error: 'Authentication error' });
    }
    return null;
  }
}

export async function requireApprovedUser(req, res, next) {
  try {
    const user = await requireAuth(req, res);
    if (!user) {
      return null; // Error already handled by requireAuth
    }

    // Only allow active users
    if (user.status !== 'active') {
      if (res) {
        return res.status(403).json({ 
          error: user.status === 'pending' ? 'Account pending approval' : 'Account has been rejected' 
        });
      }
      return null;
    }

    if (next) next();
    return user;
  } catch (error) {
    console.error('Approved user check error:', error);
    if (res) {
      return res.status(500).json({ error: 'Authorization error' });
    }
    return null;
  }
}

export async function requireAdmin(req, res, next) {
  try {
    const user = await requireApprovedUser(req, res);
    
    if (!user) {
      return null; // Error already handled by requireApprovedUser
    }

    if (user.role !== 'admin') {
      if (res) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      return null;
    }

    if (next) next();
    return user;
  } catch (error) {
    console.error('Admin check error:', error);
    if (res) {
      return res.status(500).json({ error: 'Authorization error' });
    }
    return null;
  }
}