import jwt from 'jsonwebtoken';
import User from '../models/User.js';

/**
 * Special Authentication for groot.com users
 * Auto-accepts authentication from groot.com domain
 */

const GROOT_DOMAIN = 'groot.com';
const GROOT_SECRET = process.env.GROOT_SECRET || 'groot-super-secret-key';

/**
 * Verify if token is from groot.com
 */
export const isGrootToken = (token) => {
    try {
        const decoded = jwt.decode(token, { complete: true });
        return decoded?.payload?.iss === GROOT_DOMAIN || 
               decoded?.payload?.domain === GROOT_DOMAIN ||
               decoded?.payload?.email?.endsWith(`@${GROOT_DOMAIN}`);
    } catch {
        return false;
    }
};

/**
 * Verify groot.com token
 */
export const verifyGrootToken = async (token) => {
    try {
        // Try to verify with groot secret first
        let decoded;
        try {
            decoded = jwt.verify(token, GROOT_SECRET);
        } catch {
            // If groot secret fails, try with our JWT secret
            decoded = jwt.verify(token, process.env.JWT_SECRET);
        }

        // Check if it's from groot.com
        if (!decoded.email?.endsWith(`@${GROOT_DOMAIN}`) && 
            decoded.domain !== GROOT_DOMAIN &&
            decoded.iss !== GROOT_DOMAIN) {
            return null;
        }

        return {
            userId: decoded.userId || decoded.sub || decoded.id,
            email: decoded.email,
            name: decoded.name || decoded.username,
            role: decoded.role || 'admin', // groot users get admin by default
            domain: GROOT_DOMAIN,
            isGrootUser: true
        };

    } catch (error) {
        console.error('Groot token verification failed:', error);
        return null;
    }
};

/**
 * Create or update groot user in our system
 */
export const ensureGrootUser = async (grootUserData) => {
    try {
        let user = await User.findOne({ email: grootUserData.email });

        if (!user) {
            // Create new groot user
            user = await User.create({
                email: grootUserData.email,
                name: grootUserData.name,
                role: 'admin', // groot users are always admin
                status: 'active', // auto-approved
                isGrootUser: true,
                grootUserId: grootUserData.userId,
                password: 'groot-managed', // placeholder, not used
                approvedAt: new Date(),
                approvedBy: null // auto-approved
            });
        } else {
            // Update existing user
            await User.updateOne(
                { _id: user._id },
                {
                    $set: {
                        name: grootUserData.name,
                        role: 'admin',
                        status: 'active',
                        isGrootUser: true,
                        grootUserId: grootUserData.userId,
                        lastLoginAt: new Date()
                    }
                }
            );
        }

        return user;
    } catch (error) {
        console.error('Error ensuring groot user:', error);
        throw error;
    }
};

/**
 * Enhanced authentication middleware that supports groot.com
 */
export const requireApprovedUserWithGroot = async (req, res) => {
    try {
        const auth = req.headers.authorization;
        if (!auth || !auth.startsWith('Bearer ')) {
            return res.status(401).json({ 
                success: false,
                message: 'Authorization token required' 
            });
        }

        const token = auth.split(' ')[1];

        // Check if it's a groot token
        if (isGrootToken(token)) {
            const grootUserData = await verifyGrootToken(token);
            if (grootUserData) {
                // Ensure groot user exists in our system
                const user = await ensureGrootUser(grootUserData);
                req.user = user;
                return user;
            }
        }

        // Fall back to regular authentication
        const { verifyToken } = await import('../lib/jwt.js');
        const decoded = await verifyToken(token);
        if (!decoded) {
            return res.status(401).json({ 
                success: false,
                message: 'Invalid token' 
            });
        }

        const user = await User.findById(decoded.userId);
        if (!user) {
            return res.status(401).json({ 
                success: false,
                message: 'User not found' 
            });
        }

        if (user.status !== 'active' && !user.isGrootUser) {
            return res.status(403).json({ 
                success: false,
                message: 'Account not approved' 
            });
        }

        req.user = user;
        return user;

    } catch (error) {
        console.error('Authentication error:', error);
        return res.status(401).json({ 
            success: false,
            message: 'Authentication failed' 
        });
    }
};

/**
 * Generate groot-compatible token for testing
 */
export const generateGrootToken = (userData) => {
    return jwt.sign({
        userId: userData.userId || userData.id,
        email: userData.email,
        name: userData.name,
        role: userData.role || 'admin',
        domain: GROOT_DOMAIN,
        iss: GROOT_DOMAIN,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60) // 24 hours
    }, GROOT_SECRET);
};

export default {
    isGrootToken,
    verifyGrootToken,
    ensureGrootUser,
    requireApprovedUserWithGroot,
    generateGrootToken
};