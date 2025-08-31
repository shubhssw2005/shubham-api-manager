import jwt from 'jsonwebtoken';
import { getScyllaDB } from '../lib/dbConnect.js';

const JWT_SECRET = process.env.JWT_SECRET || 'b802e635a669a62c06677a295dfe2f6c';

/**
 * Simple auth middleware for ScyllaDB
 */
export async function requireApprovedUser(req, res) {
    try {
        // Skip auth for GET requests in development
        if (req.method === 'GET' && process.env.NODE_ENV === 'development') {
            return { id: 'dev-user', email: 'dev@example.com', approved: true };
        }

        const authHeader = req.headers.authorization;
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            res.status(401).json({
                success: false,
                message: 'Authorization token required'
            });
            return null;
        }

        const token = authHeader.substring(7);
        
        try {
            const decoded = jwt.verify(token, JWT_SECRET);
            
            // In a real implementation, you'd check the user in ScyllaDB
            // For now, we'll accept any valid JWT
            return {
                id: decoded.userId || decoded.id || 'unknown',
                email: decoded.email || 'unknown@example.com',
                approved: true
            };
        } catch (jwtError) {
            res.status(401).json({
                success: false,
                message: 'Invalid or expired token'
            });
            return null;
        }
    } catch (error) {
        console.error('Auth middleware error:', error);
        res.status(500).json({
            success: false,
            message: 'Authentication error'
        });
        return null;
    }
}

/**
 * Generate a JWT token for testing
 */
export function generateTestToken(userId = 'test-user', email = 'test@example.com') {
    return jwt.sign(
        { 
            userId, 
            email, 
            approved: true,
            iat: Math.floor(Date.now() / 1000),
            exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60) // 24 hours
        },
        JWT_SECRET
    );
}

export default { requireApprovedUser, generateTestToken };