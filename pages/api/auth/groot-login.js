import dbConnect from '../../../lib/dbConnect.js';
import { generateGrootToken, ensureGrootUser } from '../../../middleware/grootAuth.js';

/**
 * Groot.com Authentication Endpoint
 * POST /api/auth/groot-login
 * 
 * This endpoint simulates authentication from groot.com
 * In production, this would be called by groot.com's auth service
 */
export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({
            success: false,
            message: 'Method not allowed'
        });
    }

    try {
        await dbConnect();

        const {
            userId,
            email,
            name,
            role = 'admin'
        } = req.body;

        // Validate required fields
        if (!userId || !email || !name) {
            return res.status(400).json({
                success: false,
                message: 'userId, email, and name are required'
            });
        }

        // Validate groot.com email domain
        if (!email.endsWith('@groot.com')) {
            return res.status(400).json({
                success: false,
                message: 'Only groot.com email addresses are allowed'
            });
        }

        // Create groot user data
        const grootUserData = {
            userId,
            email,
            name,
            role
        };

        // Generate groot token
        const token = generateGrootToken(grootUserData);

        // Ensure user exists in our system
        const user = await ensureGrootUser(grootUserData);

        res.status(200).json({
            success: true,
            message: 'Groot authentication successful',
            data: {
                token,
                user: {
                    id: user._id,
                    email: user.email,
                    name: user.name,
                    role: user.role,
                    isGrootUser: true,
                    domain: 'groot.com'
                },
                expiresIn: '24h'
            }
        });

    } catch (error) {
        console.error('Groot login error:', error);
        res.status(500).json({
            success: false,
            message: 'Groot authentication failed',
            error: error.message
        });
    }
}