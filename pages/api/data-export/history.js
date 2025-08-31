import dbConnect from '../../../lib/dbConnect.js';
import { requireApprovedUserWithGroot } from '../../../middleware/grootAuth.js';
import DataExportService from '../../../services/DataExportService.js';

/**
 * Get User's Data Export History
 * GET /api/data-export/history
 */
export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({
            success: false,
            message: 'Method not allowed'
        });
    }

    try {
        await dbConnect();

        // Authenticate user (supports groot.com)
        const user = await requireApprovedUserWithGroot(req, res);
        if (!user) return;

        const { userId, limit = 10 } = req.query;

        // If no userId specified, get current user's history
        const targetUserId = userId || user._id;

        // Check permissions
        if (targetUserId.toString() !== user._id.toString() && 
            user.role !== 'admin' && 
            !user.isGrootUser) {
            return res.status(403).json({
                success: false,
                message: 'You can only view your own export history'
            });
        }

        const exportService = new DataExportService();
        const history = await exportService.getUserExportHistory(targetUserId, parseInt(limit));

        res.status(200).json({
            success: true,
            data: {
                exports: history,
                total: history.length
            }
        });

    } catch (error) {
        console.error('Export history error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to get export history',
            error: error.message
        });
    }
}