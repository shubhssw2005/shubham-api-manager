import dbConnect from '../../../lib/dbConnect.js';
import { requireApprovedUserWithGroot } from '../../../middleware/grootAuth.js';
import DataExportService from '../../../services/DataExportService.js';

/**
 * Request User Data Export
 * POST /api/data-export/request
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

        // Authenticate user (supports groot.com)
        const user = await requireApprovedUserWithGroot(req, res);
        if (!user) return;

        const {
            userId,
            exportType = 'full',
            includeMedia = true,
            includeDeleted = false
        } = req.body;

        // If no userId specified, export current user's data
        const targetUserId = userId || user._id;

        // Check permissions - only allow users to export their own data
        // unless they're admin or groot user
        if (targetUserId.toString() !== user._id.toString() && 
            user.role !== 'admin' && 
            !user.isGrootUser) {
            return res.status(403).json({
                success: false,
                message: 'You can only export your own data'
            });
        }

        const exportService = new DataExportService();
        const result = await exportService.requestExport(targetUserId, user._id, {
            exportType,
            includeMedia,
            includeDeleted
        });

        if (!result.success) {
            return res.status(409).json(result);
        }

        res.status(202).json({
            success: true,
            message: 'Data export requested successfully',
            data: {
                jobId: result.jobId,
                estimatedTime: result.estimatedTime,
                status: 'pending',
                checkStatusUrl: `/api/data-export/status/${result.jobId}`
            }
        });

    } catch (error) {
        console.error('Data export request error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to request data export',
            error: error.message
        });
    }
}