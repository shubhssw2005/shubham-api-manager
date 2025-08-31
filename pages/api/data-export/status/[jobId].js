import dbConnect from '../../../../lib/dbConnect.js';
import { requireApprovedUserWithGroot } from '../../../../middleware/grootAuth.js';
import DataExportService from '../../../../services/DataExportService.js';

/**
 * Get Data Export Job Status
 * GET /api/data-export/status/[jobId]
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

        const { jobId } = req.query;

        if (!jobId) {
            return res.status(400).json({
                success: false,
                message: 'Job ID is required'
            });
        }

        const exportService = new DataExportService();
        const jobStatus = await exportService.getJobStatus(jobId);

        // Check permissions - users can only view their own export jobs
        // unless they're admin or groot user
        if (jobStatus.userId?.toString() !== user._id.toString() && 
            user.role !== 'admin' && 
            !user.isGrootUser) {
            return res.status(403).json({
                success: false,
                message: 'Access denied'
            });
        }

        res.status(200).json({
            success: true,
            data: jobStatus
        });

    } catch (error) {
        console.error('Export status error:', error);
        
        if (error.message === 'Export job not found') {
            return res.status(404).json({
                success: false,
                message: 'Export job not found'
            });
        }

        res.status(500).json({
            success: false,
            message: 'Failed to get export status',
            error: error.message
        });
    }
}