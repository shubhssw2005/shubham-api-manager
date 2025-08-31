import { requireApprovedUser } from '../../middleware/auth.js';

/**
 * Download Backup API (for local development)
 */
export default async function handler(req, res) {
    try {
        const user = await requireApprovedUser(req, res);
        if (!user) return;

        if (req.method !== 'GET') {
            return res.status(405).json({
                success: false,
                message: 'Method not allowed'
            });
        }

        const { key } = req.query;
        
        if (!key) {
            return res.status(400).json({
                success: false,
                message: 'Backup key is required'
            });
        }

        // In local mode, try to serve the file
        try {
            const fs = await import('fs/promises');
            const path = await import('path');
            
            const localBackupDir = './local-backups';
            const localFilePath = path.join(localBackupDir, key.replace(/\//g, '_'));
            
            const fileBuffer = await fs.readFile(localFilePath);
            
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Disposition', `attachment; filename="blog-backup-${Date.now()}.json"`);
            res.setHeader('Content-Length', fileBuffer.length);
            
            res.status(200).send(fileBuffer);
            
        } catch (error) {
            console.error('Download error:', error);
            res.status(404).json({
                success: false,
                message: 'Backup file not found',
                error: error.message
            });
        }
        
    } catch (error) {
        console.error('Download API error:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error',
            error: error.message
        });
    }
}