import PresignedURLService from '../../../services/PresignedURLService.js';
import { verifyToken } from '../../../lib/jwt.js';

/**
 * API endpoint for aborting multipart uploads
 * POST /api/media/abort-multipart
 */
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ 
      error: 'Method not allowed',
      message: 'Only POST requests are supported'
    });
  }

  try {
    // Verify authentication
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ 
        error: 'Unauthorized',
        message: 'Authentication token is required'
      });
    }

    const decoded = verifyToken(token);
    const tenantId = decoded.tenantId || 'default';

    // Validate request body
    const { s3Key, uploadId } = req.body;
    
    if (!s3Key || !uploadId) {
      return res.status(400).json({
        error: 'Bad Request',
        message: 's3Key and uploadId are required'
      });
    }

    // Initialize presigned URL service
    const presignedService = new PresignedURLService();
    
    // Abort multipart upload
    const result = await presignedService.abortMultipartUpload({
      s3Key,
      uploadId,
      tenantId
    });

    // Log the abortion for monitoring
    console.log(`🚫 Multipart upload aborted for ${s3Key}`);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('❌ Error aborting multipart upload:', error);
    
    // Return appropriate error response
    if (error.message.includes('Unauthorized')) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: error.message
      });
    }
    
    if (error.message.includes('not found') || error.message.includes('Invalid')) {
      return res.status(400).json({
        error: 'Bad Request',
        message: error.message
      });
    }
    
    res.status(500).json({
      error: 'Internal Server Error',
      message: 'Failed to abort multipart upload'
    });
  }
}