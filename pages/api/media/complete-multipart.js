import PresignedURLService from '../../../services/PresignedURLService.js';
import { verifyToken } from '../../../lib/jwt.js';

/**
 * API endpoint for completing multipart uploads
 * POST /api/media/complete-multipart
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
    const { s3Key, uploadId, parts } = req.body;
    
    if (!s3Key || !uploadId || !parts || !Array.isArray(parts)) {
      return res.status(400).json({
        error: 'Bad Request',
        message: 's3Key, uploadId, and parts array are required'
      });
    }

    // Validate parts structure
    for (const part of parts) {
      if (!part.PartNumber || !part.ETag) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Each part must have PartNumber and ETag'
        });
      }
    }

    // Initialize presigned URL service
    const presignedService = new PresignedURLService();
    
    // Complete multipart upload
    const result = await presignedService.completeMultipartUpload({
      s3Key,
      uploadId,
      parts,
      tenantId
    });

    // Log the completion for monitoring
    console.log(`✅ Multipart upload completed for ${s3Key} with ${parts.length} parts`);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('❌ Error completing multipart upload:', error);
    
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
      message: 'Failed to complete multipart upload'
    });
  }
}