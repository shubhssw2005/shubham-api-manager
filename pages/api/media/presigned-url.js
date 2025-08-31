import PresignedURLService from '../../../services/PresignedURLService.js';
import { verifyToken } from '../../../lib/jwt.js';

/**
 * API endpoint for generating presigned URLs for direct S3 uploads
 * POST /api/media/presigned-url
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

    let decoded;
    try {
      decoded = verifyToken(token);
    } catch (jwtError) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Invalid authentication token'
      });
    }

    const userId = decoded.userId || decoded.id;
    const tenantId = decoded.tenantId || 'default';

    // Validate request body
    const { originalName, contentType, size, metadata } = req.body;
    
    if (!originalName || !contentType || !size) {
      return res.status(400).json({
        error: 'Bad Request',
        message: 'originalName, contentType, and size are required'
      });
    }

    // Initialize presigned URL service
    const presignedService = new PresignedURLService();
    
    // Generate presigned URL
    const result = await presignedService.generateUploadURL({
      tenantId,
      originalName,
      contentType,
      size: parseInt(size),
      userId,
      metadata: metadata || {}
    });

    // Log the upload request for monitoring
    console.log(`📤 Presigned URL generated for ${originalName} (${contentType}, ${size} bytes) by user ${userId} in tenant ${tenantId}`);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('❌ Error generating presigned URL:', error);
    
    // Return appropriate error response
    if (error.message.includes('not allowed') || error.message.includes('Invalid')) {
      return res.status(400).json({
        error: 'Bad Request',
        message: error.message
      });
    }
    
    if (error.message.includes('Unauthorized')) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: error.message
      });
    }
    
    res.status(500).json({
      error: 'Internal Server Error',
      message: 'Failed to generate presigned URL'
    });
  }
}