import PresignedURLService from '../../../services/PresignedURLService.js';
import { verifyToken } from '../../../lib/jwt.js';

/**
 * API endpoint for creating resumable upload sessions
 * POST /api/media/resumable-session
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
    
    // Create resumable session
    const result = await presignedService.createResumableSession({
      tenantId,
      originalName,
      contentType,
      size: parseInt(size),
      userId,
      metadata: metadata || {}
    });

    // Log the session creation for monitoring
    console.log(`🔄 Resumable session created for ${originalName} (${contentType}, ${size} bytes) by user ${userId} in tenant ${tenantId}`);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('❌ Error creating resumable session:', error);
    
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
      message: 'Failed to create resumable session'
    });
  }
}