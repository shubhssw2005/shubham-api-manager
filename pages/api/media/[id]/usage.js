import dbConnect from '../../../../lib/dbConnect';
import Media from '../../../../models/Media';
import { requireApprovedUser } from '../../../../middleware/auth';
import { asyncHandler } from '../../../../lib/errorHandler';
import { NotFoundError } from '../../../../lib/errors';

const usageHandler = async (req, res) => {
  if (req.method !== 'GET') {
    return res.status(405).json({
      success: false,
      error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
    });
  }

  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  const { id } = req.query;

  try {
    // Find the media file
    const media = await Media.findById(id);
    
    if (!media) {
      throw new NotFoundError('Media file not found');
    }

    // Return usage information
    const response = {
      success: true,
      data: {
        mediaId: media._id,
        filename: media.originalName,
        usage: media.usage || [],
        usageCount: media.usage ? media.usage.length : 0
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Get media usage error:', error);
    
    if (error instanceof NotFoundError) {
      return res.status(404).json({
        success: false,
        error: { message: error.message, code: 'NOT_FOUND' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to retrieve media usage', code: 'RETRIEVAL_ERROR' }
    });
  }
};

export default asyncHandler(usageHandler);