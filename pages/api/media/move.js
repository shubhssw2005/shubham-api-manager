import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import MediaFolder from '../../../models/MediaFolder';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler } from '../../../lib/errorHandler';
import { ValidationError, NotFoundError } from '../../../lib/errors';

const moveMediaHandler = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({
      success: false,
      error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
    });
  }

  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  try {
    const { mediaIds, targetFolderId } = req.body;

    // Validate request
    if (!mediaIds || !Array.isArray(mediaIds) || mediaIds.length === 0) {
      throw new ValidationError('Media IDs array is required');
    }

    // Validate media IDs format
    const invalidIds = mediaIds.filter(id => !id.match(/^[0-9a-fA-F]{24}$/));
    if (invalidIds.length > 0) {
      throw new ValidationError('Invalid media ID format');
    }

    // Validate target folder
    let targetFolder = null;
    if (targetFolderId && targetFolderId !== 'root') {
      targetFolder = await MediaFolder.findById(targetFolderId);
      if (!targetFolder) {
        throw new ValidationError('Target folder not found');
      }
      
      // Check if user has write access to target folder
      if (!targetFolder.canUserAccess(user._id, 'write')) {
        throw new ValidationError('No write access to target folder');
      }
    }

    // Find all media files
    const mediaFiles = await Media.find({ _id: { $in: mediaIds } });
    
    if (mediaFiles.length === 0) {
      throw new NotFoundError('No media files found');
    }

    const results = {
      moved: [],
      failed: []
    };

    for (const media of mediaFiles) {
      try {
        // Check if user has write access to the media file
        const hasWriteAccess = media.uploadedBy.toString() === user._id.toString() ||
                              user.role === 'admin' ||
                              (media.folder && await checkFolderAccess(media.folder, user._id, 'write'));

        if (!hasWriteAccess) {
          results.failed.push({
            id: media._id,
            filename: media.originalName,
            error: 'No write access to media file'
          });
          continue;
        }

        // Update folder
        const oldFolder = media.folder;
        await Media.findByIdAndUpdate(media._id, {
          folder: targetFolder ? targetFolder._id : null
        });

        results.moved.push({
          id: media._id,
          filename: media.originalName,
          fromFolder: oldFolder,
          toFolder: targetFolder ? {
            id: targetFolder._id,
            name: targetFolder.name,
            path: targetFolder.path
          } : null
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName || 'Unknown',
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Move operation completed: ${results.moved.length} files moved, ${results.failed.length} failed`,
      targetFolder: targetFolder ? {
        id: targetFolder._id,
        name: targetFolder.name,
        path: targetFolder.path
      } : null
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Move media error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }
    
    if (error instanceof NotFoundError) {
      return res.status(404).json({
        success: false,
        error: { message: error.message, code: 'NOT_FOUND' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Move operation failed', code: 'MOVE_ERROR' }
    });
  }
};

// Helper function to check folder access
const checkFolderAccess = async (folderId, userId, permission) => {
  try {
    const folder = await MediaFolder.findById(folderId);
    return folder ? folder.canUserAccess(userId, permission) : false;
  } catch (error) {
    return false;
  }
};

export default asyncHandler(moveMediaHandler);