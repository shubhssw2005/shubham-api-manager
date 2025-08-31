import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import MediaFolder from '../../../models/MediaFolder';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler } from '../../../lib/errorHandler';
import { ValidationError, NotFoundError, ConflictError } from '../../../lib/errors';
import { StorageFactory } from '../../../lib/storage';

const bulkMediaHandler = async (req, res) => {
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

  const { action, mediaIds, data } = req.body;

  // Validate request
  if (!action) {
    return res.status(400).json({
      success: false,
      error: { message: 'Action is required', code: 'VALIDATION_ERROR' }
    });
  }

  if (!mediaIds || !Array.isArray(mediaIds) || mediaIds.length === 0) {
    return res.status(400).json({
      success: false,
      error: { message: 'Media IDs array is required', code: 'VALIDATION_ERROR' }
    });
  }

  // Validate media IDs format
  const invalidIds = mediaIds.filter(id => !id.match(/^[0-9a-fA-F]{24}$/));
  if (invalidIds.length > 0) {
    return res.status(400).json({
      success: false,
      error: { 
        message: 'Invalid media ID format', 
        code: 'VALIDATION_ERROR',
        details: { invalidIds }
      }
    });
  }

  switch (action) {
    case 'delete':
      return await bulkDeleteMedia(req, res, user, mediaIds, data);
    case 'move':
      return await bulkMoveMedia(req, res, user, mediaIds, data);
    case 'update':
      return await bulkUpdateMedia(req, res, user, mediaIds, data);
    case 'addTags':
      return await bulkAddTags(req, res, user, mediaIds, data);
    case 'removeTags':
      return await bulkRemoveTags(req, res, user, mediaIds, data);
    default:
      return res.status(400).json({
        success: false,
        error: { message: 'Invalid action', code: 'VALIDATION_ERROR' }
      });
  }
};

const bulkDeleteMedia = async (req, res, user, mediaIds, data) => {
  try {
    const { force = false } = data || {};

    // Find all media files
    const mediaFiles = await Media.find({ _id: { $in: mediaIds } });
    
    if (mediaFiles.length === 0) {
      throw new NotFoundError('No media files found');
    }

    const results = {
      deleted: [],
      failed: [],
      skipped: []
    };

    const storageProvider = StorageFactory.createFromEnv();

    for (const media of mediaFiles) {
      try {
        // Check if user has delete access
        const hasDeleteAccess = media.uploadedBy.toString() === user._id.toString() ||
                               user.role === 'admin' ||
                               (media.folder && await checkFolderAccess(media.folder, user._id, 'admin'));

        if (!hasDeleteAccess) {
          results.failed.push({
            id: media._id,
            filename: media.originalName,
            error: 'No delete access'
          });
          continue;
        }

        // Check if file is being used
        if (media.usage && media.usage.length > 0 && !force) {
          results.skipped.push({
            id: media._id,
            filename: media.originalName,
            reason: `File is being used in ${media.usage.length} location(s)`
          });
          continue;
        }

        // Delete from storage
        try {
          await storageProvider.delete(media.path);
          
          // Delete thumbnails
          if (media.thumbnails && media.thumbnails.length > 0) {
            for (const thumbnail of media.thumbnails) {
              try {
                await storageProvider.delete(thumbnail.path);
              } catch (thumbError) {
                console.warn(`Failed to delete thumbnail ${thumbnail.path}:`, thumbError);
              }
            }
          }
        } catch (storageError) {
          console.warn('Storage deletion error:', storageError);
        }

        // Delete from database
        await Media.findByIdAndDelete(media._id);

        results.deleted.push({
          id: media._id,
          filename: media.originalName
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName,
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Bulk delete completed: ${results.deleted.length} deleted, ${results.failed.length} failed, ${results.skipped.length} skipped`
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Bulk delete error:', error);
    return res.status(500).json({
      success: false,
      error: { message: 'Bulk delete failed', code: 'BULK_DELETE_ERROR' }
    });
  }
};

const bulkMoveMedia = async (req, res, user, mediaIds, data) => {
  try {
    const { folderId } = data || {};

    // Validate target folder
    let targetFolder = null;
    if (folderId && folderId !== 'root') {
      targetFolder = await MediaFolder.findById(folderId);
      if (!targetFolder) {
        throw new ValidationError('Target folder not found');
      }
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
            error: 'No write access'
          });
          continue;
        }

        // Update folder
        await Media.findByIdAndUpdate(media._id, {
          folder: targetFolder ? targetFolder._id : null
        });

        results.moved.push({
          id: media._id,
          filename: media.originalName,
          targetFolder: targetFolder ? targetFolder.name : 'Root'
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName,
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Bulk move completed: ${results.moved.length} moved, ${results.failed.length} failed`
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Bulk move error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Bulk move failed', code: 'BULK_MOVE_ERROR' }
    });
  }
};

const bulkUpdateMedia = async (req, res, user, mediaIds, data) => {
  try {
    const { updates } = data || {};

    if (!updates || typeof updates !== 'object') {
      throw new ValidationError('Updates object is required');
    }

    // Validate update fields
    const allowedFields = ['alt', 'caption', 'description', 'isPublic'];
    const updateData = {};
    
    for (const [key, value] of Object.entries(updates)) {
      if (allowedFields.includes(key)) {
        updateData[key] = value;
      }
    }

    if (Object.keys(updateData).length === 0) {
      throw new ValidationError('No valid update fields provided');
    }

    // Find all media files
    const mediaFiles = await Media.find({ _id: { $in: mediaIds } });
    
    if (mediaFiles.length === 0) {
      throw new NotFoundError('No media files found');
    }

    const results = {
      updated: [],
      failed: []
    };

    for (const media of mediaFiles) {
      try {
        // Check if user has write access
        const hasWriteAccess = media.uploadedBy.toString() === user._id.toString() ||
                              user.role === 'admin' ||
                              (media.folder && await checkFolderAccess(media.folder, user._id, 'write'));

        if (!hasWriteAccess) {
          results.failed.push({
            id: media._id,
            filename: media.originalName,
            error: 'No write access'
          });
          continue;
        }

        // Update media file
        await Media.findByIdAndUpdate(media._id, updateData, { runValidators: true });

        results.updated.push({
          id: media._id,
          filename: media.originalName
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName,
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Bulk update completed: ${results.updated.length} updated, ${results.failed.length} failed`
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Bulk update error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Bulk update failed', code: 'BULK_UPDATE_ERROR' }
    });
  }
};

const bulkAddTags = async (req, res, user, mediaIds, data) => {
  try {
    const { tags } = data || {};

    if (!tags || !Array.isArray(tags) || tags.length === 0) {
      throw new ValidationError('Tags array is required');
    }

    const normalizedTags = tags.map(tag => tag.toLowerCase().trim()).filter(tag => tag);

    // Find all media files
    const mediaFiles = await Media.find({ _id: { $in: mediaIds } });
    
    if (mediaFiles.length === 0) {
      throw new NotFoundError('No media files found');
    }

    const results = {
      updated: [],
      failed: []
    };

    for (const media of mediaFiles) {
      try {
        // Check if user has write access
        const hasWriteAccess = media.uploadedBy.toString() === user._id.toString() ||
                              user.role === 'admin' ||
                              (media.folder && await checkFolderAccess(media.folder, user._id, 'write'));

        if (!hasWriteAccess) {
          results.failed.push({
            id: media._id,
            filename: media.originalName,
            error: 'No write access'
          });
          continue;
        }

        // Add tags (avoid duplicates)
        const existingTags = media.tags || [];
        const newTags = [...new Set([...existingTags, ...normalizedTags])];

        await Media.findByIdAndUpdate(media._id, { tags: newTags });

        results.updated.push({
          id: media._id,
          filename: media.originalName,
          addedTags: normalizedTags.filter(tag => !existingTags.includes(tag))
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName,
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Bulk add tags completed: ${results.updated.length} updated, ${results.failed.length} failed`
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Bulk add tags error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Bulk add tags failed', code: 'BULK_ADD_TAGS_ERROR' }
    });
  }
};

const bulkRemoveTags = async (req, res, user, mediaIds, data) => {
  try {
    const { tags } = data || {};

    if (!tags || !Array.isArray(tags) || tags.length === 0) {
      throw new ValidationError('Tags array is required');
    }

    const normalizedTags = tags.map(tag => tag.toLowerCase().trim()).filter(tag => tag);

    // Find all media files
    const mediaFiles = await Media.find({ _id: { $in: mediaIds } });
    
    if (mediaFiles.length === 0) {
      throw new NotFoundError('No media files found');
    }

    const results = {
      updated: [],
      failed: []
    };

    for (const media of mediaFiles) {
      try {
        // Check if user has write access
        const hasWriteAccess = media.uploadedBy.toString() === user._id.toString() ||
                              user.role === 'admin' ||
                              (media.folder && await checkFolderAccess(media.folder, user._id, 'write'));

        if (!hasWriteAccess) {
          results.failed.push({
            id: media._id,
            filename: media.originalName,
            error: 'No write access'
          });
          continue;
        }

        // Remove tags
        const existingTags = media.tags || [];
        const newTags = existingTags.filter(tag => !normalizedTags.includes(tag));

        await Media.findByIdAndUpdate(media._id, { tags: newTags });

        results.updated.push({
          id: media._id,
          filename: media.originalName,
          removedTags: normalizedTags.filter(tag => existingTags.includes(tag))
        });

      } catch (error) {
        results.failed.push({
          id: media._id,
          filename: media.originalName,
          error: error.message
        });
      }
    }

    const response = {
      success: true,
      data: results,
      message: `Bulk remove tags completed: ${results.updated.length} updated, ${results.failed.length} failed`
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Bulk remove tags error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Bulk remove tags failed', code: 'BULK_REMOVE_TAGS_ERROR' }
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

export default asyncHandler(bulkMediaHandler);