import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import MediaFolder from '../../../models/MediaFolder';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler } from '../../../lib/errorHandler';
import { ValidationError, NotFoundError, ConflictError } from '../../../lib/errors';
import { StorageFactory } from '../../../lib/storage';

const mediaItemHandler = async (req, res) => {
  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  const { id } = req.query;

  // Validate media ID
  if (!id || !id.match(/^[0-9a-fA-F]{24}$/)) {
    return res.status(400).json({
      success: false,
      error: { message: 'Invalid media ID', code: 'INVALID_ID' }
    });
  }

  switch (req.method) {
    case 'GET':
      return await getMediaFile(req, res, user, id);
    case 'PUT':
      return await updateMediaFile(req, res, user, id);
    case 'DELETE':
      return await deleteMediaFile(req, res, user, id);
    default:
      return res.status(405).json({
        success: false,
        error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
      });
  }
};

const getMediaFile = async (req, res, user, id) => {
  try {
    const media = await Media.findById(id)
      .populate('folder', 'name path')
      .populate('uploadedBy', 'username email')
      .populate('usage.documentId');

    if (!media) {
      throw new NotFoundError('Media file not found');
    }

    // Check if user has access to the media file
    // Users can access their own files or files in folders they have access to
    const hasAccess = media.uploadedBy._id.toString() === user._id.toString() ||
                     user.role === 'admin' ||
                     (media.folder && await checkFolderAccess(media.folder._id, user._id, 'read')) ||
                     media.isPublic;

    if (!hasAccess) {
      throw new ValidationError('No access to this media file');
    }

    const response = {
      success: true,
      data: {
        id: media._id,
        filename: media.filename,
        originalName: media.originalName,
        mimeType: media.mimeType,
        size: media.size,
        formattedSize: formatFileSize(media.size),
        url: media.url,
        storageProvider: media.storageProvider,
        thumbnails: media.thumbnails,
        metadata: media.metadata,
        folder: media.folder,
        tags: media.tags,
        alt: media.alt,
        caption: media.caption,
        description: media.description,
        usage: media.usage,
        usageCount: media.usage.length,
        isPublic: media.isPublic,
        uploadedBy: media.uploadedBy,
        processingStatus: media.processingStatus,
        processingError: media.processingError,
        createdAt: media.createdAt,
        updatedAt: media.updatedAt
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Get media file error:', error);
    
    if (error instanceof NotFoundError) {
      return res.status(404).json({
        success: false,
        error: { message: error.message, code: 'NOT_FOUND' }
      });
    }
    
    if (error instanceof ValidationError) {
      return res.status(403).json({
        success: false,
        error: { message: error.message, code: 'ACCESS_DENIED' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to retrieve media file', code: 'RETRIEVAL_ERROR' }
    });
  }
};

const updateMediaFile = async (req, res, user, id) => {
  try {
    const media = await Media.findById(id);

    if (!media) {
      throw new NotFoundError('Media file not found');
    }

    // Check if user has write access
    const hasWriteAccess = media.uploadedBy.toString() === user._id.toString() ||
                          user.role === 'admin' ||
                          (media.folder && await checkFolderAccess(media.folder, user._id, 'write'));

    if (!hasWriteAccess) {
      throw new ValidationError('No write access to this media file');
    }

    // Extract updatable fields from request body
    const {
      alt,
      caption,
      description,
      tags,
      isPublic,
      folder
    } = req.body;

    const updateData = {};

    // Update metadata fields
    if (alt !== undefined) updateData.alt = alt;
    if (caption !== undefined) updateData.caption = caption;
    if (description !== undefined) updateData.description = description;
    if (isPublic !== undefined) updateData.isPublic = isPublic;

    // Update tags
    if (tags !== undefined) {
      if (Array.isArray(tags)) {
        updateData.tags = tags.map(tag => tag.toLowerCase().trim()).filter(tag => tag);
      } else {
        throw new ValidationError('Tags must be an array');
      }
    }

    // Update folder
    if (folder !== undefined) {
      if (folder === null || folder === '') {
        updateData.folder = null;
      } else {
        // Validate folder exists and user has write access
        const folderDoc = await MediaFolder.findById(folder);
        if (!folderDoc) {
          throw new ValidationError('Folder not found');
        }
        if (!folderDoc.canUserAccess(user._id, 'write')) {
          throw new ValidationError('No write access to target folder');
        }
        updateData.folder = folder;
      }
    }

    // Update the media file
    const updatedMedia = await Media.findByIdAndUpdate(
      id,
      updateData,
      { new: true, runValidators: true }
    ).populate('folder', 'name path')
     .populate('uploadedBy', 'username email');

    const response = {
      success: true,
      data: {
        id: updatedMedia._id,
        filename: updatedMedia.filename,
        originalName: updatedMedia.originalName,
        mimeType: updatedMedia.mimeType,
        size: updatedMedia.size,
        url: updatedMedia.url,
        thumbnails: updatedMedia.thumbnails,
        metadata: updatedMedia.metadata,
        folder: updatedMedia.folder,
        tags: updatedMedia.tags,
        alt: updatedMedia.alt,
        caption: updatedMedia.caption,
        description: updatedMedia.description,
        isPublic: updatedMedia.isPublic,
        uploadedBy: updatedMedia.uploadedBy,
        updatedAt: updatedMedia.updatedAt
      },
      message: 'Media file updated successfully'
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Update media file error:', error);
    
    if (error instanceof NotFoundError) {
      return res.status(404).json({
        success: false,
        error: { message: error.message, code: 'NOT_FOUND' }
      });
    }
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to update media file', code: 'UPDATE_ERROR' }
    });
  }
};

const deleteMediaFile = async (req, res, user, id) => {
  try {
    const media = await Media.findById(id);

    if (!media) {
      throw new NotFoundError('Media file not found');
    }

    // Check if user has delete access
    const hasDeleteAccess = media.uploadedBy.toString() === user._id.toString() ||
                           user.role === 'admin' ||
                           (media.folder && await checkFolderAccess(media.folder, user._id, 'admin'));

    if (!hasDeleteAccess) {
      throw new ValidationError('No delete access to this media file');
    }

    // Check if file is being used
    if (media.usage && media.usage.length > 0) {
      const { force } = req.query;
      
      if (!force || force !== 'true') {
        throw new ConflictError(
          `Media file is being used in ${media.usage.length} location(s). Use force=true to delete anyway.`,
          { usage: media.usage }
        );
      }
    }

    // Initialize storage provider
    const storageProvider = StorageFactory.createFromEnv();

    try {
      // Delete main file from storage
      await storageProvider.delete(media.path);

      // Delete thumbnails from storage
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
      // Continue with database deletion even if storage deletion fails
    }

    // Delete from database
    await Media.findByIdAndDelete(id);

    const response = {
      success: true,
      message: 'Media file deleted successfully',
      data: {
        id: media._id,
        filename: media.filename,
        originalName: media.originalName
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Delete media file error:', error);
    
    if (error instanceof NotFoundError) {
      return res.status(404).json({
        success: false,
        error: { message: error.message, code: 'NOT_FOUND' }
      });
    }
    
    if (error instanceof ValidationError) {
      return res.status(403).json({
        success: false,
        error: { message: error.message, code: 'ACCESS_DENIED' }
      });
    }
    
    if (error instanceof ConflictError) {
      return res.status(409).json({
        success: false,
        error: { 
          message: error.message, 
          code: 'CONFLICT',
          details: error.details
        }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to delete media file', code: 'DELETE_ERROR' }
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

// Helper function to format file size
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export default asyncHandler(mediaItemHandler);