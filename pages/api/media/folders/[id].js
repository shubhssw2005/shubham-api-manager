import dbConnect from '../../../../lib/dbConnect';
import MediaFolder from '../../../../models/MediaFolder';
import Media from '../../../../models/Media';
import { requireApprovedUser } from '../../../../middleware/auth';
import { asyncHandler } from '../../../../lib/errorHandler';
import { ValidationError, NotFoundError, ConflictError } from '../../../../lib/errors';

const folderHandler = async (req, res) => {
  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  const { id } = req.query;

  // Validate folder ID
  if (!id || !id.match(/^[0-9a-fA-F]{24}$/)) {
    return res.status(400).json({
      success: false,
      error: { message: 'Invalid folder ID', code: 'VALIDATION_ERROR' }
    });
  }

  switch (req.method) {
    case 'GET':
      return await getFolder(req, res, user, id);
    case 'PUT':
      return await updateFolder(req, res, user, id);
    case 'DELETE':
      return await deleteFolder(req, res, user, id);
    default:
      return res.status(405).json({
        success: false,
        error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
      });
  }
};

const getFolder = async (req, res, user, folderId) => {
  try {
    const folder = await MediaFolder.findById(folderId)
      .populate('parent', 'name path')
      .populate('createdBy', 'username email');

    if (!folder) {
      throw new NotFoundError('Folder not found');
    }

    // Check if user has read access
    if (!folder.canUserAccess(user._id, 'read')) {
      throw new ValidationError('No read access to folder');
    }

    // Get folder statistics
    const [mediaCount, subfolderCount, totalSize] = await Promise.all([
      folder.getMediaCount(),
      folder.getSubfolderCount(),
      folder.getTotalSize()
    ]);

    const response = {
      success: true,
      data: {
        folder: {
          id: folder._id,
          name: folder.name,
          slug: folder.slug,
          description: folder.description,
          path: folder.path,
          level: folder.level,
          parent: folder.parent,
          isPublic: folder.isPublic,
          color: folder.color,
          icon: folder.icon,
          permissions: folder.permissions,
          createdBy: folder.createdBy,
          createdAt: folder.createdAt,
          updatedAt: folder.updatedAt,
          stats: {
            mediaCount,
            subfolderCount,
            totalSize,
            formattedSize: formatFileSize(totalSize)
          }
        }
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Get folder error:', error);
    
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
      error: { message: 'Failed to retrieve folder', code: 'RETRIEVAL_ERROR' }
    });
  }
};

const updateFolder = async (req, res, user, folderId) => {
  try {
    const {
      name,
      description,
      isPublic,
      color,
      icon
    } = req.body;

    const folder = await MediaFolder.findById(folderId);

    if (!folder) {
      throw new NotFoundError('Folder not found');
    }

    // Check if user has admin access
    if (!folder.canUserAccess(user._id, 'admin')) {
      throw new ValidationError('No admin access to folder');
    }

    // Validate name if provided
    if (name !== undefined) {
      if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new ValidationError('Folder name is required');
      }

      if (name.length > 100) {
        throw new ValidationError('Folder name must be 100 characters or less');
      }

      // Check for duplicate name at the same level (only if name changed)
      if (name.trim() !== folder.name) {
        const slug = name
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '');

        const duplicateQuery = { 
          slug, 
          parent: folder.parent,
          _id: { $ne: folderId }
        };
        
        const existingFolder = await MediaFolder.findOne(duplicateQuery);
        if (existingFolder) {
          throw new ConflictError('A folder with this name already exists at this location');
        }

        folder.name = name.trim();
        folder.slug = slug;
      }
    }

    // Update other fields
    if (description !== undefined) {
      folder.description = description.trim();
    }

    if (isPublic !== undefined) {
      folder.isPublic = Boolean(isPublic);
    }

    if (color !== undefined) {
      if (color && !/^#[0-9A-F]{6}$/i.test(color)) {
        throw new ValidationError('Invalid color format');
      }
      folder.color = color || '#3498db';
    }

    if (icon !== undefined) {
      folder.icon = icon || 'folder';
    }

    // Save the updated folder
    await folder.save();

    // Populate references for response
    await folder.populate('parent', 'name path');
    await folder.populate('createdBy', 'username email');

    const response = {
      success: true,
      data: {
        folder: {
          id: folder._id,
          name: folder.name,
          slug: folder.slug,
          description: folder.description,
          path: folder.path,
          level: folder.level,
          parent: folder.parent,
          isPublic: folder.isPublic,
          color: folder.color,
          icon: folder.icon,
          permissions: folder.permissions,
          createdBy: folder.createdBy,
          createdAt: folder.createdAt,
          updatedAt: folder.updatedAt
        }
      },
      message: 'Folder updated successfully'
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Update folder error:', error);
    
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
    
    if (error instanceof ConflictError) {
      return res.status(409).json({
        success: false,
        error: { message: error.message, code: 'CONFLICT' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to update folder', code: 'UPDATE_ERROR' }
    });
  }
};

const deleteFolder = async (req, res, user, folderId) => {
  try {
    const { force = false } = req.query;

    const folder = await MediaFolder.findById(folderId);

    if (!folder) {
      throw new NotFoundError('Folder not found');
    }

    // Check if user has admin access
    if (!folder.canUserAccess(user._id, 'admin')) {
      throw new ValidationError('No admin access to folder');
    }

    // Check if folder has subfolders
    const subfolderCount = await MediaFolder.countDocuments({ parent: folderId });
    if (subfolderCount > 0 && !force) {
      throw new ConflictError('Cannot delete folder that contains subfolders. Use force=true to delete recursively.');
    }

    // Check if folder has media files
    const mediaCount = await Media.countDocuments({ folder: folderId });
    if (mediaCount > 0 && !force) {
      throw new ConflictError('Cannot delete folder that contains media files. Use force=true to move files to root.');
    }

    // If force delete, handle subfolders and media files
    if (force) {
      // Move all media files to root folder
      await Media.updateMany(
        { folder: folderId },
        { folder: null }
      );

      // Recursively delete subfolders
      const subfolders = await MediaFolder.find({ parent: folderId });
      for (const subfolder of subfolders) {
        await deleteFolder(req, res, user, subfolder._id.toString());
      }
    }

    // Delete the folder
    await MediaFolder.findByIdAndDelete(folderId);

    const response = {
      success: true,
      data: {
        deletedFolder: {
          id: folder._id,
          name: folder.name,
          path: folder.path
        },
        movedMediaFiles: force ? mediaCount : 0,
        deletedSubfolders: force ? subfolderCount : 0
      },
      message: 'Folder deleted successfully'
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Delete folder error:', error);
    
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
        error: { message: error.message, code: 'CONFLICT' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Failed to delete folder', code: 'DELETE_ERROR' }
    });
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

export default asyncHandler(folderHandler);