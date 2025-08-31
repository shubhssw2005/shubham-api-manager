import dbConnect from '../../../../lib/dbConnect';
import MediaFolder from '../../../../models/MediaFolder';
import Media from '../../../../models/Media';
import { requireApprovedUser } from '../../../../middleware/auth';
import { asyncHandler } from '../../../../lib/errorHandler';
import { ValidationError, ConflictError } from '../../../../lib/errors';

const foldersHandler = async (req, res) => {
  // Authenticate user
  const user = await requireApprovedUser(req, res);
  if (!user) return; // Error already handled by middleware

  await dbConnect();

  switch (req.method) {
    case 'GET':
      return await getFolders(req, res, user);
    case 'POST':
      return await createFolder(req, res, user);
    default:
      return res.status(405).json({
        success: false,
        error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
      });
  }
};

const getFolders = async (req, res, user) => {
  try {
    const {
      parent,
      tree = 'false',
      includeStats = 'false',
      search
    } = req.query;

    let folders;

    if (search) {
      // Search folders by name and description
      folders = await MediaFolder.searchFolders(search, user._id)
        .populate('parent', 'name path')
        .populate('createdBy', 'username email')
        .sort({ score: { $meta: 'textScore' } });
    } else if (tree === 'true') {
      // Return folder tree structure
      folders = await MediaFolder.buildFolderTree(parent || null, user._id);
    } else {
      // Get folders by parent
      const query = {};
      
      if (parent === 'root' || parent === '' || parent === undefined) {
        query.parent = null;
      } else {
        query.parent = parent;
      }

      // Add user access filter
      query.$or = [
        { isPublic: true },
        { createdBy: user._id },
        { 'permissions.read': user._id },
        { 'permissions.write': user._id },
        { 'permissions.admin': user._id }
      ];

      folders = await MediaFolder.find(query)
        .populate('parent', 'name path')
        .populate('createdBy', 'username email')
        .sort({ name: 1 });
    }

    // Include statistics if requested
    if (includeStats === 'true' && Array.isArray(folders)) {
      for (const folder of folders) {
        if (folder._id) {
          const [mediaCount, subfolderCount, totalSize] = await Promise.all([
            folder.getMediaCount(),
            folder.getSubfolderCount(),
            folder.getTotalSize()
          ]);

          folder._doc = folder._doc || folder;
          folder._doc.stats = {
            mediaCount,
            subfolderCount,
            totalSize,
            formattedSize: formatFileSize(totalSize)
          };
        }
      }
    }

    const response = {
      success: true,
      data: {
        folders: Array.isArray(folders) ? folders : [folders].filter(Boolean),
        isTree: tree === 'true',
        parent: parent || null
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Get folders error:', error);
    return res.status(500).json({
      success: false,
      error: { message: 'Failed to retrieve folders', code: 'RETRIEVAL_ERROR' }
    });
  }
};

const createFolder = async (req, res, user) => {
  try {
    const {
      name,
      description = '',
      parent = null,
      isPublic = false,
      color = '#3498db',
      icon = 'folder'
    } = req.body;

    // Validate required fields
    if (!name || typeof name !== 'string' || name.trim().length === 0) {
      throw new ValidationError('Folder name is required');
    }

    if (name.length > 100) {
      throw new ValidationError('Folder name must be 100 characters or less');
    }

    // Validate parent folder if provided
    let parentFolder = null;
    if (parent) {
      parentFolder = await MediaFolder.findById(parent);
      if (!parentFolder) {
        throw new ValidationError('Parent folder not found');
      }

      // Check if user has write access to parent folder
      if (!parentFolder.canUserAccess(user._id, 'write')) {
        throw new ValidationError('No write access to parent folder');
      }

      // Check nesting level
      if (parentFolder.level >= 9) { // Max 10 levels (0-9)
        throw new ValidationError('Maximum folder nesting level exceeded');
      }
    }

    // Generate slug from name
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');

    // Check for duplicate slug at the same level
    const duplicateQuery = { slug, parent: parent || null };
    const existingFolder = await MediaFolder.findOne(duplicateQuery);
    if (existingFolder) {
      throw new ConflictError('A folder with this name already exists at this location');
    }

    // Create folder data
    const folderData = {
      name: name.trim(),
      slug,
      description: description.trim(),
      parent: parent || null,
      isPublic,
      color,
      icon,
      createdBy: user._id,
      permissions: {
        read: [],
        write: [],
        admin: [user._id] // Creator gets admin access
      }
    };

    // Create the folder
    const folder = new MediaFolder(folderData);
    await folder.save();

    // Populate references for response
    await folder.populate('parent', 'name path');
    await folder.populate('createdBy', 'username email');

    const response = {
      success: true,
      data: {
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
        createdAt: folder.createdAt
      },
      message: 'Folder created successfully'
    };

    res.status(201).json(response);

  } catch (error) {
    console.error('Create folder error:', error);
    
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
      error: { message: 'Failed to create folder', code: 'CREATE_ERROR' }
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

export default asyncHandler(foldersHandler);