import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import MediaFolder from '../../../models/MediaFolder';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler } from '../../../lib/errorHandler';
import { ValidationError, NotFoundError } from '../../../lib/errors';
import formidable from 'formidable';
import { promises as fs } from 'fs';
import path from 'path';

export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

const uploadDir = path.join(process.cwd(), 'uploads');

// Ensure upload directory exists
const ensureUploadDir = async () => {
  try {
    await fs.access(uploadDir);
  } catch {
    await fs.mkdir(uploadDir, { recursive: true });
  }
};

const mediaHandler = async (req, res) => {
  try {
    // Authenticate user
    const user = await requireApprovedUser(req, res);
    if (!user) return;

    await dbConnect();

    switch (req.method) {
      case 'GET':
        return await getMediaFiles(req, res, user);
      case 'POST':
        return await uploadMedia(req, res, user);
      default:
        return res.status(405).json({
          success: false,
          error: { message: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' }
        });
    }
  } catch (error) {
    console.error('Error in mediaHandler:', error);
    return res.status(500).json({
      success: false,
      error: {
        message: error.message || 'Internal server error',
        code: error.code || 'INTERNAL_ERROR'
      }
    });
  }
};

const uploadMedia = async (req, res, user) => {
  await ensureUploadDir();

  const form = formidable({
    uploadDir,
    keepExtensions: true,
    maxFileSize: 50 * 1024 * 1024, // 50MB
    filename: (name, ext, part) => {
      return `${Date.now()}-${part.originalFilename}`;
    }
  });

  try {
    const [fields, files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        else resolve([fields, files]);
      });
    });

    if (!files.file) {
      throw new ValidationError('No file uploaded');
    }

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    const metadata = fields.metadata ? JSON.parse(fields.metadata) : {};

    // Create media entry
    const media = await Media.create({
      title: metadata.title || file.originalFilename,
      description: metadata.description,
      filename: file.originalFilename,
      originalName: file.originalFilename,
      mimeType: file.mimetype,
      size: file.size,
      type: fields.type || determineFileType(file.mimetype),
      tags: metadata.tags || [],
      uploadedBy: user._id,
      folder: fields.folder || null,
      path: file.filepath,
      url: `/uploads/${path.basename(file.filepath)}`,
    });

    return res.status(201).json({
      success: true,
      data: media
    });

  } catch (error) {
    // Clean up any uploaded files on error
    if (error.httpCode === 413) {
      return res.status(413).json({
        success: false,
        error: {
          message: 'File too large',
          code: 'FILE_TOO_LARGE'
        }
      });
    }

    console.error('Error in uploadMedia:', error);
    return res.status(500).json({
      success: false,
      error: {
        message: error.message || 'Error uploading file',
        code: error.code || 'UPLOAD_ERROR'
      }
    });
  }
};

const determineFileType = (mimeType) => {
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('video/')) return 'video';
  if (mimeType.startsWith('audio/')) return 'audio';
  if (mimeType.includes('pdf') || mimeType.includes('document')) return 'document';
  return 'other';
};

const getMediaFiles = async (req, res, user) => {
  try {
    const {
      page = 1,
      limit = 20,
      folder,
      type,
      search,
      sortBy = 'createdAt',
      sortOrder = 'desc',
      tags,
      mimeType
    } = req.query;

    // Build query
    const query = {};

    // Filter by folder
    if (folder !== undefined) {
      if (folder === 'root' || folder === '') {
        query.folder = null;
      } else {
        const folderExists = await MediaFolder.findById(folder);
        if (!folderExists) {
          throw new NotFoundError('Folder not found');
        }
        query.folder = folder;
      }
    }

    // Filter by type
    if (type && type !== 'all') {
      query.type = type;
    }

    // Filter by mime type
    if (mimeType) {
      query.mimeType = mimeType;
    }

    // Filter by tags
    if (tags) {
      query.tags = { $in: Array.isArray(tags) ? tags : [tags] };
    }

    // Search by filename or title
    if (search) {
      query.$or = [
        { filename: { $regex: search, $options: 'i' } },
        { title: { $regex: search, $options: 'i' } },
        { description: { $regex: search, $options: 'i' } }
      ];
    }

    // Count total matching documents
    const totalCount = await Media.countDocuments(query);
    const totalPages = Math.ceil(totalCount / limit);

    // Get paginated results
    const mediaFiles = await Media.find(query)
      .sort({ [sortBy]: sortOrder === 'desc' ? -1 : 1 })
      .skip((page - 1) * limit)
      .limit(limit)
      .populate('folder', 'name color')
      .lean();

    return res.status(200).json({
      success: true,
      data: {
        mediaFiles,
        pagination: {
          currentPage: page,
          totalPages,
          totalCount,
          hasMore: page < totalPages
        }
      }
    });

  } catch (error) {
    console.error('Error in getMediaFiles:', error);
    throw error;
  }
};

export default asyncHandler(mediaHandler);
