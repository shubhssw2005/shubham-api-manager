import formidable from 'formidable';
import fs from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import dbConnect from '../../../lib/dbConnect';
import Media from '../../../models/Media';
import MediaFolder from '../../../models/MediaFolder';
import { requireApprovedUser } from '../../../middleware/auth';
import { asyncHandler, errorHandler } from '../../../lib/errorHandler';
import { FileValidator } from '../../../lib/fileProcessor';
import { FileProcessor } from '../../../lib/fileProcessor';
import { StorageFactory } from '../../../lib/storage';
import { ValidationError, StorageError, ProcessingError } from '../../../lib/errors';

// Disable Next.js body parser for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};

const uploadHandler = async (req, res) => {
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

  // Initialize file processor and validator
  const fileValidator = new FileValidator({
    maxFileSize: 50 * 1024 * 1024, // 50MB
    allowedTypes: [
      // Images
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
      // Videos
      'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/webm',
      // Documents
      'application/pdf', 'text/plain', 'text/csv',
      'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      // Audio
      'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4'
    ]
  });

  const fileProcessor = new FileProcessor();
  const storageProvider = StorageFactory.createFromEnv();

  // Configure formidable
  const form = formidable({
    maxFileSize: 50 * 1024 * 1024, // 50MB
    maxFiles: 10, // Allow up to 10 files at once
    keepExtensions: true,
    multiples: true,
    uploadDir: '/tmp', // Temporary directory
  });

  try {
    // Parse the form data
    const [fields, files] = await form.parse(req);
    
    // Extract folder ID if provided
    const folderId = fields.folderId ? fields.folderId[0] : null;
    
    // Validate folder if provided
    let folder = null;
    if (folderId) {
      folder = await MediaFolder.findById(folderId);
      if (!folder) {
        throw new ValidationError('Folder not found');
      }
      
      // Check if user has write access to folder
      if (!folder.canUserAccess(user._id, 'write')) {
        throw new ValidationError('No write access to folder');
      }
    }

    // Process uploaded files
    const uploadedFiles = [];
    const errors = [];

    // Handle both single and multiple files
    const fileList = Array.isArray(files.files) ? files.files : [files.files].filter(Boolean);
    
    if (fileList.length === 0) {
      throw new ValidationError('No files provided');
    }

    for (const file of fileList) {
      try {
        // Read file buffer
        const fileBuffer = await fs.readFile(file.filepath);
        
        // Create file object for validation
        const fileObj = {
          buffer: fileBuffer,
          originalName: file.originalFilename,
          mimeType: file.mimetype,
          size: file.size
        };

        // Validate file
        const validation = await fileValidator.validateFile(fileObj);
        if (!validation.isValid) {
          errors.push({
            filename: file.originalFilename,
            errors: validation.errors
          });
          continue;
        }

        // Generate unique filename
        const fileExtension = path.extname(file.originalFilename);
        const uniqueFilename = `${uuidv4()}${fileExtension}`;
        const storagePath = `media/${new Date().getFullYear()}/${new Date().getMonth() + 1}/${uniqueFilename}`;

        // Process file based on type
        const processingResult = await fileProcessor.processFile(fileBuffer, {
          mimeType: file.mimetype,
          originalName: file.originalFilename
        });

        // Upload original file to storage
        const uploadResult = await storageProvider.upload(fileBuffer, storagePath, {
          contentType: file.mimetype,
          metadata: {
            originalName: file.originalFilename,
            uploadedBy: user._id.toString()
          }
        });

        // Upload thumbnails if they exist
        const thumbnails = [];
        if (processingResult.thumbnails) {
          for (const thumbnail of processingResult.thumbnails) {
            const thumbPath = `media/thumbnails/${new Date().getFullYear()}/${new Date().getMonth() + 1}/${thumbnail.name}_${uniqueFilename}`;
            const thumbUploadResult = await storageProvider.upload(thumbnail.buffer, thumbPath, {
              contentType: 'image/jpeg'
            });

            thumbnails.push({
              size: thumbnail.name,
              url: thumbUploadResult.url,
              path: thumbPath,
              width: thumbnail.width,
              height: thumbnail.height,
              fileSize: thumbnail.size
            });
          }
        }

        // Create media record in database
        const mediaData = {
          filename: uniqueFilename,
          originalName: file.originalFilename,
          mimeType: file.mimetype,
          size: file.size,
          path: storagePath,
          url: uploadResult.url,
          storageProvider: storageProvider.constructor.name.toLowerCase().replace('storageprovider', ''),
          thumbnails,
          metadata: processingResult.metadata || {},
          folder: folder ? folder._id : null,
          uploadedBy: user._id,
          processedAt: new Date(),
          processingStatus: 'completed'
        };

        const media = new Media(mediaData);
        await media.save();

        uploadedFiles.push({
          id: media._id,
          filename: media.filename,
          originalName: media.originalName,
          mimeType: media.mimeType,
          size: media.size,
          url: media.url,
          thumbnails: media.thumbnails,
          metadata: media.metadata,
          folder: media.folder,
          createdAt: media.createdAt
        });

        // Clean up temporary file
        await fs.unlink(file.filepath).catch(() => {}); // Ignore errors

      } catch (fileError) {
        console.error(`Error processing file ${file.originalFilename}:`, fileError);
        errors.push({
          filename: file.originalFilename,
          errors: [fileError.message]
        });
        
        // Clean up temporary file
        await fs.unlink(file.filepath).catch(() => {}); // Ignore errors
      }
    }

    // Return response
    const response = {
      success: true,
      data: {
        uploadedFiles,
        totalUploaded: uploadedFiles.length,
        totalFiles: fileList.length
      }
    };

    if (errors.length > 0) {
      response.errors = errors;
      response.message = `${uploadedFiles.length} files uploaded successfully, ${errors.length} files failed`;
    } else {
      response.message = `${uploadedFiles.length} files uploaded successfully`;
    }

    res.status(200).json(response);

  } catch (error) {
    console.error('Upload error:', error);
    
    if (error instanceof ValidationError) {
      return res.status(400).json({
        success: false,
        error: { message: error.message, code: 'VALIDATION_ERROR' }
      });
    }
    
    if (error instanceof StorageError) {
      return res.status(500).json({
        success: false,
        error: { message: 'Storage error occurred', code: 'STORAGE_ERROR' }
      });
    }
    
    if (error instanceof ProcessingError) {
      return res.status(500).json({
        success: false,
        error: { message: 'File processing error occurred', code: 'PROCESSING_ERROR' }
      });
    }

    return res.status(500).json({
      success: false,
      error: { message: 'Upload failed', code: 'UPLOAD_ERROR' }
    });
  }
};

export default asyncHandler(uploadHandler);