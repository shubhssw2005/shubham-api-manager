import { S3Client, PutObjectCommand, CreateMultipartUploadCommand, UploadPartCommand, CompleteMultipartUploadCommand, AbortMultipartUploadCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import crypto from 'crypto';
import mimeTypes from 'mime-types';

/**
 * Presigned URL Service for Direct S3 Uploads
 * Handles secure direct-to-S3 uploads with multipart support and tenant isolation
 */
class PresignedURLService {
  constructor() {
    this.s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    });
    
    this.bucket = process.env.MEDIA_BUCKET || process.env.S3_BACKUP_BUCKET || 'media-uploads';
    this.defaultExpiration = 300; // 5 minutes
    this.multipartThreshold = 100 * 1024 * 1024; // 100MB
    this.maxFileSize = 5 * 1024 * 1024 * 1024; // 5GB
    this.allowedMimeTypes = new Set([
      // Images
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
      // Videos
      'video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm', 'video/x-msvideo',
      // Audio
      'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp3',
      // Documents
      'application/pdf', 'text/plain', 'text/csv',
      'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ]);
  }

  /**
   * Generate presigned URL for single file upload
   * @param {Object} params - Upload parameters
   * @param {string} params.tenantId - Tenant identifier
   * @param {string} params.originalName - Original filename
   * @param {string} params.contentType - MIME type
   * @param {number} params.size - File size in bytes
   * @param {string} params.userId - User identifier
   * @param {Object} params.metadata - Additional metadata
   * @returns {Promise<Object>} Presigned URL response
   */
  async generateUploadURL(params) {
    const { tenantId, originalName, contentType, size, userId, metadata = {} } = params;
    
    // Validate input parameters
    this.validateUploadParams(params);
    
    // Generate S3 key with tenant isolation
    const s3Key = this.generateS3Key(tenantId, originalName, userId);
    
    // Determine if multipart upload is needed
    const useMultipart = size > this.multipartThreshold;
    
    if (useMultipart) {
      return await this.generateMultipartUploadURL(s3Key, params);
    } else {
      return await this.generateSingleUploadURL(s3Key, params);
    }
  }

  /**
   * Generate presigned URL for single file upload
   * @private
   */
  async generateSingleUploadURL(s3Key, params) {
    const { contentType, size, tenantId, userId, originalName, metadata = {} } = params;
    
    const uploadMetadata = {
      'tenant-id': tenantId,
      'user-id': userId,
      'original-name': originalName,
      'upload-timestamp': Date.now().toString(),
      'upload-type': 'single',
      ...metadata
    };

    const command = new PutObjectCommand({
      Bucket: this.bucket,
      Key: s3Key,
      ContentType: contentType,
      ContentLength: size,
      Metadata: uploadMetadata,
      ServerSideEncryption: 'AES256'
    });

    const presignedUrl = await getSignedUrl(this.s3Client, command, {
      expiresIn: this.defaultExpiration
    });

    return {
      uploadType: 'single',
      uploadUrl: presignedUrl,
      s3Key,
      bucket: this.bucket,
      expiresIn: this.defaultExpiration,
      metadata: uploadMetadata,
      maxSize: size
    };
  }

  /**
   * Generate multipart upload URLs
   * @private
   */
  async generateMultipartUploadURL(s3Key, params) {
    const { contentType, size, tenantId, userId, originalName, metadata = {} } = params;
    
    const uploadMetadata = {
      'tenant-id': tenantId,
      'user-id': userId,
      'original-name': originalName,
      'upload-timestamp': Date.now().toString(),
      'upload-type': 'multipart',
      'total-size': size.toString(),
      ...metadata
    };

    // Create multipart upload
    const createCommand = new CreateMultipartUploadCommand({
      Bucket: this.bucket,
      Key: s3Key,
      ContentType: contentType,
      Metadata: uploadMetadata,
      ServerSideEncryption: 'AES256'
    });

    const createResult = await this.s3Client.send(createCommand);
    const uploadId = createResult.UploadId;

    // Calculate part size and count
    const partSize = Math.max(5 * 1024 * 1024, Math.ceil(size / 10000)); // Min 5MB, max 10000 parts
    const partCount = Math.ceil(size / partSize);

    // Generate presigned URLs for each part
    const partUrls = [];
    for (let partNumber = 1; partNumber <= partCount; partNumber++) {
      const partCommand = new UploadPartCommand({
        Bucket: this.bucket,
        Key: s3Key,
        PartNumber: partNumber,
        UploadId: uploadId
      });

      const partUrl = await getSignedUrl(this.s3Client, partCommand, {
        expiresIn: this.defaultExpiration
      });

      partUrls.push({
        partNumber,
        uploadUrl: partUrl,
        minSize: partNumber === partCount ? 0 : 5 * 1024 * 1024, // Last part can be smaller
        maxSize: partSize
      });
    }

    return {
      uploadType: 'multipart',
      uploadId,
      s3Key,
      bucket: this.bucket,
      partSize,
      partCount,
      partUrls,
      expiresIn: this.defaultExpiration,
      metadata: uploadMetadata,
      completeUrl: `/api/media/complete-multipart`,
      abortUrl: `/api/media/abort-multipart`
    };
  }

  /**
   * Complete multipart upload
   * @param {Object} params - Completion parameters
   * @returns {Promise<Object>} Completion result
   */
  async completeMultipartUpload(params) {
    const { s3Key, uploadId, parts, tenantId } = params;
    
    // Validate tenant access to this upload
    if (!this.validateTenantAccess(s3Key, tenantId)) {
      throw new Error('Unauthorized access to upload');
    }

    // Sort parts by part number
    const sortedParts = parts.sort((a, b) => a.PartNumber - b.PartNumber);

    const command = new CompleteMultipartUploadCommand({
      Bucket: this.bucket,
      Key: s3Key,
      UploadId: uploadId,
      MultipartUpload: {
        Parts: sortedParts.map(part => ({
          ETag: part.ETag,
          PartNumber: part.PartNumber
        }))
      }
    });

    const result = await this.s3Client.send(command);

    return {
      success: true,
      s3Key,
      bucket: this.bucket,
      location: result.Location,
      etag: result.ETag,
      versionId: result.VersionId
    };
  }

  /**
   * Abort multipart upload
   * @param {Object} params - Abort parameters
   * @returns {Promise<Object>} Abort result
   */
  async abortMultipartUpload(params) {
    const { s3Key, uploadId, tenantId } = params;
    
    // Validate tenant access to this upload
    if (!this.validateTenantAccess(s3Key, tenantId)) {
      throw new Error('Unauthorized access to upload');
    }

    const command = new AbortMultipartUploadCommand({
      Bucket: this.bucket,
      Key: s3Key,
      UploadId: uploadId
    });

    await this.s3Client.send(command);

    return {
      success: true,
      message: 'Multipart upload aborted successfully'
    };
  }

  /**
   * Generate tenant-based S3 key
   * @private
   */
  generateS3Key(tenantId, originalName, userId) {
    const timestamp = Date.now();
    const randomId = crypto.randomBytes(8).toString('hex');
    const extension = this.getFileExtension(originalName);
    const sanitizedName = this.sanitizeFilename(originalName);
    
    // Create hierarchical structure: tenants/{tenantId}/users/{userId}/media/{timestamp}-{randomId}-{sanitizedName}
    return `tenants/${tenantId}/users/${userId}/media/${timestamp}-${randomId}-${sanitizedName}${extension}`;
  }

  /**
   * Validate upload parameters
   * @private
   */
  validateUploadParams(params) {
    const { tenantId, originalName, contentType, size, userId } = params;
    
    // Required parameters
    if (!tenantId) throw new Error('Tenant ID is required');
    if (!originalName) throw new Error('Original filename is required');
    if (!contentType) throw new Error('Content type is required');
    if (!size || size <= 0) throw new Error('Valid file size is required');
    if (!userId) throw new Error('User ID is required');
    
    // File size validation
    if (size > this.maxFileSize) {
      throw new Error(`File size exceeds maximum allowed size of ${this.formatBytes(this.maxFileSize)}`);
    }
    
    // MIME type validation
    if (!this.allowedMimeTypes.has(contentType)) {
      throw new Error(`File type ${contentType} is not allowed`);
    }
    
    // Filename validation
    if (originalName.length > 255) {
      throw new Error('Filename is too long (max 255 characters)');
    }
    
    // Check for malicious filenames
    if (this.isMaliciousFilename(originalName)) {
      throw new Error('Invalid filename detected');
    }
  }

  /**
   * Validate tenant access to S3 key
   * @private
   */
  validateTenantAccess(s3Key, tenantId) {
    return s3Key.startsWith(`tenants/${tenantId}/`);
  }

  /**
   * Check for malicious filenames
   * @private
   */
  isMaliciousFilename(filename) {
    const maliciousPatterns = [
      /\.\./,           // Directory traversal
      /[<>:"|?*]/,      // Invalid characters
      /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)/i, // Windows reserved names (with or without extension)
      /^\./,            // Hidden files
      /\.(exe|bat|cmd|scr|pif|com)$/i // Executable files
    ];
    
    return maliciousPatterns.some(pattern => pattern.test(filename));
  }

  /**
   * Sanitize filename for S3 key
   * @private
   */
  sanitizeFilename(filename) {
    // Remove extension for sanitization
    const nameWithoutExt = filename.replace(/\.[^/.]+$/, '');
    
    return nameWithoutExt
      .replace(/[^a-zA-Z0-9.-]/g, '-')  // Replace invalid chars with dash
      .replace(/-+/g, '-')              // Replace multiple dashes with single
      .replace(/^-|-$/g, '')            // Remove leading/trailing dashes
      .toLowerCase()
      .substring(0, 100);               // Limit length
  }

  /**
   * Get file extension
   * @private
   */
  getFileExtension(filename) {
    const match = filename.match(/\.[^/.]+$/);
    return match ? match[0] : '';
  }

  /**
   * Format bytes to human readable string
   * @private
   */
  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Generate resumable upload session
   * @param {Object} params - Session parameters
   * @returns {Promise<Object>} Session details
   */
  async createResumableSession(params) {
    const { tenantId, originalName, contentType, size, userId, metadata = {} } = params;
    
    // Validate parameters
    this.validateUploadParams(params);
    
    // Generate session ID
    const sessionId = crypto.randomUUID();
    const s3Key = this.generateS3Key(tenantId, originalName, userId);
    
    // For resumable uploads, we'll use multipart upload
    const sessionMetadata = {
      sessionId,
      s3Key,
      tenantId,
      userId,
      originalName,
      contentType,
      size,
      createdAt: new Date().toISOString(),
      status: 'created',
      ...metadata
    };

    // Store session metadata (in production, this would be stored in Redis or database)
    // For now, we'll return it in the response
    
    return {
      sessionId,
      s3Key,
      resumable: true,
      chunkSize: Math.max(5 * 1024 * 1024, Math.ceil(size / 1000)), // Min 5MB chunks
      totalChunks: Math.ceil(size / Math.max(5 * 1024 * 1024, Math.ceil(size / 1000))),
      expiresIn: 24 * 60 * 60, // 24 hours for resumable uploads
      uploadUrl: `/api/media/resumable-upload/${sessionId}`,
      statusUrl: `/api/media/resumable-status/${sessionId}`,
      metadata: sessionMetadata
    };
  }
}

export default PresignedURLService;