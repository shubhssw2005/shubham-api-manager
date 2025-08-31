import { S3Client, GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3';
import Media from '../../models/Media.js';

/**
 * Base Media Processor - Common functionality for all media processors
 */
class BaseProcessor {
  constructor() {
    this.s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    });
    
    this.bucket = process.env.MEDIA_BUCKET || process.env.S3_BACKUP_BUCKET;
    this.processingTimeout = parseInt(process.env.PROCESSING_TIMEOUT) || 300000; // 5 minutes
  }

  /**
   * Process media file - to be implemented by subclasses
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async process(job) {
    throw new Error('Process method must be implemented by subclass');
  }

  /**
   * Download file from S3
   * @param {string} bucket - S3 bucket
   * @param {string} key - S3 key
   * @returns {Promise<Buffer>} File buffer
   */
  async downloadFile(bucket, key) {
    const command = new GetObjectCommand({
      Bucket: bucket,
      Key: key
    });

    const response = await this.s3Client.send(command);
    
    // Convert stream to buffer
    const chunks = [];
    for await (const chunk of response.Body) {
      chunks.push(chunk);
    }
    
    return Buffer.concat(chunks);
  }

  /**
   * Upload processed file to S3
   * @param {Buffer} buffer - File buffer
   * @param {string} key - S3 key
   * @param {Object} options - Upload options
   * @returns {Promise<Object>} Upload result
   */
  async uploadFile(buffer, key, options = {}) {
    const command = new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      Body: buffer,
      ContentType: options.contentType || 'application/octet-stream',
      Metadata: options.metadata || {},
      ServerSideEncryption: 'AES256'
    });

    const result = await this.s3Client.send(command);
    
    return {
      key,
      bucket: this.bucket,
      etag: result.ETag,
      versionId: result.VersionId,
      url: `https://${this.bucket}.s3.${process.env.AWS_REGION || 'us-east-1'}.amazonaws.com/${key}`
    };
  }

  /**
   * Generate processed file key
   * @param {string} originalKey - Original S3 key
   * @param {string} suffix - File suffix (e.g., 'thumb_small', 'transcoded')
   * @param {string} extension - New file extension
   * @returns {string} New S3 key
   */
  generateProcessedKey(originalKey, suffix, extension = null) {
    const keyParts = originalKey.split('/');
    const filename = keyParts.pop();
    const nameWithoutExt = filename.replace(/\.[^/.]+$/, '');
    const originalExt = filename.split('.').pop();
    const newExt = extension || originalExt;
    
    const newFilename = `${nameWithoutExt}_${suffix}.${newExt}`;
    keyParts.push('processed', newFilename);
    
    return keyParts.join('/');
  }

  /**
   * Update media record in database
   * @param {Object} job - Processing job
   * @param {Object} processingResult - Result of processing
   * @returns {Promise<Object>} Updated media record
   */
  async updateMediaRecord(job, processingResult) {
    try {
      // Find media record by S3 key or create new one
      let media = await Media.findOne({ path: job.key });
      
      if (!media) {
        // Create new media record
        const filename = job.key.split('/').pop();
        media = new Media({
          filename,
          originalName: filename,
          path: job.key,
          url: `https://${job.bucket}.s3.${process.env.AWS_REGION || 'us-east-1'}.amazonaws.com/${job.key}`,
          size: job.size,
          mimeType: this.getMimeTypeFromKey(job.key),
          storageProvider: 's3',
          uploadedBy: job.userId ? new mongoose.Types.ObjectId(job.userId) : null,
          processingStatus: 'processing'
        });
      }

      // Update processing status and results
      media.processingStatus = 'completed';
      media.processedAt = new Date();
      
      // Add processing results to metadata
      if (processingResult.metadata) {
        media.metadata = { ...media.metadata, ...processingResult.metadata };
      }
      
      // Add thumbnails if generated
      if (processingResult.thumbnails) {
        media.thumbnails = processingResult.thumbnails;
      }
      
      // Add video metadata if processed
      if (processingResult.videoMetadata) {
        media.metadata = { ...media.metadata, ...processingResult.videoMetadata };
      }

      await media.save();
      return media;
      
    } catch (error) {
      console.error('Error updating media record:', error);
      throw error;
    }
  }

  /**
   * Get MIME type from file key
   * @param {string} key - S3 key
   * @returns {string} MIME type
   */
  getMimeTypeFromKey(key) {
    const extension = key.split('.').pop()?.toLowerCase();
    
    const mimeTypes = {
      // Images
      'jpg': 'image/jpeg',
      'jpeg': 'image/jpeg',
      'png': 'image/png',
      'gif': 'image/gif',
      'webp': 'image/webp',
      'svg': 'image/svg+xml',
      'bmp': 'image/bmp',
      'tiff': 'image/tiff',
      
      // Videos
      'mp4': 'video/mp4',
      'avi': 'video/x-msvideo',
      'mov': 'video/quicktime',
      'wmv': 'video/x-ms-wmv',
      'flv': 'video/x-flv',
      'webm': 'video/webm',
      'mkv': 'video/x-matroska',
      
      // Audio
      'mp3': 'audio/mpeg',
      'wav': 'audio/wav',
      'ogg': 'audio/ogg',
      'aac': 'audio/aac',
      'flac': 'audio/flac',
      
      // Documents
      'pdf': 'application/pdf',
      'doc': 'application/msword',
      'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'txt': 'text/plain'
    };
    
    return mimeTypes[extension] || 'application/octet-stream';
  }

  /**
   * Validate file size and type
   * @param {Object} job - Processing job
   * @returns {boolean} Validation result
   */
  validateFile(job) {
    const maxSize = this.getMaxFileSize(job.fileType);
    
    if (job.size > maxSize) {
      throw new Error(`File size ${job.size} exceeds maximum allowed size ${maxSize} for ${job.fileType}`);
    }
    
    return true;
  }

  /**
   * Get maximum file size for file type
   * @param {string} fileType - File type
   * @returns {number} Maximum size in bytes
   */
  getMaxFileSize(fileType) {
    const maxSizes = {
      'image': 50 * 1024 * 1024,      // 50MB
      'video': 2 * 1024 * 1024 * 1024, // 2GB
      'audio': 100 * 1024 * 1024,     // 100MB
      'document': 100 * 1024 * 1024   // 100MB
    };
    
    return maxSizes[fileType] || 50 * 1024 * 1024;
  }

  /**
   * Create processing timeout promise
   * @returns {Promise} Timeout promise
   */
  createTimeoutPromise() {
    return new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Processing timeout after ${this.processingTimeout}ms`));
      }, this.processingTimeout);
    });
  }

  /**
   * Execute processing with timeout
   * @param {Promise} processingPromise - Processing promise
   * @returns {Promise} Result or timeout
   */
  async executeWithTimeout(processingPromise) {
    return Promise.race([
      processingPromise,
      this.createTimeoutPromise()
    ]);
  }

  /**
   * Log processing metrics
   * @param {Object} job - Processing job
   * @param {number} startTime - Processing start time
   * @param {Object} result - Processing result
   */
  logProcessingMetrics(job, startTime, result) {
    const duration = Date.now() - startTime;
    const sizeInMB = (job.size / (1024 * 1024)).toFixed(2);
    
    console.log(`Processing completed for ${job.fileType}:`, {
      jobId: job.jobId,
      fileSize: `${sizeInMB}MB`,
      duration: `${duration}ms`,
      processingRate: `${(job.size / duration * 1000 / (1024 * 1024)).toFixed(2)}MB/s`,
      outputFiles: result.outputFiles?.length || 0
    });
  }

  /**
   * Handle processing error
   * @param {Error} error - Processing error
   * @param {Object} job - Processing job
   * @returns {Error} Enhanced error
   */
  handleProcessingError(error, job) {
    const enhancedError = new Error(`${this.constructor.name} failed: ${error.message}`);
    enhancedError.originalError = error;
    enhancedError.jobId = job.jobId;
    enhancedError.fileType = job.fileType;
    enhancedError.fileSize = job.size;
    
    console.error(`Processing error for job ${job.jobId}:`, {
      error: error.message,
      stack: error.stack,
      job: {
        jobId: job.jobId,
        fileType: job.fileType,
        size: job.size,
        key: job.key
      }
    });
    
    return enhancedError;
  }
}

export default BaseProcessor;