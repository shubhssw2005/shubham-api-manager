import BaseProcessor from './BaseProcessor.js';
import sharp from 'sharp';

/**
 * Image Processor - Handles image processing including thumbnail generation
 */
class ImageProcessor extends BaseProcessor {
  constructor() {
    super();
    
    // Thumbnail configurations
    this.thumbnailSizes = {
      small: { width: 150, height: 150, quality: 80 },
      medium: { width: 300, height: 300, quality: 85 },
      large: { width: 800, height: 600, quality: 90 }
    };
    
    // Supported formats for conversion
    this.supportedFormats = ['jpeg', 'png', 'webp', 'avif'];
    this.defaultFormat = 'jpeg';
    this.maxDimension = 4096; // Maximum width or height
  }

  /**
   * Process image file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async process(job) {
    const startTime = Date.now();
    
    try {
      // Validate file
      this.validateFile(job);
      
      // Download original image
      console.log(`Downloading image: ${job.key}`);
      const imageBuffer = await this.downloadFile(job.bucket, job.key);
      
      // Process image with timeout
      const result = await this.executeWithTimeout(
        this.processImage(imageBuffer, job)
      );
      
      // Update media record
      await this.updateMediaRecord(job, result);
      
      // Log metrics
      this.logProcessingMetrics(job, startTime, result);
      
      return result;
      
    } catch (error) {
      throw this.handleProcessingError(error, job);
    }
  }

  /**
   * Process image buffer
   * @param {Buffer} imageBuffer - Original image buffer
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async processImage(imageBuffer, job) {
    // Get image metadata
    const image = sharp(imageBuffer);
    const metadata = await image.metadata();
    
    console.log(`Processing image: ${metadata.width}x${metadata.height}, format: ${metadata.format}`);
    
    // Validate image dimensions
    if (metadata.width > this.maxDimension || metadata.height > this.maxDimension) {
      throw new Error(`Image dimensions ${metadata.width}x${metadata.height} exceed maximum ${this.maxDimension}x${this.maxDimension}`);
    }
    
    const results = {
      metadata: {
        width: metadata.width,
        height: metadata.height,
        format: metadata.format,
        channels: metadata.channels,
        density: metadata.density,
        hasAlpha: metadata.hasAlpha,
        colorSpace: metadata.space,
        orientation: metadata.orientation
      },
      thumbnails: [],
      outputFiles: []
    };

    // Generate thumbnails
    for (const [size, config] of Object.entries(this.thumbnailSizes)) {
      try {
        const thumbnail = await this.generateThumbnail(image, size, config, job);
        results.thumbnails.push(thumbnail);
        results.outputFiles.push(thumbnail.key);
      } catch (error) {
        console.error(`Failed to generate ${size} thumbnail:`, error);
        // Continue with other thumbnails
      }
    }

    // Generate optimized version if needed
    if (this.shouldOptimize(metadata, job)) {
      try {
        const optimized = await this.generateOptimizedVersion(image, metadata, job);
        results.optimized = optimized;
        results.outputFiles.push(optimized.key);
      } catch (error) {
        console.error('Failed to generate optimized version:', error);
      }
    }

    // Generate WebP version for modern browsers
    if (metadata.format !== 'webp') {
      try {
        const webp = await this.generateWebPVersion(image, job);
        results.webp = webp;
        results.outputFiles.push(webp.key);
      } catch (error) {
        console.error('Failed to generate WebP version:', error);
      }
    }

    return results;
  }

  /**
   * Generate thumbnail
   * @param {Sharp} image - Sharp image instance
   * @param {string} size - Thumbnail size name
   * @param {Object} config - Thumbnail configuration
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Thumbnail info
   */
  async generateThumbnail(image, size, config, job) {
    const thumbnailBuffer = await image
      .clone()
      .resize(config.width, config.height, {
        fit: 'cover',
        position: 'center'
      })
      .jpeg({ quality: config.quality })
      .toBuffer();

    // Generate S3 key for thumbnail
    const thumbnailKey = this.generateProcessedKey(job.key, `thumb_${size}`, 'jpg');
    
    // Upload thumbnail
    const uploadResult = await this.uploadFile(thumbnailBuffer, thumbnailKey, {
      contentType: 'image/jpeg',
      metadata: {
        'original-key': job.key,
        'thumbnail-size': size,
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });

    return {
      size,
      key: thumbnailKey,
      url: uploadResult.url,
      width: config.width,
      height: config.height,
      fileSize: thumbnailBuffer.length,
      format: 'jpeg'
    };
  }

  /**
   * Generate optimized version of image
   * @param {Sharp} image - Sharp image instance
   * @param {Object} metadata - Image metadata
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Optimized image info
   */
  async generateOptimizedVersion(image, metadata, job) {
    let optimizedImage = image.clone();
    
    // Resize if too large
    if (metadata.width > 1920 || metadata.height > 1080) {
      optimizedImage = optimizedImage.resize(1920, 1080, {
        fit: 'inside',
        withoutEnlargement: true
      });
    }
    
    // Optimize based on format
    let outputFormat = metadata.format;
    let quality = 85;
    
    if (metadata.format === 'png' && !metadata.hasAlpha) {
      // Convert PNG without alpha to JPEG for better compression
      outputFormat = 'jpeg';
      optimizedImage = optimizedImage.jpeg({ quality });
    } else if (metadata.format === 'jpeg') {
      optimizedImage = optimizedImage.jpeg({ quality, progressive: true });
    } else if (metadata.format === 'png') {
      optimizedImage = optimizedImage.png({ compressionLevel: 9 });
    }
    
    const optimizedBuffer = await optimizedImage.toBuffer();
    
    // Only save if significantly smaller
    const compressionRatio = optimizedBuffer.length / job.size;
    if (compressionRatio > 0.9) {
      console.log(`Skipping optimized version, compression ratio: ${compressionRatio}`);
      return null;
    }
    
    const optimizedKey = this.generateProcessedKey(job.key, 'optimized', outputFormat);
    
    const uploadResult = await this.uploadFile(optimizedBuffer, optimizedKey, {
      contentType: `image/${outputFormat}`,
      metadata: {
        'original-key': job.key,
        'optimization-type': 'size-optimized',
        'compression-ratio': compressionRatio.toString(),
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });

    return {
      key: optimizedKey,
      url: uploadResult.url,
      format: outputFormat,
      fileSize: optimizedBuffer.length,
      compressionRatio,
      savings: Math.round((1 - compressionRatio) * 100)
    };
  }

  /**
   * Generate WebP version
   * @param {Sharp} image - Sharp image instance
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} WebP image info
   */
  async generateWebPVersion(image, job) {
    const webpBuffer = await image
      .clone()
      .webp({ quality: 80, effort: 4 })
      .toBuffer();
    
    const webpKey = this.generateProcessedKey(job.key, 'webp', 'webp');
    
    const uploadResult = await this.uploadFile(webpBuffer, webpKey, {
      contentType: 'image/webp',
      metadata: {
        'original-key': job.key,
        'format': 'webp',
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });

    return {
      key: webpKey,
      url: uploadResult.url,
      format: 'webp',
      fileSize: webpBuffer.length,
      compressionRatio: webpBuffer.length / job.size
    };
  }

  /**
   * Check if image should be optimized
   * @param {Object} metadata - Image metadata
   * @param {Object} job - Processing job
   * @returns {boolean} Should optimize
   */
  shouldOptimize(metadata, job) {
    // Optimize if file is large
    if (job.size > 1024 * 1024) { // 1MB
      return true;
    }
    
    // Optimize if dimensions are large
    if (metadata.width > 1920 || metadata.height > 1080) {
      return true;
    }
    
    // Optimize PNG without alpha channel
    if (metadata.format === 'png' && !metadata.hasAlpha) {
      return true;
    }
    
    return false;
  }

  /**
   * Extract EXIF data safely
   * @param {Object} metadata - Sharp metadata
   * @returns {Object} Cleaned EXIF data
   */
  extractExifData(metadata) {
    if (!metadata.exif) {
      return {};
    }
    
    try {
      // Extract safe EXIF data (remove potentially sensitive info)
      const safeExif = {};
      const allowedFields = [
        'Make', 'Model', 'DateTime', 'Orientation',
        'XResolution', 'YResolution', 'Software',
        'ColorSpace', 'ExifImageWidth', 'ExifImageHeight'
      ];
      
      for (const field of allowedFields) {
        if (metadata.exif[field]) {
          safeExif[field] = metadata.exif[field];
        }
      }
      
      return safeExif;
    } catch (error) {
      console.error('Error extracting EXIF data:', error);
      return {};
    }
  }

  /**
   * Validate image file
   * @param {Object} job - Processing job
   * @returns {boolean} Validation result
   */
  validateFile(job) {
    super.validateFile(job);
    
    // Check if it's actually an image
    const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp', 'tiff'];
    const extension = job.key.split('.').pop()?.toLowerCase();
    
    if (!imageExtensions.includes(extension)) {
      throw new Error(`Unsupported image format: ${extension}`);
    }
    
    return true;
  }

  /**
   * Get processing capabilities
   * @returns {Object} Processor capabilities
   */
  getCapabilities() {
    return {
      supportedFormats: ['jpeg', 'jpg', 'png', 'gif', 'webp', 'svg', 'bmp', 'tiff'],
      outputFormats: ['jpeg', 'png', 'webp'],
      thumbnailSizes: Object.keys(this.thumbnailSizes),
      maxDimension: this.maxDimension,
      features: [
        'thumbnail-generation',
        'format-conversion',
        'size-optimization',
        'webp-conversion',
        'exif-extraction',
        'progressive-jpeg'
      ]
    };
  }
}

export default ImageProcessor;