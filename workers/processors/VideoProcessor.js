import BaseProcessor from './BaseProcessor.js';
import ffmpeg from 'fluent-ffmpeg';
import { promisify } from 'util';
import { createWriteStream, createReadStream, unlinkSync, mkdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

/**
 * Video Processor - Handles video transcoding and thumbnail generation
 */
class VideoProcessor extends BaseProcessor {
  constructor() {
    super();
    
    // Video processing configurations
    this.transcodingProfiles = {
      '480p': {
        resolution: '854x480',
        videoBitrate: '1000k',
        audioBitrate: '128k',
        fps: 30,
        format: 'mp4',
        codec: 'libx264'
      },
      '720p': {
        resolution: '1280x720',
        videoBitrate: '2500k',
        audioBitrate: '128k',
        fps: 30,
        format: 'mp4',
        codec: 'libx264'
      },
      '1080p': {
        resolution: '1920x1080',
        videoBitrate: '5000k',
        audioBitrate: '192k',
        fps: 30,
        format: 'mp4',
        codec: 'libx264'
      }
    };
    
    // HLS configuration
    this.hlsConfig = {
      segmentDuration: 10, // seconds
      playlistType: 'vod',
      segmentFormat: 'mp4'
    };
    
    // Thumbnail configuration
    this.thumbnailConfig = {
      count: 5,
      width: 320,
      height: 180,
      format: 'jpg',
      quality: 80
    };
    
    // Temporary directory for processing
    this.tempDir = join(tmpdir(), 'video-processing');
    this.ensureTempDir();
  }

  /**
   * Ensure temporary directory exists
   */
  ensureTempDir() {
    try {
      mkdirSync(this.tempDir, { recursive: true });
    } catch (error) {
      console.error('Failed to create temp directory:', error);
    }
  }

  /**
   * Process video file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async process(job) {
    const startTime = Date.now();
    let tempFiles = [];
    
    try {
      // Validate file
      this.validateFile(job);
      
      // Download original video
      console.log(`Downloading video: ${job.key}`);
      const videoBuffer = await this.downloadFile(job.bucket, job.key);
      
      // Save to temporary file
      const tempInputPath = join(this.tempDir, `input_${job.jobId}.${this.getFileExtension(job.key)}`);
      await this.writeBufferToFile(videoBuffer, tempInputPath);
      tempFiles.push(tempInputPath);
      
      // Process video with timeout
      const result = await this.executeWithTimeout(
        this.processVideo(tempInputPath, job)
      );
      
      // Update media record
      await this.updateMediaRecord(job, result);
      
      // Log metrics
      this.logProcessingMetrics(job, startTime, result);
      
      return result;
      
    } catch (error) {
      throw this.handleProcessingError(error, job);
    } finally {
      // Clean up temporary files
      this.cleanupTempFiles(tempFiles);
    }
  }

  /**
   * Process video file
   * @param {string} inputPath - Path to input video file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async processVideo(inputPath, job) {
    // Get video metadata
    const metadata = await this.getVideoMetadata(inputPath);
    console.log(`Processing video: ${metadata.width}x${metadata.height}, duration: ${metadata.duration}s`);
    
    const results = {
      metadata: {
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration,
        bitrate: metadata.bitrate,
        fps: metadata.fps,
        codec: metadata.codec,
        format: metadata.format
      },
      transcoded: [],
      thumbnails: [],
      hlsManifest: null,
      outputFiles: []
    };

    // Generate video thumbnails
    try {
      const thumbnails = await this.generateVideoThumbnails(inputPath, job, metadata);
      results.thumbnails = thumbnails;
      results.outputFiles.push(...thumbnails.map(t => t.key));
    } catch (error) {
      console.error('Failed to generate video thumbnails:', error);
    }

    // Determine which transcoding profiles to use
    const profilesToUse = this.selectTranscodingProfiles(metadata);
    
    // Transcode to different resolutions
    for (const profileName of profilesToUse) {
      try {
        const profile = this.transcodingProfiles[profileName];
        const transcoded = await this.transcodeVideo(inputPath, profile, profileName, job);
        results.transcoded.push(transcoded);
        results.outputFiles.push(transcoded.key);
      } catch (error) {
        console.error(`Failed to transcode to ${profileName}:`, error);
      }
    }

    // Generate HLS playlist if enabled
    if (process.env.ENABLE_HLS_TRANSCODING === 'true') {
      try {
        const hls = await this.generateHLSPlaylist(inputPath, job, metadata);
        results.hlsManifest = hls;
        results.outputFiles.push(hls.manifestKey);
        results.outputFiles.push(...hls.segmentKeys);
      } catch (error) {
        console.error('Failed to generate HLS playlist:', error);
      }
    }

    return results;
  }

  /**
   * Get video metadata using ffprobe
   * @param {string} inputPath - Path to video file
   * @returns {Promise<Object>} Video metadata
   */
  async getVideoMetadata(inputPath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) {
          reject(new Error(`Failed to get video metadata: ${err.message}`));
          return;
        }
        
        const videoStream = metadata.streams.find(stream => stream.codec_type === 'video');
        const audioStream = metadata.streams.find(stream => stream.codec_type === 'audio');
        
        if (!videoStream) {
          reject(new Error('No video stream found'));
          return;
        }
        
        resolve({
          width: videoStream.width,
          height: videoStream.height,
          duration: parseFloat(metadata.format.duration),
          bitrate: parseInt(metadata.format.bit_rate),
          fps: this.parseFPS(videoStream.r_frame_rate),
          codec: videoStream.codec_name,
          format: metadata.format.format_name,
          hasAudio: !!audioStream,
          audioCodec: audioStream?.codec_name,
          fileSize: parseInt(metadata.format.size)
        });
      });
    });
  }

  /**
   * Generate video thumbnails
   * @param {string} inputPath - Path to input video
   * @param {Object} job - Processing job
   * @param {Object} metadata - Video metadata
   * @returns {Promise<Array>} Array of thumbnail info
   */
  async generateVideoThumbnails(inputPath, job, metadata) {
    const thumbnails = [];
    const duration = metadata.duration;
    const count = Math.min(this.thumbnailConfig.count, Math.floor(duration / 10)); // Max 1 per 10 seconds
    
    for (let i = 0; i < count; i++) {
      const timestamp = (duration / (count + 1)) * (i + 1); // Evenly distributed
      const thumbnailPath = join(this.tempDir, `thumb_${job.jobId}_${i}.jpg`);
      
      try {
        await this.extractThumbnail(inputPath, thumbnailPath, timestamp);
        
        // Upload thumbnail
        const thumbnailBuffer = await this.readFileToBuffer(thumbnailPath);
        const thumbnailKey = this.generateProcessedKey(job.key, `thumb_${i}`, 'jpg');
        
        const uploadResult = await this.uploadFile(thumbnailBuffer, thumbnailKey, {
          contentType: 'image/jpeg',
          metadata: {
            'original-key': job.key,
            'thumbnail-index': i.toString(),
            'timestamp': timestamp.toString(),
            'tenant-id': job.tenantId,
            'processing-job-id': job.jobId
          }
        });
        
        thumbnails.push({
          index: i,
          key: thumbnailKey,
          url: uploadResult.url,
          timestamp,
          width: this.thumbnailConfig.width,
          height: this.thumbnailConfig.height,
          fileSize: thumbnailBuffer.length
        });
        
        // Clean up temp file
        unlinkSync(thumbnailPath);
        
      } catch (error) {
        console.error(`Failed to generate thumbnail ${i}:`, error);
      }
    }
    
    return thumbnails;
  }

  /**
   * Extract single thumbnail from video
   * @param {string} inputPath - Input video path
   * @param {string} outputPath - Output thumbnail path
   * @param {number} timestamp - Timestamp in seconds
   * @returns {Promise<void>}
   */
  async extractThumbnail(inputPath, outputPath, timestamp) {
    return new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .seekInput(timestamp)
        .frames(1)
        .size(`${this.thumbnailConfig.width}x${this.thumbnailConfig.height}`)
        .format('image2')
        .output(outputPath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
  }

  /**
   * Transcode video to specific profile
   * @param {string} inputPath - Input video path
   * @param {Object} profile - Transcoding profile
   * @param {string} profileName - Profile name
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Transcoded video info
   */
  async transcodeVideo(inputPath, profile, profileName, job) {
    const outputPath = join(this.tempDir, `transcoded_${job.jobId}_${profileName}.${profile.format}`);
    
    await new Promise((resolve, reject) => {
      let command = ffmpeg(inputPath)
        .videoCodec(profile.codec)
        .videoBitrate(profile.videoBitrate)
        .size(profile.resolution)
        .fps(profile.fps)
        .format(profile.format);
      
      // Add audio encoding if video has audio
      if (profile.audioBitrate) {
        command = command
          .audioCodec('aac')
          .audioBitrate(profile.audioBitrate);
      }
      
      command
        .output(outputPath)
        .on('progress', (progress) => {
          if (progress.percent) {
            console.log(`Transcoding ${profileName}: ${Math.round(progress.percent)}%`);
          }
        })
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
    
    // Upload transcoded video
    const transcodedBuffer = await this.readFileToBuffer(outputPath);
    const transcodedKey = this.generateProcessedKey(job.key, profileName, profile.format);
    
    const uploadResult = await this.uploadFile(transcodedBuffer, transcodedKey, {
      contentType: `video/${profile.format}`,
      metadata: {
        'original-key': job.key,
        'transcoding-profile': profileName,
        'resolution': profile.resolution,
        'bitrate': profile.videoBitrate,
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });
    
    // Clean up temp file
    unlinkSync(outputPath);
    
    return {
      profile: profileName,
      key: transcodedKey,
      url: uploadResult.url,
      resolution: profile.resolution,
      bitrate: profile.videoBitrate,
      format: profile.format,
      fileSize: transcodedBuffer.length
    };
  }

  /**
   * Generate HLS playlist
   * @param {string} inputPath - Input video path
   * @param {Object} job - Processing job
   * @param {Object} metadata - Video metadata
   * @returns {Promise<Object>} HLS playlist info
   */
  async generateHLSPlaylist(inputPath, job, metadata) {
    const hlsDir = join(this.tempDir, `hls_${job.jobId}`);
    mkdirSync(hlsDir, { recursive: true });
    
    const playlistPath = join(hlsDir, 'playlist.m3u8');
    const segmentPattern = join(hlsDir, 'segment_%03d.ts');
    
    // Generate HLS segments
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .videoCodec('libx264')
        .audioCodec('aac')
        .format('hls')
        .addOption('-hls_time', this.hlsConfig.segmentDuration)
        .addOption('-hls_playlist_type', this.hlsConfig.playlistType)
        .addOption('-hls_segment_filename', segmentPattern)
        .output(playlistPath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
    
    // Upload playlist and segments
    const manifestBuffer = await this.readFileToBuffer(playlistPath);
    const manifestKey = this.generateProcessedKey(job.key, 'hls/playlist', 'm3u8');
    
    const manifestUpload = await this.uploadFile(manifestBuffer, manifestKey, {
      contentType: 'application/vnd.apple.mpegurl',
      metadata: {
        'original-key': job.key,
        'content-type': 'hls-manifest',
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });
    
    // Upload segments
    const segmentKeys = [];
    const segmentFiles = await this.getFilesInDirectory(hlsDir, '.ts');
    
    for (const segmentFile of segmentFiles) {
      const segmentBuffer = await this.readFileToBuffer(segmentFile);
      const segmentName = segmentFile.split('/').pop();
      const segmentKey = this.generateProcessedKey(job.key, `hls/${segmentName}`, 'ts');
      
      await this.uploadFile(segmentBuffer, segmentKey, {
        contentType: 'video/mp2t',
        metadata: {
          'original-key': job.key,
          'content-type': 'hls-segment',
          'tenant-id': job.tenantId,
          'processing-job-id': job.jobId
        }
      });
      
      segmentKeys.push(segmentKey);
    }
    
    // Clean up HLS directory
    this.cleanupDirectory(hlsDir);
    
    return {
      manifestKey,
      manifestUrl: manifestUpload.url,
      segmentKeys,
      segmentCount: segmentKeys.length,
      segmentDuration: this.hlsConfig.segmentDuration
    };
  }

  /**
   * Select appropriate transcoding profiles based on input video
   * @param {Object} metadata - Video metadata
   * @returns {Array} Array of profile names
   */
  selectTranscodingProfiles(metadata) {
    const profiles = [];
    const inputHeight = metadata.height;
    
    // Always include 480p for compatibility
    profiles.push('480p');
    
    // Add 720p if input is higher resolution
    if (inputHeight >= 720) {
      profiles.push('720p');
    }
    
    // Add 1080p if input is higher resolution
    if (inputHeight >= 1080) {
      profiles.push('1080p');
    }
    
    return profiles;
  }

  /**
   * Parse frame rate string
   * @param {string} frameRate - Frame rate string (e.g., "30/1")
   * @returns {number} Frame rate as number
   */
  parseFPS(frameRate) {
    if (!frameRate) return 0;
    
    const parts = frameRate.split('/');
    if (parts.length === 2) {
      return Math.round(parseInt(parts[0]) / parseInt(parts[1]));
    }
    
    return parseInt(frameRate) || 0;
  }

  /**
   * Write buffer to file
   * @param {Buffer} buffer - Buffer to write
   * @param {string} filePath - File path
   * @returns {Promise<void>}
   */
  async writeBufferToFile(buffer, filePath) {
    return new Promise((resolve, reject) => {
      const stream = createWriteStream(filePath);
      stream.write(buffer);
      stream.end();
      stream.on('finish', resolve);
      stream.on('error', reject);
    });
  }

  /**
   * Read file to buffer
   * @param {string} filePath - File path
   * @returns {Promise<Buffer>} File buffer
   */
  async readFileToBuffer(filePath) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      const stream = createReadStream(filePath);
      
      stream.on('data', chunk => chunks.push(chunk));
      stream.on('end', () => resolve(Buffer.concat(chunks)));
      stream.on('error', reject);
    });
  }

  /**
   * Get file extension from path
   * @param {string} filePath - File path
   * @returns {string} File extension
   */
  getFileExtension(filePath) {
    return filePath.split('.').pop()?.toLowerCase() || 'mp4';
  }

  /**
   * Clean up temporary files
   * @param {Array} files - Array of file paths to clean up
   */
  cleanupTempFiles(files) {
    for (const file of files) {
      try {
        if (file && require('fs').existsSync(file)) {
          unlinkSync(file);
        }
      } catch (error) {
        console.error(`Failed to cleanup temp file ${file}:`, error);
      }
    }
  }

  /**
   * Clean up directory
   * @param {string} dirPath - Directory path
   */
  cleanupDirectory(dirPath) {
    try {
      const fs = require('fs');
      if (fs.existsSync(dirPath)) {
        const files = fs.readdirSync(dirPath);
        for (const file of files) {
          fs.unlinkSync(join(dirPath, file));
        }
        fs.rmdirSync(dirPath);
      }
    } catch (error) {
      console.error(`Failed to cleanup directory ${dirPath}:`, error);
    }
  }

  /**
   * Get files in directory with extension
   * @param {string} dirPath - Directory path
   * @param {string} extension - File extension
   * @returns {Promise<Array>} Array of file paths
   */
  async getFilesInDirectory(dirPath, extension) {
    const fs = require('fs');
    const files = fs.readdirSync(dirPath);
    return files
      .filter(file => file.endsWith(extension))
      .map(file => join(dirPath, file));
  }

  /**
   * Validate video file
   * @param {Object} job - Processing job
   * @returns {boolean} Validation result
   */
  validateFile(job) {
    super.validateFile(job);
    
    const videoExtensions = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v'];
    const extension = job.key.split('.').pop()?.toLowerCase();
    
    if (!videoExtensions.includes(extension)) {
      throw new Error(`Unsupported video format: ${extension}`);
    }
    
    return true;
  }

  /**
   * Get processing capabilities
   * @returns {Object} Processor capabilities
   */
  getCapabilities() {
    return {
      supportedFormats: ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v'],
      outputFormats: ['mp4', 'webm'],
      transcodingProfiles: Object.keys(this.transcodingProfiles),
      hlsSupport: true,
      thumbnailGeneration: true,
      features: [
        'multi-resolution-transcoding',
        'hls-streaming',
        'thumbnail-extraction',
        'metadata-extraction',
        'progress-tracking'
      ]
    };
  }
}

export default VideoProcessor;