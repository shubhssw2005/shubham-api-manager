import BaseProcessor from './BaseProcessor.js';
import ffmpeg from 'fluent-ffmpeg';
import { createWriteStream, createReadStream, unlinkSync, mkdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

/**
 * Audio Processor - Handles audio processing including format conversion and metadata extraction
 */
class AudioProcessor extends BaseProcessor {
  constructor() {
    super();
    
    // Audio processing configurations
    this.transcodingProfiles = {
      'mp3_128': {
        format: 'mp3',
        bitrate: '128k',
        codec: 'libmp3lame',
        quality: 2
      },
      'mp3_320': {
        format: 'mp3',
        bitrate: '320k',
        codec: 'libmp3lame',
        quality: 0
      },
      'aac_128': {
        format: 'aac',
        bitrate: '128k',
        codec: 'aac',
        quality: 2
      },
      'ogg_128': {
        format: 'ogg',
        bitrate: '128k',
        codec: 'libvorbis',
        quality: 4
      }
    };
    
    // Waveform generation config
    this.waveformConfig = {
      width: 800,
      height: 200,
      color: '#3498db',
      backgroundColor: 'transparent',
      samples: 1000
    };
    
    // Temporary directory for processing
    this.tempDir = join(tmpdir(), 'audio-processing');
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
   * Process audio file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async process(job) {
    const startTime = Date.now();
    let tempFiles = [];
    
    try {
      // Validate file
      this.validateFile(job);
      
      // Download original audio
      console.log(`Downloading audio: ${job.key}`);
      const audioBuffer = await this.downloadFile(job.bucket, job.key);
      
      // Save to temporary file
      const extension = this.getFileExtension(job.key);
      const tempInputPath = join(this.tempDir, `input_${job.jobId}.${extension}`);
      await this.writeBufferToFile(audioBuffer, tempInputPath);
      tempFiles.push(tempInputPath);
      
      // Process audio with timeout
      const result = await this.executeWithTimeout(
        this.processAudio(tempInputPath, job)
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
   * Process audio file
   * @param {string} inputPath - Path to input audio file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async processAudio(inputPath, job) {
    // Get audio metadata
    const metadata = await this.getAudioMetadata(inputPath);
    console.log(`Processing audio: ${metadata.duration}s, ${metadata.bitrate} bitrate`);
    
    const results = {
      metadata: {
        duration: metadata.duration,
        bitrate: metadata.bitrate,
        sampleRate: metadata.sampleRate,
        channels: metadata.channels,
        codec: metadata.codec,
        format: metadata.format
      },
      transcoded: [],
      waveform: null,
      outputFiles: []
    };

    // Generate waveform visualization
    try {
      const waveform = await this.generateWaveform(inputPath, job);
      results.waveform = waveform;
      results.outputFiles.push(waveform.key);
    } catch (error) {
      console.error('Failed to generate waveform:', error);
    }

    // Determine which transcoding profiles to use
    const profilesToUse = this.selectTranscodingProfiles(metadata);
    
    // Transcode to different formats/bitrates
    for (const profileName of profilesToUse) {
      try {
        const profile = this.transcodingProfiles[profileName];
        const transcoded = await this.transcodeAudio(inputPath, profile, profileName, job);
        results.transcoded.push(transcoded);
        results.outputFiles.push(transcoded.key);
      } catch (error) {
        console.error(`Failed to transcode to ${profileName}:`, error);
      }
    }

    return results;
  }

  /**
   * Get audio metadata using ffprobe
   * @param {string} inputPath - Path to audio file
   * @returns {Promise<Object>} Audio metadata
   */
  async getAudioMetadata(inputPath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) {
          reject(new Error(`Failed to get audio metadata: ${err.message}`));
          return;
        }
        
        const audioStream = metadata.streams.find(stream => stream.codec_type === 'audio');
        
        if (!audioStream) {
          reject(new Error('No audio stream found'));
          return;
        }
        
        resolve({
          duration: parseFloat(metadata.format.duration),
          bitrate: parseInt(metadata.format.bit_rate) || parseInt(audioStream.bit_rate),
          sampleRate: parseInt(audioStream.sample_rate),
          channels: audioStream.channels,
          codec: audioStream.codec_name,
          format: metadata.format.format_name,
          fileSize: parseInt(metadata.format.size)
        });
      });
    });
  }

  /**
   * Generate waveform visualization
   * @param {string} inputPath - Path to input audio
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Waveform info
   */
  async generateWaveform(inputPath, job) {
    const waveformPath = join(this.tempDir, `waveform_${job.jobId}.png`);
    
    // Generate waveform using ffmpeg
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .complexFilter([
          `[0:a]showwavespic=s=${this.waveformConfig.width}x${this.waveformConfig.height}:colors=${this.waveformConfig.color}[v]`
        ])
        .map('[v]')
        .format('image2')
        .output(waveformPath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
    
    // Upload waveform
    const waveformBuffer = await this.readFileToBuffer(waveformPath);
    const waveformKey = this.generateProcessedKey(job.key, 'waveform', 'png');
    
    const uploadResult = await this.uploadFile(waveformBuffer, waveformKey, {
      contentType: 'image/png',
      metadata: {
        'original-key': job.key,
        'visualization-type': 'waveform',
        'width': this.waveformConfig.width.toString(),
        'height': this.waveformConfig.height.toString(),
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });
    
    // Clean up temp file
    unlinkSync(waveformPath);
    
    return {
      key: waveformKey,
      url: uploadResult.url,
      width: this.waveformConfig.width,
      height: this.waveformConfig.height,
      fileSize: waveformBuffer.length,
      format: 'png'
    };
  }

  /**
   * Transcode audio to specific profile
   * @param {string} inputPath - Input audio path
   * @param {Object} profile - Transcoding profile
   * @param {string} profileName - Profile name
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Transcoded audio info
   */
  async transcodeAudio(inputPath, profile, profileName, job) {
    const outputPath = join(this.tempDir, `transcoded_${job.jobId}_${profileName}.${profile.format}`);
    
    await new Promise((resolve, reject) => {
      let command = ffmpeg(inputPath)
        .audioCodec(profile.codec)
        .audioBitrate(profile.bitrate)
        .format(profile.format);
      
      // Add quality settings if specified
      if (profile.quality !== undefined) {
        if (profile.codec === 'libmp3lame') {
          command = command.audioQuality(profile.quality);
        } else if (profile.codec === 'libvorbis') {
          command = command.addOption('-qscale:a', profile.quality);
        }
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
    
    // Upload transcoded audio
    const transcodedBuffer = await this.readFileToBuffer(outputPath);
    const transcodedKey = this.generateProcessedKey(job.key, profileName, profile.format);
    
    const uploadResult = await this.uploadFile(transcodedBuffer, transcodedKey, {
      contentType: `audio/${profile.format}`,
      metadata: {
        'original-key': job.key,
        'transcoding-profile': profileName,
        'bitrate': profile.bitrate,
        'codec': profile.codec,
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
      bitrate: profile.bitrate,
      format: profile.format,
      codec: profile.codec,
      fileSize: transcodedBuffer.length
    };
  }

  /**
   * Select appropriate transcoding profiles based on input audio
   * @param {Object} metadata - Audio metadata
   * @returns {Array} Array of profile names
   */
  selectTranscodingProfiles(metadata) {
    const profiles = [];
    const inputBitrate = metadata.bitrate;
    
    // Always include MP3 128k for compatibility
    profiles.push('mp3_128');
    
    // Add higher quality MP3 if input has sufficient quality
    if (inputBitrate >= 256000) {
      profiles.push('mp3_320');
    }
    
    // Add AAC for modern compatibility
    profiles.push('aac_128');
    
    // Add OGG for open source compatibility
    profiles.push('ogg_128');
    
    return profiles;
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
    return filePath.split('.').pop()?.toLowerCase() || 'mp3';
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
   * Validate audio file
   * @param {Object} job - Processing job
   * @returns {boolean} Validation result
   */
  validateFile(job) {
    super.validateFile(job);
    
    const audioExtensions = ['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a', 'wma'];
    const extension = job.key.split('.').pop()?.toLowerCase();
    
    if (!audioExtensions.includes(extension)) {
      throw new Error(`Unsupported audio format: ${extension}`);
    }
    
    return true;
  }

  /**
   * Get processing capabilities
   * @returns {Object} Processor capabilities
   */
  getCapabilities() {
    return {
      supportedFormats: ['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a', 'wma'],
      outputFormats: ['mp3', 'aac', 'ogg'],
      transcodingProfiles: Object.keys(this.transcodingProfiles),
      waveformGeneration: true,
      features: [
        'multi-format-transcoding',
        'waveform-visualization',
        'metadata-extraction',
        'bitrate-optimization',
        'quality-conversion'
      ]
    };
  }
}

export default AudioProcessor;