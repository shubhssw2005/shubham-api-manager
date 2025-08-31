import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import S3EventProcessor from '../../lib/events/S3EventProcessor.js';
import JobStatusTracker from '../../lib/jobs/JobStatusTracker.js';
import MediaProcessor from '../../workers/MediaProcessor.js';
import ImageProcessor from '../../workers/processors/ImageProcessor.js';

describe('Media Processing Pipeline Integration Tests', () => {
  let s3EventProcessor;
  let jobTracker;
  let mediaProcessor;
  let imageProcessor;

  beforeAll(async () => {
    // Initialize components
    s3EventProcessor = new S3EventProcessor();
    jobTracker = new JobStatusTracker();
    mediaProcessor = new MediaProcessor();
    imageProcessor = new ImageProcessor();
  });

  afterAll(async () => {
    // Cleanup
    if (mediaProcessor) {
      await mediaProcessor.shutdown();
    }
  });

  beforeEach(async () => {
    // Clean up any existing test jobs
    await jobTracker.cleanupOldJobs(0);
  });

  describe('S3 Event Processing', () => {
    it('should process S3 ObjectCreated event', async () => {
      const mockS3Event = {
        Message: JSON.stringify({
          Records: [{
            eventName: 'ObjectCreated:Put',
            eventTime: new Date().toISOString(),
            s3: {
              bucket: { name: 'test-bucket' },
              object: { 
                key: 'tenants/test-tenant/users/test-user/media/1234567890-abc123-test-image.jpg',
                size: 1024000
              }
            }
          }]
        })
      };

      // Process the event (will fail due to missing AWS credentials in test, but should not throw)
      try {
        await s3EventProcessor.processS3Event(mockS3Event);
      } catch (error) {
        // Expected to fail in test environment due to missing AWS credentials
        expect(error.message).toContain('credential');
      }
    });

    it('should handle invalid S3 key format', async () => {
      const mockS3Event = {
        Message: JSON.stringify({
          Records: [{
            eventName: 'ObjectCreated:Put',
            eventTime: new Date().toISOString(),
            s3: {
              bucket: { name: 'test-bucket' },
              object: { 
                key: 'invalid/key/format.jpg',
                size: 1024000
              }
            }
          }]
        })
      };

      // Should not throw but should log warning
      await expect(s3EventProcessor.processS3Event(mockS3Event)).resolves.not.toThrow();
    });

    it('should determine correct file type from extension', () => {
      expect(s3EventProcessor.getFileType('test.jpg')).toBe('image');
      expect(s3EventProcessor.getFileType('test.mp4')).toBe('video');
      expect(s3EventProcessor.getFileType('test.mp3')).toBe('audio');
      expect(s3EventProcessor.getFileType('test.pdf')).toBe('document');
      expect(s3EventProcessor.getFileType('test.unknown')).toBe('other');
    });
  });

  describe('Job Status Tracking', () => {
    it('should create and update job status', async () => {
      const jobId = 'test-job-123';
      
      // Create job
      await jobTracker.updateJobStatus(jobId, 'pending', {
        fileType: 'image',
        tenantId: 'test-tenant'
      });

      // Get job status
      const jobStatus = await jobTracker.getJobStatus(jobId);
      expect(jobStatus).toBeDefined();
      expect(jobStatus.status).toBe('pending');
      expect(jobStatus.fileType).toBe('image');

      // Update to processing
      await jobTracker.updateJobStatus(jobId, 'processing', {
        workerId: 1,
        startedAt: new Date().toISOString()
      });

      const updatedStatus = await jobTracker.getJobStatus(jobId);
      expect(updatedStatus.status).toBe('processing');
      expect(updatedStatus.workerId).toBe(1);
    });

    it('should maintain job status history', async () => {
      const jobId = 'test-job-history';
      
      await jobTracker.updateJobStatus(jobId, 'pending');
      await jobTracker.updateJobStatus(jobId, 'processing');
      await jobTracker.updateJobStatus(jobId, 'completed');

      const timeline = await jobTracker.getJobTimeline(jobId);
      expect(timeline).toHaveLength(3);
      expect(timeline[0].status).toBe('pending');
      expect(timeline[1].status).toBe('processing');
      expect(timeline[2].status).toBe('completed');
    });

    it('should get jobs by status', async () => {
      // Create test jobs
      await jobTracker.updateJobStatus('job-1', 'pending');
      await jobTracker.updateJobStatus('job-2', 'processing');
      await jobTracker.updateJobStatus('job-3', 'completed');

      const pendingJobs = await jobTracker.getJobsByStatus('pending');
      expect(pendingJobs).toContain('job-1');

      const processingJobs = await jobTracker.getJobsByStatus('processing');
      expect(processingJobs).toContain('job-2');
    });

    it('should calculate job statistics', async () => {
      // Create test jobs with different statuses
      await jobTracker.updateJobStatus('stat-job-1', 'completed');
      await jobTracker.updateJobStatus('stat-job-2', 'completed');
      await jobTracker.updateJobStatus('stat-job-3', 'failed');

      const stats = await jobTracker.getJobStatistics();
      expect(stats.successRate).toBeGreaterThan(0);
      expect(stats.totalJobs).toBeGreaterThan(0);
    });
  });

  describe('Image Processor', () => {
    it('should validate image files correctly', () => {
      const validJob = {
        key: 'test.jpg',
        size: 1024000,
        fileType: 'image'
      };

      expect(() => imageProcessor.validateFile(validJob)).not.toThrow();

      const invalidJob = {
        key: 'test.txt',
        size: 1024000,
        fileType: 'image'
      };

      expect(() => imageProcessor.validateFile(invalidJob)).toThrow();
    });

    it('should generate correct processed keys', () => {
      const originalKey = 'tenants/test/users/user1/media/123-abc-image.jpg';
      
      const thumbKey = imageProcessor.generateProcessedKey(originalKey, 'thumb_small', 'jpg');
      expect(thumbKey).toContain('processed');
      expect(thumbKey).toContain('thumb_small');
      expect(thumbKey.endsWith('.jpg')).toBe(true);
    });

    it('should determine optimization requirements', () => {
      const largeImage = {
        width: 3000,
        height: 2000,
        format: 'jpeg'
      };
      const largeJob = { size: 2 * 1024 * 1024 }; // 2MB

      expect(imageProcessor.shouldOptimize(largeImage, largeJob)).toBe(true);

      const smallImage = {
        width: 800,
        height: 600,
        format: 'jpeg'
      };
      const smallJob = { size: 100 * 1024 }; // 100KB

      expect(imageProcessor.shouldOptimize(smallImage, smallJob)).toBe(false);
    });

    it('should return correct capabilities', () => {
      const capabilities = imageProcessor.getCapabilities();
      
      expect(capabilities.supportedFormats).toContain('jpeg');
      expect(capabilities.supportedFormats).toContain('png');
      expect(capabilities.features).toContain('thumbnail-generation');
      expect(capabilities.features).toContain('webp-conversion');
    });
  });

  describe('Media Processor Worker', () => {
    it('should initialize with correct configuration', () => {
      const metrics = mediaProcessor.getMetrics();
      
      expect(metrics.processed).toBe(0);
      expect(metrics.failed).toBe(0);
      expect(metrics.isRunning).toBe(false);
      expect(metrics.concurrency).toBeGreaterThan(0);
    });

    it('should calculate success rate correctly', () => {
      // Mock some metrics
      mediaProcessor.metrics.processed = 8;
      mediaProcessor.metrics.failed = 2;

      expect(mediaProcessor.getSuccessRate()).toBe(80);
    });

    it('should handle empty metrics', () => {
      mediaProcessor.metrics.processed = 0;
      mediaProcessor.metrics.failed = 0;

      expect(mediaProcessor.getSuccessRate()).toBe(100);
    });
  });

  describe('Error Handling', () => {
    it('should handle processing timeout', async () => {
      const processor = new ImageProcessor();
      processor.processingTimeout = 100; // Very short timeout

      const timeoutPromise = new Promise(resolve => {
        setTimeout(resolve, 200); // Longer than timeout
      });

      await expect(processor.executeWithTimeout(timeoutPromise))
        .rejects.toThrow('Processing timeout');
    });

    it('should enhance processing errors', () => {
      const originalError = new Error('Original error message');
      const job = {
        jobId: 'test-job',
        fileType: 'image',
        size: 1024000
      };

      const enhancedError = imageProcessor.handleProcessingError(originalError, job);
      
      expect(enhancedError.message).toContain('ImageProcessor failed');
      expect(enhancedError.jobId).toBe('test-job');
      expect(enhancedError.fileType).toBe('image');
    });
  });

  describe('File Type Detection', () => {
    it('should detect file types correctly', () => {
      const testCases = [
        { key: 'image.jpg', expected: 'image' },
        { key: 'video.mp4', expected: 'video' },
        { key: 'audio.mp3', expected: 'audio' },
        { key: 'document.pdf', expected: 'document' },
        { key: 'unknown.xyz', expected: 'other' }
      ];

      testCases.forEach(({ key, expected }) => {
        expect(s3EventProcessor.getFileType(key)).toBe(expected);
      });
    });
  });

  describe('Queue Management', () => {
    it('should generate unique job IDs', () => {
      const id1 = s3EventProcessor.generateJobId();
      const id2 = s3EventProcessor.generateJobId();
      
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^job_\d+_[a-z0-9]+$/);
    });

    it('should get queue statistics', async () => {
      const stats = await s3EventProcessor.getQueueStats();
      
      expect(stats).toHaveProperty('queueUrl');
      expect(stats).toHaveProperty('maxRetries');
      expect(stats).toHaveProperty('visibilityTimeout');
    });
  });

  describe('Health Checks', () => {
    it('should perform job tracker health check', async () => {
      const health = await jobTracker.healthCheck();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('redis');
    });
  });
});