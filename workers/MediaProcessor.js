import { SQSClient, ReceiveMessageCommand, DeleteMessageCommand, ChangeMessageVisibilityCommand } from '@aws-sdk/client-sqs';
import S3EventProcessor from '../lib/events/S3EventProcessor.js';
import ImageProcessor from './processors/ImageProcessor.js';
import VideoProcessor from './processors/VideoProcessor.js';
import DocumentProcessor from './processors/DocumentProcessor.js';
import AudioProcessor from './processors/AudioProcessor.js';
import JobStatusTracker from '../lib/jobs/JobStatusTracker.js';

/**
 * Media Processing Worker - Processes media files from SQS queue
 */
class MediaProcessor {
  constructor() {
    this.eventProcessor = new S3EventProcessor();
    this.jobTracker = new JobStatusTracker();
    
    // Initialize processors
    this.processors = {
      'image': new ImageProcessor(),
      'video': new VideoProcessor(),
      'document': new DocumentProcessor(),
      'audio': new AudioProcessor()
    };

    // Worker configuration
    this.isRunning = false;
    this.concurrency = parseInt(process.env.MEDIA_PROCESSOR_CONCURRENCY) || 5;
    this.pollInterval = parseInt(process.env.POLL_INTERVAL) || 1000; // 1 second
    this.maxRetries = parseInt(process.env.MAX_PROCESSING_RETRIES) || 3;
    
    // Metrics
    this.metrics = {
      processed: 0,
      failed: 0,
      retried: 0,
      startTime: null
    };
  }

  /**
   * Start the media processor worker
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isRunning) {
      console.log('Media processor already running');
      return;
    }

    this.isRunning = true;
    this.metrics.startTime = new Date();
    console.log(`Starting media processor with concurrency: ${this.concurrency}`);

    // Start multiple concurrent workers
    const workers = [];
    for (let i = 0; i < this.concurrency; i++) {
      workers.push(this.workerLoop(i));
    }

    // Start metrics reporting
    this.startMetricsReporting();

    // Wait for all workers to complete
    await Promise.all(workers);
  }

  /**
   * Stop the media processor
   */
  stop() {
    console.log('Stopping media processor...');
    this.isRunning = false;
  }

  /**
   * Main worker loop
   * @param {number} workerId - Worker identifier
   * @returns {Promise<void>}
   */
  async workerLoop(workerId) {
    console.log(`Worker ${workerId} started`);

    while (this.isRunning) {
      try {
        const messages = await this.eventProcessor.receiveMessages();
        
        if (messages.length > 0) {
          console.log(`Worker ${workerId} received ${messages.length} messages`);
          
          // Process messages concurrently within this worker
          await Promise.all(
            messages.map(message => this.processMessage(message, workerId))
          );
        } else {
          // No messages, wait before polling again
          await this.sleep(this.pollInterval);
        }
      } catch (error) {
        console.error(`Worker ${workerId} error:`, error);
        await this.sleep(5000); // Wait longer on error
      }
    }

    console.log(`Worker ${workerId} stopped`);
  }

  /**
   * Process a single SQS message
   * @param {Object} message - SQS message
   * @param {number} workerId - Worker identifier
   * @returns {Promise<void>}
   */
  async processMessage(message, workerId) {
    let job;
    
    try {
      job = JSON.parse(message.Body);
      console.log(`Worker ${workerId} processing job ${job.jobId} (${job.fileType})`);

      // Update job status to processing
      await this.jobTracker.updateJobStatus(job.jobId, 'processing', {
        workerId,
        startedAt: new Date().toISOString()
      });

      // Get the appropriate processor
      const processor = this.processors[job.fileType];
      if (!processor) {
        throw new Error(`No processor available for file type: ${job.fileType}`);
      }

      // Process the media file
      const result = await processor.process(job);

      // Update job status to completed
      await this.jobTracker.updateJobStatus(job.jobId, 'completed', {
        completedAt: new Date().toISOString(),
        result,
        workerId
      });

      // Delete message from queue
      await this.eventProcessor.deleteMessage(message.ReceiptHandle);

      // Publish completion notification
      await this.eventProcessor.publishProcessingNotification(job, 'processed');

      this.metrics.processed++;
      console.log(`Worker ${workerId} completed job ${job.jobId}`);

    } catch (error) {
      console.error(`Worker ${workerId} failed to process job:`, error);
      
      if (job) {
        await this.handleProcessingError(job, message, error, workerId);
      }
      
      this.metrics.failed++;
    }
  }

  /**
   * Handle processing errors with retry logic
   * @param {Object} job - Processing job
   * @param {Object} message - SQS message
   * @param {Error} error - Processing error
   * @param {number} workerId - Worker identifier
   * @returns {Promise<void>}
   */
  async handleProcessingError(job, message, error, workerId) {
    const retryCount = job.retryCount || 0;
    const errorMessage = error.message || 'Unknown error';

    // Update job status to failed
    await this.jobTracker.updateJobStatus(job.jobId, 'failed', {
      error: errorMessage,
      retryCount,
      failedAt: new Date().toISOString(),
      workerId
    });

    if (retryCount < this.maxRetries) {
      // Retry the job
      console.log(`Retrying job ${job.jobId} (attempt ${retryCount + 1}/${this.maxRetries})`);
      
      // Increment retry count
      job.retryCount = retryCount + 1;
      
      // Calculate exponential backoff delay
      const delaySeconds = Math.min(300, Math.pow(2, retryCount) * 30); // Max 5 minutes
      
      // Change message visibility to delay retry
      await this.changeMessageVisibility(message.ReceiptHandle, delaySeconds);
      
      // Update job status to retrying
      await this.jobTracker.updateJobStatus(job.jobId, 'retrying', {
        retryCount: job.retryCount,
        nextRetryAt: new Date(Date.now() + delaySeconds * 1000).toISOString(),
        workerId
      });

      this.metrics.retried++;
    } else {
      // Max retries exceeded, send to dead letter queue
      console.log(`Max retries exceeded for job ${job.jobId}, sending to DLQ`);
      
      await this.eventProcessor.sendToDeadLetterQueue(job, errorMessage);
      await this.eventProcessor.deleteMessage(message.ReceiptHandle);
      
      // Publish failure notification
      await this.eventProcessor.publishProcessingNotification(job, 'failed');
    }
  }

  /**
   * Change message visibility timeout
   * @param {string} receiptHandle - Message receipt handle
   * @param {number} visibilityTimeout - New visibility timeout in seconds
   * @returns {Promise<void>}
   */
  async changeMessageVisibility(receiptHandle, visibilityTimeout) {
    const command = new ChangeMessageVisibilityCommand({
      QueueUrl: process.env.MEDIA_PROCESSING_QUEUE_URL,
      ReceiptHandle: receiptHandle,
      VisibilityTimeout: visibilityTimeout
    });

    await this.eventProcessor.sqsClient.send(command);
  }

  /**
   * Start metrics reporting
   */
  startMetricsReporting() {
    const reportInterval = parseInt(process.env.METRICS_REPORT_INTERVAL) || 60000; // 1 minute
    
    setInterval(() => {
      if (this.isRunning) {
        this.reportMetrics();
      }
    }, reportInterval);
  }

  /**
   * Report processing metrics
   */
  reportMetrics() {
    const uptime = Date.now() - this.metrics.startTime.getTime();
    const uptimeMinutes = Math.floor(uptime / 60000);
    
    console.log('=== Media Processor Metrics ===');
    console.log(`Uptime: ${uptimeMinutes} minutes`);
    console.log(`Processed: ${this.metrics.processed}`);
    console.log(`Failed: ${this.metrics.failed}`);
    console.log(`Retried: ${this.metrics.retried}`);
    console.log(`Success Rate: ${this.getSuccessRate()}%`);
    console.log(`Processing Rate: ${this.getProcessingRate()} jobs/min`);
    console.log('==============================');
  }

  /**
   * Calculate success rate
   * @returns {number} Success rate percentage
   */
  getSuccessRate() {
    const total = this.metrics.processed + this.metrics.failed;
    if (total === 0) return 100;
    return Math.round((this.metrics.processed / total) * 100);
  }

  /**
   * Calculate processing rate
   * @returns {number} Jobs per minute
   */
  getProcessingRate() {
    if (!this.metrics.startTime) return 0;
    
    const uptime = Date.now() - this.metrics.startTime.getTime();
    const uptimeMinutes = uptime / 60000;
    
    if (uptimeMinutes === 0) return 0;
    
    return Math.round(this.metrics.processed / uptimeMinutes);
  }

  /**
   * Get current metrics
   * @returns {Object} Current metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      uptime: this.metrics.startTime ? Date.now() - this.metrics.startTime.getTime() : 0,
      successRate: this.getSuccessRate(),
      processingRate: this.getProcessingRate(),
      isRunning: this.isRunning,
      concurrency: this.concurrency
    };
  }

  /**
   * Process specific job by ID (for manual processing)
   * @param {string} jobId - Job identifier
   * @returns {Promise<Object>} Processing result
   */
  async processJobById(jobId) {
    const jobStatus = await this.jobTracker.getJobStatus(jobId);
    if (!jobStatus) {
      throw new Error(`Job ${jobId} not found`);
    }

    const job = jobStatus.job;
    const processor = this.processors[job.fileType];
    
    if (!processor) {
      throw new Error(`No processor available for file type: ${job.fileType}`);
    }

    // Update status to processing
    await this.jobTracker.updateJobStatus(jobId, 'processing', {
      startedAt: new Date().toISOString(),
      manual: true
    });

    try {
      const result = await processor.process(job);
      
      await this.jobTracker.updateJobStatus(jobId, 'completed', {
        completedAt: new Date().toISOString(),
        result,
        manual: true
      });

      return result;
    } catch (error) {
      await this.jobTracker.updateJobStatus(jobId, 'failed', {
        error: error.message,
        failedAt: new Date().toISOString(),
        manual: true
      });
      
      throw error;
    }
  }

  /**
   * Sleep utility
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise<void>}
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Graceful shutdown
   * @returns {Promise<void>}
   */
  async shutdown() {
    console.log('Initiating graceful shutdown...');
    this.stop();
    
    // Wait for current processing to complete (max 30 seconds)
    const shutdownTimeout = 30000;
    const startTime = Date.now();
    
    while (this.isRunning && (Date.now() - startTime) < shutdownTimeout) {
      await this.sleep(1000);
    }
    
    console.log('Media processor shutdown complete');
  }
}

export default MediaProcessor;