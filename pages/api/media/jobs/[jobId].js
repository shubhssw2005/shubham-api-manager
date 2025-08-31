import JobStatusTracker from '../../../../lib/jobs/JobStatusTracker.js';
import MediaProcessor from '../../../../workers/MediaProcessor.js';

/**
 * API endpoint for job status management
 * GET /api/media/jobs/[jobId] - Get job status
 * POST /api/media/jobs/[jobId] - Retry job
 * DELETE /api/media/jobs/[jobId] - Cancel job
 */

const jobTracker = new JobStatusTracker();

export default async function handler(req, res) {
  const { jobId } = req.query;

  if (!jobId) {
    return res.status(400).json({ error: 'Job ID is required' });
  }

  try {
    switch (req.method) {
      case 'GET':
        return await handleGetJobStatus(req, res, jobId);
      case 'POST':
        return await handleRetryJob(req, res, jobId);
      case 'DELETE':
        return await handleCancelJob(req, res, jobId);
      default:
        return res.status(405).json({ error: 'Method not allowed' });
    }
  } catch (error) {
    console.error(`Error handling job ${jobId}:`, error);
    return res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
}

/**
 * Get job status and details
 */
async function handleGetJobStatus(req, res, jobId) {
  const jobStatus = await jobTracker.getJobStatus(jobId);
  
  if (!jobStatus) {
    return res.status(404).json({ error: 'Job not found' });
  }

  // Get job timeline if requested
  const includeTimeline = req.query.timeline === 'true';
  let timeline = null;
  
  if (includeTimeline) {
    timeline = await jobTracker.getJobTimeline(jobId);
  }

  return res.status(200).json({
    job: jobStatus,
    timeline
  });
}

/**
 * Retry a failed job
 */
async function handleRetryJob(req, res, jobId) {
  const jobStatus = await jobTracker.getJobStatus(jobId);
  
  if (!jobStatus) {
    return res.status(404).json({ error: 'Job not found' });
  }

  if (jobStatus.status !== 'failed') {
    return res.status(400).json({ 
      error: 'Only failed jobs can be retried',
      currentStatus: jobStatus.status 
    });
  }

  // Reset job status for retry
  await jobTracker.updateJobStatus(jobId, 'pending', {
    retryCount: 0,
    retriedAt: new Date().toISOString(),
    retriedBy: 'manual',
    previousError: jobStatus.error
  });

  // If manual processing is requested, process immediately
  if (req.body.processNow === true) {
    try {
      const mediaProcessor = new MediaProcessor();
      const result = await mediaProcessor.processJobById(jobId);
      
      return res.status(200).json({
        message: 'Job retried and processed successfully',
        jobId,
        result
      });
    } catch (error) {
      return res.status(500).json({
        message: 'Job queued for retry but immediate processing failed',
        jobId,
        error: error.message
      });
    }
  }

  return res.status(200).json({
    message: 'Job queued for retry',
    jobId,
    status: 'pending'
  });
}

/**
 * Cancel a job (mark as cancelled)
 */
async function handleCancelJob(req, res, jobId) {
  const jobStatus = await jobTracker.getJobStatus(jobId);
  
  if (!jobStatus) {
    return res.status(404).json({ error: 'Job not found' });
  }

  if (['completed', 'failed'].includes(jobStatus.status)) {
    return res.status(400).json({ 
      error: 'Cannot cancel completed or failed jobs',
      currentStatus: jobStatus.status 
    });
  }

  // Mark job as cancelled
  await jobTracker.updateJobStatus(jobId, 'cancelled', {
    cancelledAt: new Date().toISOString(),
    cancelledBy: 'manual'
  });

  return res.status(200).json({
    message: 'Job cancelled successfully',
    jobId,
    status: 'cancelled'
  });
}