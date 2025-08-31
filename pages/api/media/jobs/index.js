import JobStatusTracker from '../../../../lib/jobs/JobStatusTracker.js';

/**
 * API endpoint for job management
 * GET /api/media/jobs - List jobs with filtering
 * POST /api/media/jobs/retry - Retry multiple failed jobs
 * DELETE /api/media/jobs/cleanup - Clean up old jobs
 */

const jobTracker = new JobStatusTracker();

export default async function handler(req, res) {
  try {
    switch (req.method) {
      case 'GET':
        return await handleListJobs(req, res);
      case 'POST':
        return await handleBatchOperations(req, res);
      case 'DELETE':
        return await handleCleanup(req, res);
      default:
        return res.status(405).json({ error: 'Method not allowed' });
    }
  } catch (error) {
    console.error('Error in jobs API:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
}

/**
 * List jobs with filtering and pagination
 */
async function handleListJobs(req, res) {
  const {
    status = 'all',
    limit = 50,
    offset = 0,
    includeDetails = 'false',
    fileType,
    tenantId
  } = req.query;

  const options = {
    limit: Math.min(parseInt(limit), 100), // Max 100 items
    offset: parseInt(offset),
    includeDetails: includeDetails === 'true'
  };

  let jobs = [];
  
  if (status === 'all') {
    // Get jobs from all statuses
    const statuses = ['pending', 'processing', 'completed', 'failed', 'retrying'];
    const allJobs = [];
    
    for (const s of statuses) {
      const statusJobs = await jobTracker.getJobsByStatus(s, options);
      if (options.includeDetails) {
        allJobs.push(...statusJobs);
      } else {
        allJobs.push(...statusJobs.map(jobId => ({ jobId, status: s })));
      }
    }
    
    jobs = allJobs.slice(options.offset, options.offset + options.limit);
  } else {
    jobs = await jobTracker.getJobsByStatus(status, options);
  }

  // Filter by file type if specified
  if (fileType && options.includeDetails) {
    jobs = jobs.filter(job => job.fileType === fileType);
  }

  // Filter by tenant ID if specified
  if (tenantId && options.includeDetails) {
    jobs = jobs.filter(job => job.tenantId === tenantId);
  }

  // Get statistics
  const statistics = await jobTracker.getJobStatistics();

  return res.status(200).json({
    jobs,
    pagination: {
      limit: options.limit,
      offset: options.offset,
      total: jobs.length
    },
    statistics
  });
}

/**
 * Handle batch operations
 */
async function handleBatchOperations(req, res) {
  const { operation, ...options } = req.body;

  switch (operation) {
    case 'retry':
      return await handleBatchRetry(req, res, options);
    case 'process':
      return await handleBatchProcess(req, res, options);
    default:
      return res.status(400).json({ error: 'Invalid operation' });
  }
}

/**
 * Retry multiple failed jobs
 */
async function handleBatchRetry(req, res, options) {
  const {
    maxAge = 24 * 60 * 60 * 1000, // 24 hours
    limit = 100,
    fileType = null
  } = options;

  const retriedJobs = await jobTracker.retryFailedJobs({
    maxAge,
    limit,
    fileType
  });

  return res.status(200).json({
    message: `Retried ${retriedJobs.length} failed jobs`,
    retriedJobs,
    count: retriedJobs.length
  });
}

/**
 * Process jobs manually (for testing/debugging)
 */
async function handleBatchProcess(req, res, options) {
  const { jobIds } = options;

  if (!Array.isArray(jobIds) || jobIds.length === 0) {
    return res.status(400).json({ error: 'Job IDs array is required' });
  }

  if (jobIds.length > 10) {
    return res.status(400).json({ error: 'Maximum 10 jobs can be processed at once' });
  }

  const results = [];
  const MediaProcessor = (await import('../../../../workers/MediaProcessor.js')).default;
  const mediaProcessor = new MediaProcessor();

  for (const jobId of jobIds) {
    try {
      const result = await mediaProcessor.processJobById(jobId);
      results.push({ jobId, status: 'success', result });
    } catch (error) {
      results.push({ jobId, status: 'error', error: error.message });
    }
  }

  const successCount = results.filter(r => r.status === 'success').length;
  const errorCount = results.filter(r => r.status === 'error').length;

  return res.status(200).json({
    message: `Processed ${successCount} jobs successfully, ${errorCount} failed`,
    results,
    summary: {
      total: results.length,
      success: successCount,
      errors: errorCount
    }
  });
}

/**
 * Clean up old jobs
 */
async function handleCleanup(req, res) {
  const { olderThanDays = 7 } = req.query;
  
  const cleanedCount = await jobTracker.cleanupOldJobs(parseInt(olderThanDays));
  
  return res.status(200).json({
    message: `Cleaned up ${cleanedCount} old jobs`,
    cleanedCount,
    olderThanDays: parseInt(olderThanDays)
  });
}