import JobStatusTracker from '../../../../lib/jobs/JobStatusTracker.js';

/**
 * API endpoint for job statistics and metrics
 * GET /api/media/jobs/stats - Get comprehensive job statistics
 */

const jobTracker = new JobStatusTracker();

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { timeRange = '24h', detailed = 'false' } = req.query;

    // Get basic statistics
    const statistics = await jobTracker.getJobStatistics({ timeRange });
    
    // Get health check
    const healthCheck = await jobTracker.healthCheck();

    // Get detailed metrics if requested
    let detailedMetrics = null;
    if (detailed === 'true') {
      detailedMetrics = await getDetailedMetrics();
    }

    return res.status(200).json({
      statistics,
      healthCheck,
      detailedMetrics,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting job statistics:', error);
    return res.status(500).json({ 
      error: 'Failed to get job statistics',
      details: error.message 
    });
  }
}

/**
 * Get detailed metrics for monitoring
 */
async function getDetailedMetrics() {
  const statuses = ['pending', 'processing', 'completed', 'failed', 'retrying'];
  const fileTypes = ['image', 'video', 'audio', 'document'];
  
  const metrics = {
    byStatus: {},
    byFileType: {},
    processingTimes: {},
    errorRates: {}
  };

  // Get job counts by status
  for (const status of statuses) {
    const jobs = await jobTracker.getJobsByStatus(status, { limit: 1000, includeDetails: true });
    metrics.byStatus[status] = {
      count: jobs.length,
      jobs: jobs.slice(0, 10) // Include first 10 jobs for debugging
    };
  }

  // Get job counts by file type
  for (const fileType of fileTypes) {
    const allJobs = [];
    for (const status of statuses) {
      const jobs = await jobTracker.getJobsByStatus(status, { limit: 1000, includeDetails: true });
      allJobs.push(...jobs.filter(job => job.fileType === fileType));
    }
    
    metrics.byFileType[fileType] = {
      total: allJobs.length,
      completed: allJobs.filter(job => job.status === 'completed').length,
      failed: allJobs.filter(job => job.status === 'failed').length,
      processing: allJobs.filter(job => job.status === 'processing').length
    };
  }

  // Calculate average processing times
  const completedJobs = await jobTracker.getJobsByStatus('completed', { 
    limit: 100, 
    includeDetails: true 
  });
  
  for (const fileType of fileTypes) {
    const typeJobs = completedJobs.filter(job => job.fileType === fileType);
    const processingTimes = typeJobs
      .filter(job => job.startedAt && job.completedAt)
      .map(job => {
        const start = new Date(job.startedAt).getTime();
        const end = new Date(job.completedAt).getTime();
        return end - start;
      });
    
    if (processingTimes.length > 0) {
      metrics.processingTimes[fileType] = {
        average: Math.round(processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length),
        min: Math.min(...processingTimes),
        max: Math.max(...processingTimes),
        samples: processingTimes.length
      };
    }
  }

  // Calculate error rates
  for (const fileType of fileTypes) {
    const typeMetrics = metrics.byFileType[fileType];
    const total = typeMetrics.completed + typeMetrics.failed;
    
    if (total > 0) {
      metrics.errorRates[fileType] = {
        rate: Math.round((typeMetrics.failed / total) * 100 * 100) / 100, // 2 decimal places
        failed: typeMetrics.failed,
        total
      };
    }
  }

  return metrics;
}