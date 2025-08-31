import dbConnect, { getDatabaseHealth } from '../../lib/dbConnect.js';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ success: false, message: `Method ${req.method} not allowed` });
  }

  try {
    // Ensure DBs are initialized (uses Scylla/Foundation, no mongoose)
    await dbConnect();

    const dbHealth = await getDatabaseHealth();
    const uptimeSeconds = Math.round(process.uptime());
    const memory = process.memoryUsage();

    const status =
      dbHealth.overall === 'healthy' ? 'ok' : dbHealth.overall === 'degraded' ? 'degraded' : 'error';

    res.setHeader('Cache-Control', 'no-store');
    return res.status(status === 'ok' ? 200 : 503).json({
      success: status === 'ok',
      status,
      timestamp: new Date().toISOString(),
      env: process.env.NODE_ENV || 'development',
      uptimeSeconds,
      memoryMB: {
        rss: Math.round(memory.rss / (1024 * 1024)),
        heapTotal: Math.round(memory.heapTotal / (1024 * 1024)),
        heapUsed: Math.round(memory.heapUsed / (1024 * 1024)),
        external: Math.round(memory.external / (1024 * 1024)),
      },
      databases: dbHealth,
    });
  } catch (error) {
    console.error('Health check error:', error);
    return res.status(500).json({
      success: false,
      status: 'error',
      error: error.message,
    });
  }
}
