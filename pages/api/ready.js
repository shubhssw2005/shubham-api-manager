import dbConnect, { getDatabaseHealth } from '../../lib/dbConnect.js';

function validateEnv() {
  // Minimal required for production DBs (tweak as needed)
  const required = ['SCYLLA_HOSTS', 'SCYLLA_KEYSPACE'];
  const recommended = ['SCYLLA_DATACENTER', 'FDB_API_VERSION', 'FDB_CLUSTER_FILE'];

  const missingRequired = required.filter((k) => !process.env[k]);
  const missingRecommended = recommended.filter((k) => !process.env[k]);

  // Detect if defaults are used (ok for dev, warn for prod)
  const usingDefaults = {
    scyllaHostsDefault:
      !process.env.SCYLLA_HOSTS || process.env.SCYLLA_HOSTS.split(',').join(',') === '127.0.0.1',
    fdbClusterFileDefault:
      !process.env.FDB_CLUSTER_FILE || process.env.FDB_CLUSTER_FILE === '/etc/foundationdb/fdb.cluster',
  };

  return { missingRequired, missingRecommended, usingDefaults };
}

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ success: false, message: `Method ${req.method} not allowed` });
  }

  try {
    await dbConnect();

    const dbHealth = await getDatabaseHealth();
    const envCheck = validateEnv();

    const dbHealthy = dbHealth?.overall === 'healthy';
    const noMissingRequired = envCheck.missingRequired.length === 0;

    const deploymentReady = dbHealthy && noMissingRequired;

    const advice = [];
    if (!dbHealthy) advice.push('Databases are not healthy. Check ScyllaDB/FoundationDB connectivity.');
    if (!noMissingRequired) advice.push(`Set required env vars: ${envCheck.missingRequired.join(', ')}`);
    if (envCheck.usingDefaults.scyllaHostsDefault)
      advice.push('SCYLLA_HOSTS is using default localhost; set production cluster hosts.');
    if (envCheck.usingDefaults.fdbClusterFileDefault)
      advice.push('FDB_CLUSTER_FILE is default; point to your production cluster file.');
    if (process.env.NODE_ENV !== 'production')
      advice.push('NODE_ENV is not production. For production, set NODE_ENV=production.');

    return res.status(deploymentReady ? 200 : 503).json({
      success: deploymentReady,
      deploymentReady,
      timestamp: new Date().toISOString(),
      env: process.env.NODE_ENV || 'development',
      databases: dbHealth,
      envCheck,
      advice,
    });
  } catch (error) {
    console.error('Readiness check error:', error);
    return res.status(500).json({ success: false, deploymentReady: false, error: error.message });
  }
}
