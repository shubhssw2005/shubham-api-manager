import axios from 'axios';

const BASE = process.env.SMOKE_BASE_URL || 'http://localhost:3005';

async function main() {
  try {
    console.log(`Running smoke tests against: ${BASE}`);

    const health = await axios.get(`${BASE}/api/health`, { timeout: 10_000 });
    console.log('Health:', {
      status: health.data?.status,
      env: health.data?.env,
      uptimeSeconds: health.data?.uptimeSeconds,
      dbOverall: health.data?.databases?.overall,
    });

    const ready = await axios.get(`${BASE}/api/ready`, { timeout: 10_000, validateStatus: () => true });
    console.log('Readiness:', {
      deploymentReady: ready.data?.deploymentReady,
      env: ready.data?.env,
      dbOverall: ready.data?.databases?.overall,
      missingRequired: ready.data?.envCheck?.missingRequired,
      advice: ready.data?.advice,
      statusCode: ready.status,
    });

    const bench = await axios.get(`${BASE}/api/diagnostics/benchmark?mode=read&ops=200`, { timeout: 30_000 });
    console.log('Benchmark:', {
      mode: bench.data?.mode,
      opsExecuted: bench.data?.opsExecuted,
      durationMs: bench.data?.durationMs,
      opsPerSec: bench.data?.opsPerSec,
      notes: bench.data?.notes,
    });

    const ok = health.status === 200 && ready.status === 200;
    process.exit(ok ? 0 : 2);
  } catch (err) {
    console.error('Smoke tests failed:', err?.message || err);
    process.exit(2);
  }
}

main();
