import axios from 'axios';

function parseArg(name, def) {
  const val = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (!val) return def;
  const [, v] = val.split('=');
  return v;
}

function toNumber(v, d) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

async function run() {
  const BASE = process.env.STRESS_BASE_URL || parseArg('base', null) || 'http://localhost:3005';
  const total = toNumber(parseArg('total', process.env.TOTAL || 1000), 1000);
  const concurrency = Math.max(1, toNumber(parseArg('concurrency', process.env.CONCURRENCY || 100), 100));
  const mode = (parseArg('mode', process.env.MODE || 'mixed') || 'mixed').toLowerCase(); // read|write|mixed
  const timeout = toNumber(parseArg('timeout', process.env.TIMEOUT || 20000), 20000);

  console.log(`HTTP CRUD Stress
  Base:         ${BASE}
  Requests:     ${total}
  Concurrency:  ${concurrency}
  Mode:         ${mode}
  Timeout:      ${timeout}ms
  `);

  const latencies = [];
  let success = 0;
  let failures = 0;

  const tasks = Array.from({ length: total }, (_, i) => i);
  let active = 0;
  let index = 0;

  const startedAll = process.hrtime.bigint();

  async function worker() {
    while (true) {
      const myIndex = index++;
      if (myIndex >= tasks.length) break;
      active++;

      const start = process.hrtime.bigint();
      try {
        const url = `${BASE}/api/diagnostics/benchmark?mode=${encodeURIComponent(mode)}&ops=1&allowWrites=true`;
        const res = await axios.get(url, { timeout, validateStatus: () => true });
        const end = process.hrtime.bigint();
        const durMs = Number(end - start) / 1e6;

        if (res.status === 200 && res.data?.success) {
          success++;
          latencies.push(durMs);
        } else {
          failures++;
          latencies.push(durMs);
        }
      } catch (e) {
        const end = process.hrtime.bigint();
        const durMs = Number(end - start) / 1e6;
        failures++;
        latencies.push(durMs);
      } finally {
        active--;
      }
    }
  }

  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);

  const endedAll = process.hrtime.bigint();
  const totalDurationMs = Math.max(1, Math.round(Number(endedAll - startedAll) / 1e6));
  const sorted = latencies.slice().sort((a, b) => a - b);
  const p50 = Math.round(percentile(sorted, 50));
  const p95 = Math.round(percentile(sorted, 95));
  const p99 = Math.round(percentile(sorted, 99));
  const min = Math.round(sorted[0] || 0);
  const max = Math.round(sorted[sorted.length - 1] || 0);
  const avg = Math.round(sorted.reduce((s, v) => s + v, 0) / (sorted.length || 1));
  const rps = Math.round((success + failures) / (totalDurationMs / 1000));

  console.log('Results:');
  console.log(JSON.stringify({
    totalRequests: total,
    success,
    failures,
    totalDurationMs,
    rps,
    latencyMs: { min, p50, p95, p99, max, avg },
  }, null, 2));

  // Exit non-zero if any failures
  process.exit(failures === 0 ? 0 : 2);
}

run().catch((e) => {
  console.error('Stress run error:', e?.message || e);
  process.exit(2);
});
