import dbConnect, {
  findDocuments,
  createDocument,
  deleteDocument,
} from '../../../lib/dbConnect.js';

function parseBool(v, def = false) {
  if (v === undefined) return def;
  if (typeof v === 'boolean') return v;
  return ['1', 'true', 'yes', 'on'].includes(String(v).toLowerCase());
}

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ success: false, message: `Method ${req.method} not allowed` });
  }

  try {
    await dbConnect();

    const ops = Math.max(1, Math.min(parseInt(req.query.ops) || 100, 5000));
    const mode = (req.query.mode || 'read').toLowerCase(); // read | write | mixed
    const allowWrites = parseBool(req.query.allowWrites, false);
    const table = 'posts';

    const started = globalThis.process?.hrtime?.bigint?.() || BigInt(Date.now());

    let executed = 0;
    const notes = [];

    if (mode === 'read') {
      // Perform batched read (up to "ops" rows, limited by DB page size)
      const result = await findDocuments(table, Math.min(ops, 1000));
      executed = Array.isArray(result?.rows) ? result.rows.length : 0;
      notes.push('Read benchmark executed with a single paged query.');
    } else if (mode === 'write') {
      if (!allowWrites) {
        notes.push('Writes are disabled by default. Pass allowWrites=true to enable.');
      }
      for (let i = 0; i < ops; i++) {
        if (!allowWrites) break;
        const doc = {
          title: `bench-${Date.now()}-${i}`,
          content: 'benchmark',
          author_id: require('uuid').v4(),
          tags: new Set(['bench']),
          metadata: { source: 'benchmark' },
        };
        const created = await createDocument(table, doc);
        // Soft delete to reduce junk accumulation
        await deleteDocument(table, created.id);
        executed++;
      }
      if (!allowWrites) {
        // Fallback to read-only measurement when writes not allowed
        const result = await findDocuments(table, Math.min(ops, 1000));
        executed = Array.isArray(result?.rows) ? result.rows.length : 0;
        notes.push('Writes not allowed; performed read-only benchmark instead.');
      }
    } else if (mode === 'mixed') {
      const writes = Math.floor(ops / 2);
      const reads = ops - writes;
      if (!allowWrites) {
        const result = await findDocuments(table, Math.min(reads + writes, 1000));
        executed = Array.isArray(result?.rows) ? result.rows.length : 0;
        notes.push('Writes not allowed; performed read-only mixed benchmark.');
      } else {
        for (let i = 0; i < writes; i++) {
          const doc = {
            title: `bench-${Date.now()}-${i}`,
            content: 'benchmark',
            author_id: require('uuid').v4(),
            tags: new Set(['bench']),
            metadata: { source: 'benchmark' },
          };
          const created = await createDocument(table, doc);
          await deleteDocument(table, created.id);
          executed++;
        }
        const result = await findDocuments(table, Math.min(reads, 1000));
        executed += Array.isArray(result?.rows) ? result.rows.length : 0;
        notes.push('Mixed benchmark executed with writes and reads.');
      }
    } else {
      return res.status(400).json({ success: false, message: `Unknown mode: ${mode}` });
    }

    const ended = globalThis.process?.hrtime?.bigint?.() || BigInt(Date.now());
    const durationNs = Number(ended - started);
    const durationMs = Math.max(1, Math.round(durationNs / 1e6));
    const opsPerSec = Math.round((executed / durationMs) * 1000);

    return res.status(200).json({
      success: true,
      mode,
      opsRequested: ops,
      opsExecuted: executed,
      durationMs,
      opsPerSec,
      allowWrites,
      notes,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Benchmark error:', error);
    return res.status(500).json({ success: false, message: 'Benchmark failed', error: error.message });
  }
}
