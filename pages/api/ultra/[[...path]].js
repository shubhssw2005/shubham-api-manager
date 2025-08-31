import { proxyRequest } from '../../../lib/ultraProxy.js';

export const config = {
  api: {
    bodyParser: false, // stream the body to upstream without buffering
  },
};

export default async function handler(req, res) {
  const base = process.env.ULTRA_CPP_URL || 'http://localhost:8080';
  const segments = Array.isArray(req.query.path) ? req.query.path : [];
  const pathPart = segments.map((s) => encodeURIComponent(s)).join('/');

  // Reuse the original query string (if any) to preserve exact parameters
  const idx = req.url.indexOf('?');
  const query = idx >= 0 ? req.url.slice(idx) : '';

  const targetPath = `/${pathPart}${query}`;

  try {
    await proxyRequest(req, res, base, targetPath);
  } catch (err) {
    console.error('Ultra proxy error:', err);
    if (!res.headersSent) {
      res.status(502).json({ success: false, error: 'proxy_failed', message: err.message });
    }
  }
}
