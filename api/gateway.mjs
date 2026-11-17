import {Sandbox} from '@vercel/sandbox';
import {Readable} from 'node:stream';

export const config = {api: {bodyParser: false}};
const routes = new Set(['status', 'dataset', 'sample', 'upload', 'chat', 'clear']);
let ready;
let readyUntil = 0;

async function backend() {
  if (!process.env.APP_PROXY_SECRET || !process.env.SESSION_SECRET)
    throw new Error('Backend startup failed');
  if (ready && Date.now() < readyUntil) return ready;
  readyUntil = Date.now() + 20_000;
  ready = (async () => {
    const sandbox = await Sandbox.get({name: process.env.HEALTHLENS_SANDBOX_NAME || 'healthlens-backend'});
    const result = await sandbox.runCommand({
      cmd: 'bash', args: ['cloud/start.sh'], cwd: '/vercel/healthlens', sudo: true,
      env: {
        GOOGLE_API_KEY: process.env.GOOGLE_API_KEY || '',
        GEMINI_MODEL: process.env.GEMINI_MODEL || 'gemini-3.6-flash',
        GROQ_API_KEY: process.env.GROQ_API_KEY || '',
        GROQ_MODEL: process.env.GROQ_MODEL || 'openai/gpt-oss-120b',
        LLM_PROVIDER: process.env.LLM_PROVIDER || 'gemini',
        APP_PROXY_SECRET: process.env.APP_PROXY_SECRET,
        SESSION_SECRET: process.env.SESSION_SECRET,
        HEALTHLENS_REVISION: process.env.VERCEL_GIT_COMMIT_SHA || 'education-v2',
        ANALYSIS_BACKEND: process.env.ANALYSIS_BACKEND || 'education',
        HOST: '0.0.0.0', PORT: '5000', PYTHONUTF8: '1'
      }, timeoutMs: 100_000
    });
    if (result.exitCode !== 0) throw new Error('Backend startup failed');
    return sandbox.domain(5000);
  })().catch(error => {ready = undefined; readyUntil = 0; throw error;});
  return ready;
}

export default async function handler(req, res) {
  const url = new URL(req.url, 'https://healthlens.invalid');
  const route = req.query?.route || url.searchParams.get('route');
  if (!routes.has(route)) return res.status(404).json({error: 'Unknown endpoint.'});
  if (!['GET', 'POST'].includes(req.method)) return res.status(405).json({error: 'Method not allowed.'});
  // Only same-origin browser writes; do not trust caller-supplied proxy headers.
  const origin = req.headers.origin;
  if (req.method === 'POST' && origin && new URL(origin).host !== req.headers.host)
    return res.status(403).json({error: 'Cross-origin requests are not allowed.'});
  try {
    const base = await backend();
    const headers = {'X-HealthLens-Proxy': process.env.APP_PROXY_SECRET};
    if (req.headers.cookie) headers.cookie = req.headers.cookie;
    if (req.headers['content-type']) headers['content-type'] = req.headers['content-type'];
    if (req.headers.accept) headers.accept = req.headers.accept;
    const chunks = [];
    let bytes = 0;
    for await (const chunk of req) {
      bytes += chunk.length;
      if (bytes > 4_000_000) return res.status(413).json({error: 'The hosted upload limit is 4 MB.'});
      chunks.push(chunk);
    }
    const response = await fetch(`${base}/api/${route}`, {
      method: req.method, headers, body: req.method === 'POST' ? Buffer.concat(chunks) : undefined,
      signal: AbortSignal.timeout(230_000), redirect: 'error'
    });
    res.statusCode = response.status;
    res.setHeader('Content-Type', response.headers.get('content-type') || 'application/json');
    res.setHeader('Cache-Control', 'no-store');
    for (const cookie of response.headers.getSetCookie()) res.appendHeader('Set-Cookie', cookie);
    if (response.body) {
      const stream = Readable.fromWeb(response.body);
      stream.on('error', () => res.destroy());
      stream.pipe(res);
    } else res.end();
  } catch (error) {
    console.error('HealthLens gateway:', error.name, error.message === 'Backend startup failed' ? error.message : 'upstream request failed');
    res.status(503).json({error: 'The analysis service is waking up or temporarily unavailable. Retry shortly.'});
  }
}
