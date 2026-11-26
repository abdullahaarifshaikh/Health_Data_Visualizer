import {get, put, del} from '@vercel/blob';
import {timingSafeEqual} from 'node:crypto';
import {Readable} from 'node:stream';

export async function storage(req, res, url) {
  const expected = Buffer.from(process.env.APP_PROXY_SECRET || '');
  const supplied = Buffer.from(String(req.headers['x-healthlens-storage'] || ''));
  if (!expected.length || supplied.length !== expected.length || !timingSafeEqual(expected, supplied))
    return res.status(403).json({error: 'Private storage endpoint.'});
  const key = url.searchParams.get('key');
  if (!/^[A-Za-z0-9_-]{20,80}\/(manifest|[a-f0-9]{32}\/\d{1,3})$/.test(key || ''))
    return res.status(400).json({error: 'Invalid storage key.'});
  res.setHeader('Cache-Control', 'no-store');
  const path = 'workspaces/' + key;
  try {
    if (req.method === 'GET') {
      const blob = await get(path, {access: 'private', useCache: false});
      if (!blob) return res.status(404).end();
      res.setHeader('Content-Type', 'application/octet-stream');
      return Readable.fromWeb(blob.stream).pipe(res);
    }
    if (req.method !== 'POST') return res.status(405).end();
    const chunks = []; let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 2 * 1024 * 1024 + 65536) return res.status(413).end();
      chunks.push(chunk);
    }
    let previous;
    if (key.endsWith('/manifest')) {
      const old = await get(path, {access: 'private', useCache: false});
      if (old) previous = JSON.parse(await new Response(old.stream).text());
    }
    await put(path, Buffer.concat(chunks), {access: 'private', addRandomSuffix: false, allowOverwrite: true,
      contentType: 'application/octet-stream', cacheControlMaxAge: 60});
    if (previous && /^[a-f0-9]{32}$/.test(previous.version) && previous.parts <= 128) {
      const owner = key.split('/')[0];
      // Old backups are no longer needed after the new manifest has committed.
      try { await del(Array.from({length: previous.parts}, (_, i) => `workspaces/${owner}/${previous.version}/${i}`)); }
      catch { /* The current backup is committed; stale backup cleanup can be retried later. */ }
    }
    return res.json({ok: true});
  } catch {
    return res.status(503).json({error: 'Persistent memory storage unavailable.'});
  }
}
