import {cp, mkdir, readFile, writeFile} from 'node:fs/promises';
await mkdir('public', {recursive: true});
await cp('static/index.html', 'public/index.html');
await cp('static', 'public/static', {recursive: true});
let html = await readFile('public/index.html', 'utf8');
html = html.replaceAll('10 MB', '4 MB').replace('<script src="/static/app.js">', '<script>window.HEALTHLENS_UPLOAD_LIMIT_MB=4;</script><script src="/static/app.js">');
await writeFile('public/index.html', html);
