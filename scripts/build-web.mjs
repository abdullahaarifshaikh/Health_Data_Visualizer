import {cp, mkdir} from 'node:fs/promises';
await mkdir('public', {recursive: true});
await cp('static/index.html', 'public/index.html');
await cp('static', 'public/static', {recursive: true});
