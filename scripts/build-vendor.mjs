import {mkdir, copyFile} from 'node:fs/promises';
await mkdir('static/vendor', {recursive: true});
for (const [source, target] of [
  ['node_modules/echarts/dist/echarts.min.js', 'echarts.min.js'],
  ['node_modules/marked/lib/marked.umd.js', 'marked.min.js'],
  ['node_modules/dompurify/dist/purify.min.js', 'purify.min.js']
]) await copyFile(source, `static/vendor/${target}`);
