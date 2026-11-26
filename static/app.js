const $ = selector => document.querySelector(selector);
let dataset = null, busy = false;
const MAX_UPLOAD_MB = window.HEALTHLENS_UPLOAD_LIMIT_MB || 10;
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); }
  catch { throw Error('The service could not return an answer. Please retry shortly.'); }
  if (!response.ok) throw Error(data.error || 'The request could not complete.');
  return data;
}

function notice(text) { $('#notice').textContent = text; $('#notice').hidden = !text; }
function markdown(text) {
  const node = el('div', undefined, 'lesson-text');
  if (window.marked && window.DOMPurify) {
    node.innerHTML = DOMPurify.sanitize(marked.parse(text || ''), {
      ALLOWED_TAGS: ['p', 'strong', 'em', 'ul', 'ol', 'li', 'h2', 'h3', 'h4', 'blockquote', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'td', 'th', 'br', 'a'],
      ALLOWED_ATTR: ['href', 'title']
    });
    node.querySelectorAll('a').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
    node.querySelectorAll('table').forEach(grid => {
      const wrap = el('div', undefined, 'table-wrap markdown-table');
      grid.replaceWith(wrap); wrap.append(grid);
    });
  } else node.textContent = text;
  return node;
}

function table(columns, rows) {
  const wrap = el('div', undefined, 'table-wrap'), grid = el('table'), head = el('thead'), header = el('tr');
  columns.forEach(column => header.append(el('th', column)));
  head.append(header); grid.append(head);
  const body = el('tbody');
  rows.forEach(row => {
    const tr = el('tr');
    row.forEach(value => tr.append(el('td', value === null ? '—' : String(value))));
    body.append(tr);
  });
  grid.append(body); wrap.append(grid); return wrap;
}

function disposeCharts() {
  document.querySelectorAll('.interactive-chart').forEach(node => {
    node.chartObserver?.disconnect(); node.chartInstance?.dispose();
  });
}

function showDataset(data) {
  dataset = data;
  $('#welcome').hidden = true; $('#overview').hidden = false;
  $('#dataset-name').textContent = data.name;
  $('#dataset-title').textContent = data.name;
  $('#view-data').disabled = false;
  $('#clear').disabled = busy;
  renderChats(data);
  $('#rows').textContent = data.profile.rows.toLocaleString();
  $('#cols').textContent = data.profile.column_count;
  $('#missing').textContent = data.profile.missing_cells.toLocaleString();
  $('#preview-table').replaceChildren(table(data.columns, data.preview));
  $('#column-list').replaceChildren(el('div', 'VARIABLES', 'eyebrow'));
  data.profile.columns.forEach(column => {
    const row = el('div', undefined, 'column');
    row.append(el('em', /int|float/.test(column.dtype) ? '#' : 'Aa'), el('span', column.name));
    row.title = `${column.dtype} · ${column.missing} missing`; $('#column-list').append(row);
  });
  $('#suggestions').replaceChildren();
  data.suggestions.forEach(question => $('#suggestions').append(questionButton(question)));
  disposeCharts(); $('#conversation').replaceChildren();
  for (const turn of data.conversation || []) {
    const block = el('div', undefined, 'turn');
    block.append(el('div', turn.question, 'user-message'));
    const card = el('div', undefined, 'response'); block.append(card); $('#conversation').append(block);
    renderAnswer(card, turn.result, turn.trace || []);
  }
  $('#send').disabled = busy;
  $('#composer-hint').textContent = 'Ask a question or follow up · Enter to send';
}

function questionButton(question) {
  const button = el('button', question);
  button.type = 'button';
  button.onclick = () => { if (!busy) { $('#question').value = question; $('#chat-form').requestSubmit(); } };
  return button;
}

function setBusy(value) {
  busy = value;
  for (const selector of ['#upload-button', '#sample', '#clear', '#new-chat']) $(selector).disabled = value || (selector !== '#upload-button' && selector !== '#sample' && !dataset);
  document.querySelectorAll('#chat-list button').forEach(button => button.disabled = value);
  $('#send').disabled = value || !dataset;
  document.querySelectorAll('.suggestions button, .followups button').forEach(button => button.disabled = value);
  $('#composer-hint').textContent = value ? 'Working through your question…' : 'Ask a question or follow up · Enter to send';
}

function evidenceRows(fact) {
  if (fact.operation === 'relationship') return [
    ['Measurements', `${fact.x} and ${fact.y}`], ['Complete paired records', fact.rows_used],
    ['Input scope', fact.input_scope || 'Dataset'], ['Records before filtering', fact.input_rows ?? fact.rows_total],
    ['Records excluded', fact.rows_excluded], ['Pearson correlation (r)', fact.pearson_r == null ? 'Undefined' : Number(fact.pearson_r).toFixed(4)],
    ['Significance / causality', 'Not tested']
  ];
  if (fact.operation === 'distribution') return [
    ['Variable', fact.column], ['Records used', fact.rows_used], ['Records excluded', fact.rows_excluded],
    ...Object.entries(fact.statistics || {}).map(([key, value]) => [key, Number(value).toFixed(3)])
  ];
  if (fact.operation === 'overview') return [['Records', fact.rows], ['Variables', fact.column_count], ['Missing cells', fact.missing_cells]];
  if (fact.operation === 'query') return [['Computed result rows', fact.result_rows], ['Preview truncated', fact.preview_truncated ? 'Yes' : 'No']];
  if (fact.operation === 'statistics') return [['Calculation', fact.statistic], ['Selected columns', (fact.columns || []).join(', ')],
    ['Records selected', fact.rows_selected], ['Grouping', fact.group_by || 'None'], ['Method', fact.method || 'Descriptive summary']];
  return [['Grouping', fact.category || '—'], ['Calculation', fact.aggregation || '—'], ['Missing measurements excluded', fact.rows_excluded || 0]];
}

function renderAnswer(card, result, trace) {
  card.replaceChildren(el('div', '✳ HEALTHLENS', 'response-label'));
  if (result.partial) card.append(el('p', result.partial_reason === 'grounding'
    ? 'Showing a calculation-based explanation because some generated wording could not be verified.'
    : 'The extended explanation was unavailable. Any calculated evidence and charts below are still usable.', 'partial-notice'));
  card.append(markdown(result.answer));
  for (const spec of result.charts || []) {
    const figure = el('figure', undefined, 'chart-card');
    const canvas = el('div', undefined, 'interactive-chart');
    canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', spec.title);
    figure.append(canvas, el('figcaption', spec.caption || 'Explore this calculated result.'));
    card.append(figure);
    requestAnimationFrame(() => window.HealthLensCharts?.render(canvas, spec));
  }
  // Compatibility with the optional PandasAI backend.
  if (result.kind === 'chart' && result.image) {
    const image = el('img'); image.src = result.image; image.alt = 'Generated dataset chart'; card.append(image);
    const download = el('a', 'Download chart'); download.href = result.image; download.download = 'healthlens-chart.png'; card.append(download);
  }
  const tables = result.tables || (result.table ? [{columns: result.columns, rows: result.table}] : []);
  if (tables.length || result.evidence?.length) {
    const details = el('details', undefined, 'evidence');
    details.append(el('summary', 'See the evidence and calculations'));
    for (const fact of result.evidence || []) details.append(table(['Measure', 'Calculated value'], evidenceRows(fact)));
    for (const data of tables) {
      details.append(table(data.columns, data.rows));
      const download = el('a', 'Download result table');
      const csv = [data.columns, ...data.rows].map(row => row.map(value => {
        let text = String(value ?? ''); if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
        return '"' + text.replaceAll('"', '""') + '"';
      }).join(',')).join('\r\n');
      download.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
      download.download = 'healthlens-results.csv'; details.append(download);
    }
    card.append(details);
  }
  if (result.teaching_note) card.append(el('p', result.teaching_note, 'teaching-note'));
  if (result.followups?.length) {
    const followups = el('div', undefined, 'followups');
    result.followups.forEach(question => followups.append(questionButton(question))); card.append(followups);
  }
  if (trace.length) {
    const details = el('details', undefined, 'trace');
    details.append(el('summary', 'View workflow steps'), el('p', trace.join(' → '))); card.append(details);
  }
}

async function ask(question, onProgress) {
  const response = await fetch('/api/chat', {method: 'POST',
    headers: {'Content-Type': 'application/json', 'Accept': 'application/x-ndjson'}, body: JSON.stringify({question})});
  if (!response.ok) {
    let error; try { error = (await response.json()).error; } catch { /* proxy error */ }
    throw Error(error || 'The tutor is temporarily unavailable. Please retry shortly.');
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', answer;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'progress') onProgress(event.message);
    if (event.type === 'error') throw Error(event.error);
    if (event.type === 'result') answer = event;
  }
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split('\n'); buffer = lines.pop(); lines.forEach(consume);
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
  } finally { reader.releaseLock(); }
  if (!answer) throw Error('The connection ended before the answer arrived. Your dataset is still loaded.');
  return answer;
}

$('#upload-button').onclick = () => $('#file').click();
$('#file').onchange = async () => {
  const file = $('#file').files[0]; if (!file) return;
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) { notice(`Choose a file smaller than ${MAX_UPLOAD_MB} MB.`); return; }
  const form = new FormData(); form.append('file', file); setBusy(true);
  try { showDataset(await api('/api/upload', {method: 'POST', body: form})); notice(''); }
  catch (error) { notice(error.message); }
  finally { setBusy(false); $('#file').value = ''; }
};
$('#sample').onclick = async () => {
  setBusy(true);
  try { showDataset(await api('/api/sample', {method: 'POST'})); notice('Synthetic classroom data loaded. These records are invented for learning.'); }
  catch (error) { notice(error.message); } finally { setBusy(false); }
};
$('#clear').onclick = async () => {
  if (busy) return;
  await changeChat('clear');
};
async function changeChat(action, id) {
  if (busy || !dataset) return;
  setBusy(true);
  try {
    showDataset(await api('/api/chats', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, id})}));
    $('#question').value = ''; notice('');
  } catch (error) { notice(error.message); }
  finally { setBusy(false); }
}
function renderChats(data) {
  $('#chat-list').replaceChildren();
  for (const chat of data.chats || []) {
    const button = el('button', chat.title, chat.id === data.active_chat ? 'chat-link selected' : 'chat-link');
    button.title = chat.title; button.setAttribute('aria-current', chat.id === data.active_chat ? 'true' : 'false');
    button.onclick = () => changeChat('select', chat.id); $('#chat-list').append(button);
  }
}
$('#new-chat').onclick = () => changeChat('new');
$('#view-data').onclick = () => $('#dataset-dialog').showModal();
$('#close-data').onclick = () => $('#dataset-dialog').close();
$('#dataset-dialog').onclick = event => { if (event.target === $('#dataset-dialog')) $('#dataset-dialog').close(); };
function collapseSidebar(collapsed) {
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  $('#sidebar-toggle').setAttribute('aria-expanded', String(!collapsed));
  $('#sidebar').inert = collapsed;
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}
collapseSidebar(window.innerWidth <= 720);
$('#sidebar-toggle').onclick = () => collapseSidebar(!document.body.classList.contains('sidebar-collapsed'));
$('#question').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#chat-form').requestSubmit(); }
};
function resizeComposer() {
  const field = $('#question');
  field.style.height = 'auto';
  field.style.height = Math.min(field.scrollHeight, 160) + 'px';
}
$('#question').addEventListener('input', resizeComposer);
$('#chat-form').onsubmit = async event => {
  event.preventDefault(); if (busy || !dataset) return;
  const question = $('#question').value.trim(); if (!question) return;
  setBusy(true);
  const turn = el('div', undefined, 'turn'); turn.append(el('div', question, 'user-message'));
  const card = el('div', undefined, 'response'), loading = el('div', 'Reading your question', 'response-label loading');
  card.append(loading); turn.append(card); $('#conversation').append(turn); $('#question').value = '';
  resizeComposer();
  turn.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  const started = Date.now(); let phase = 'Reading your question';
  const timer = setInterval(() => { loading.textContent = `${phase} · ${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
  try {
    const data = await ask(question, message => { phase = message; loading.textContent = message; });
    renderAnswer(card, data.result, data.trace);
    const updated = await api('/api/dataset'); dataset = updated; renderChats(updated);
  } catch (error) {
    card.classList.add('error'); card.replaceChildren(el('div', 'LET’S TRY THAT AGAIN', 'response-label'), el('p', error.message));
    $('#question').value = question;
  } finally { clearInterval(timer); setBusy(false); }
};

Promise.all([api('/api/status'), api('/api/dataset')]).then(([status, data]) => {
  $('#model-status').textContent = status.model;
  if (!status.configured) notice('Add the selected provider’s API key to the local .env file to enable the tutor. You can explore sample data now.');
  if (data.name) showDataset(data);
}).catch(error => notice(error.message));
