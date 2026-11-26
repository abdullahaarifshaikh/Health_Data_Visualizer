"""Hosted educational acceptance test; CLI handles Vercel auth, never key output."""
import json
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[1]
cli = shutil.which('vercel.cmd') or shutil.which('vercel')
base = 'https://health-data-visualizer-eight.vercel.app'
jar = str(root / 'artifacts' / 'cloud-cookies.txt')

def call(route, payload=None, stream=False):
    args = [cli, 'curl', '/api/' + route, '--deployment', base, '--scope', 'abdullahaarif3-9707',
            '--', '--silent', '--show-error', '--cookie-jar', jar, '--cookie', jar, '--max-time', '180']
    if payload is not None:
        args += ['--request', 'POST', '--header', 'Content-Type: application/json', '--data-binary', json.dumps(payload)]
    if stream:
        args += ['--header', 'Accept: application/x-ndjson', '--no-buffer']
    started = time.monotonic()
    result = subprocess.run(args, cwd=root, text=True, encoding='utf-8', errors='replace', capture_output=True, timeout=190)
    if result.returncode:
        raise RuntimeError('Hosted request failed; CLI exit ' + str(result.returncode))
    if stream:
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert any(e['type'] == 'progress' for e in events)
        response = next(e for e in reversed(events) if e['type'] in ('result', 'error'))
    else:
        response = json.loads(result.stdout)
    return response, round(time.monotonic() - started, 2)

status, _ = call('status')
assert status.get('workflow_version') == 'planned-composable-v2', status
assert status.get('memory_backend') == 'sqlite+private_blob', status
print('Hosted planned workflow and private persistence are active.', flush=True)
sample, _ = call('sample', {})
assert sample['profile']['rows'] == 180
cases = [
 ('greeting', 'Hi!'),
 ('overview', 'Help me understand this dataset.'),
 ('relationship', 'What is the relationship between heart rate and sleep hours? What conclusions can we draw?'),
 ('grouped_bar', 'Show a bar chart of mean heart_rate_bpm by group and explain it.'),
 ('matrix', 'Calculate Pearson correlations among age_years, heart_rate_bpm, systolic_bp_mmhg and sleep_hours. Show a correlation heatmap and explain the main associations. No regression.'),
 ('recall', 'In the earlier heart rate and sleep hours comparison, how many complete pairs did we use, and what was Pearson r? Recall the saved result; do not redo the analysis.')
]
for label, question in cases:
    response, elapsed = call('chat', {'question': question}, stream=True)
    (root / 'artifacts' / ('hosted-' + label + '.json')).write_text(json.dumps(response, indent=2), encoding='utf-8')
    if response.get('error'):
        raise RuntimeError(response['error'])
    result = response['result']
    print(json.dumps({'case': label, 'seconds_including_cli': elapsed, 'partial': result['partial'],
                      'charts': [c['type'] for c in result['charts']], 'followups': result['followups'],
                      'answer': result['answer'][:1000]}, ensure_ascii=True), flush=True)
    assert not result['partial'], label
    if label == 'greeting': assert 'plan_analysis' not in response['trace']
    if label == 'overview': assert any(f.get('operation') == 'overview' for f in result['evidence'])
    if label == 'relationship':
        assert any(c['type'] == 'scatter' for c in result['charts'])
        assert any(f.get('rows_used') == 166 for f in result['evidence'])
    if label == 'grouped_bar': assert any(c['type'] == 'bar' for c in result['charts'])
    if label == 'matrix': assert any(c['type'] == 'heatmap' for c in result['charts'])
    if label == 'recall':
        assert '166' in result['answer']
        assert '0.096' in result['answer'] or '0.10' in result['answer'] or '0.1' in result['answer']
restored, _ = call('dataset')
assert len(restored['conversation']) == len(cases)
active = restored['active_chat']
fresh, _ = call('chats', {'action': 'new'})
assert fresh['conversation'] == [] and fresh['profile']['rows'] == 180
restored, _ = call('chats', {'action': 'select', 'id': active})
assert len(restored['conversation']) == len(cases)
print('Hosted greeting, overview, charts, multi-variable analysis, older recall and chat isolation passed.', flush=True)
