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
            '--', '--silent', '--show-error', '--cookie-jar', jar, '--cookie', jar, '--max-time', '100']
    if payload is not None:
        args += ['--request', 'POST', '--header', 'Content-Type: application/json', '--data-binary', json.dumps(payload)]
    if stream:
        args += ['--header', 'Accept: application/x-ndjson', '--no-buffer']
    started = time.monotonic()
    result = subprocess.run(args, cwd=root, text=True, encoding='utf-8', errors='replace', capture_output=True, timeout=110)
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
print('Hosted provider/model:', status.get('provider'), status.get('model'), flush=True)
assert status.get('provider') == 'groq'
sample, _ = call('sample', {})
assert sample['profile']['rows'] == 180
for index, question in enumerate([
    'What is the relationship between heart rate and sleep hours? What conclusions can we draw?',
    'Explain that correlation more simply. Can we say sleep causes a lower heart rate?',
    'Make a pie chart for age and explain how to read it.'
]):
    response, elapsed = call('chat', {'question': question}, stream=True)
    if response.get('error'):
        raise RuntimeError(response['error'])
    result = response['result']
    assert result['kind'] == 'conversation'
    assert len(result['answer']) > 100
    if index == 0:
        assert result['charts'][0]['type'] == 'scatter'
        assert result['evidence'][0]['rows_used'] == 166
    if index == 2:
        assert result['charts'][0]['type'] == 'pie'
        assert sum(result['charts'][0]['values']) == 180
    print('Hosted result:', json.dumps({'seconds_including_cli': elapsed, 'charts': [c['type'] for c in result['charts']],
                                       'partial': result['partial'], 'answer': result['answer'][:700]}, ensure_ascii=True), flush=True)
restored, _ = call('dataset')
assert len(restored['conversation']) == 3
print('Hosted Groq conversation, charts, streaming protocol and history verified.', flush=True)
