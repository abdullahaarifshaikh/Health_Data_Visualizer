"""Opt-in deployment smoke test using Vercel CLI auth and synthetic data."""
import base64
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
cli = shutil.which('vercel.cmd') or shutil.which('vercel')
base = 'https://health-data-visualizer-eight.vercel.app'
jar = str(root / 'artifacts' / 'cloud-cookies.txt')

def call(route, payload=None):
    args = [cli, 'curl', '/api/' + route, '--deployment', base, '--scope', 'abdullahaarif3-9707',
            '--', '--silent', '--show-error', '--cookie-jar', jar, '--cookie', jar, '--max-time', '260']
    if payload is not None:
        args += ['--request', 'POST', '--header', 'Content-Type: application/json', '--data-binary', json.dumps(payload)]
    result = subprocess.run(args, cwd=root, text=True, encoding='utf-8', errors='replace', capture_output=True, timeout=280)
    if result.returncode:
        raise RuntimeError('Hosted request failed; CLI exit ' + str(result.returncode))
    return json.loads(result.stdout)

sample = call('sample', {})
print('Cloud sample rows:', sample.get('profile', {}).get('rows'), flush=True)
if 'error' in sample:
    raise SystemExit(sample['error'])
for question in ['What is the mean age_years? Return the numeric answer.',
                 "Create a Matplotlib histogram of age_years. Save it to a PNG and return the saved image path as a plot result."]:
    response = call('chat', {'question': question})
    if 'error' in response:
        print('Cloud analysis error:', response['error'], flush=True)
        raise SystemExit(1)
    result = response['result']
    print('Cloud result:', result['kind'], flush=True)
    if result['kind'] == 'chart':
        (root / 'artifacts' / 'hosted-histogram.png').write_bytes(base64.b64decode(result['image'].split(',', 1)[1]))
    else:
        print('Cloud answer:', result['answer'], flush=True)
print('Hosted workflow verified.', flush=True)
