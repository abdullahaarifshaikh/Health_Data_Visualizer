"""Transfer local configuration through CLI stdin, without printing values."""
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
values = dotenv_values(root / '.env')
key = values.get('GOOGLE_API_KEY') or values.get('GEMINI_API_KEY')
if not key:
    raise SystemExit('No local Gemini key configured.')
variables = {'GOOGLE_API_KEY': key, 'GEMINI_MODEL': values.get('GEMINI_MODEL', 'gemini-3.6-flash'),
             'APP_PROXY_SECRET': secrets.token_urlsafe(48), 'SESSION_SECRET': secrets.token_urlsafe(48),
             'HEALTHLENS_SANDBOX_NAME': 'healthlens-backend'}
cli = shutil.which('vercel.cmd') or shutil.which('vercel')
for name, value in variables.items():
    result = subprocess.run([cli, 'env', 'add', name, '--environment', 'production,preview',
                             '--project', 'prj_imLB3ukP60eiAOMqvWyD54apAi1A',
                             '--scope', 'abdullahaarif3-9707', '--sensitive', '--yes', '--force'],
                            input=value, text=True, capture_output=True, cwd=root, timeout=60)
    print(name + ': ' + ('configured' if result.returncode == 0 else 'FAILED'), flush=True)
    if result.returncode:
        raise SystemExit('Vercel environment configuration failed; no secret values were printed.')
