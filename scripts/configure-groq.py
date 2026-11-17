"""Configure the approved Groq key using protected stdin; preserve other secrets."""
from pathlib import Path
import shutil
import subprocess
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
env = dotenv_values(root / '.env')
cli = shutil.which('vercel.cmd') or shutil.which('vercel')
values = {'GROQ_API_KEY': env['GROQ_API_KEY'], 'GROQ_MODEL': env.get('GROQ_MODEL', 'openai/gpt-oss-120b'),
          'LLM_PROVIDER': 'groq', 'HEALTHLENS_SANDBOX_NAME': 'healthlens-education'}
for key, value in values.items():
    result = subprocess.run([cli, 'env', 'add', key, '--environment', 'production,preview',
                             '--project', 'prj_imLB3ukP60eiAOMqvWyD54apAi1A', '--scope', 'abdullahaarif3-9707',
                             '--sensitive', '--yes', '--force'], input=value, text=True, capture_output=True,
                            cwd=root, timeout=60)
    print(key + ': ' + ('configured' if result.returncode == 0 else 'FAILED'), flush=True)
    if result.returncode:
        raise SystemExit('Unable to configure Vercel; secret values were not printed.')
