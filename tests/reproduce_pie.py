"""Diagnose the classroom pie request without exposing API credentials."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
from core.analysis import PandasAIBackend
from pandasai import Agent
from unittest.mock import patch

def redact(text):
    for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        if os.getenv(name):
            text = text.replace(os.environ[name], '[REDACTED]')
    return text

original_execute = Agent.execute_code
attempt = 0
def traced_execute(agent, code):
    global attempt
    attempt += 1
    Path(f'artifacts/pie-code-{attempt}.py').write_text(redact(code), encoding='utf-8')
    try:
        return original_execute(agent, code)
    except Exception as exc:
        print('Execution failure:', type(exc).__name__, redact(str(exc))[:5000], flush=True)
        raise

client = web.app.test_client()
client.post('/api/sample')
frame = next(iter(web.datasets.values()))['frame']
with patch.object(Agent, 'execute_code', traced_execute):
    for run in range(3):
        print('Reproduction run:', run + 1, flush=True)
        try:
            result = PandasAIBackend().analyze(frame, 'make a pie chart for age', [])
            print('Result:', result['kind'], flush=True)
            if result['kind'] == 'chart':
                import base64
                Path(f'artifacts/reproduced-age-pie-{run+1}.png').write_bytes(base64.b64decode(result['image'].split(',', 1)[1]))
        except Exception as exc:
            print('Error type:', type(exc).__name__, flush=True)
            print('Error detail:', redact(str(exc))[:6000], flush=True)
