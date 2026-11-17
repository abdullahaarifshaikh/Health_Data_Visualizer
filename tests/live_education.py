"""Opt-in live tutor acceptance test, using only the built-in synthetic dataset."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web

client = web.app.test_client()
client.post('/api/sample')
report = []
for question in [
    'What is the relationship between heart rate and sleep hours? What conclusions can we draw?',
    'Explain that correlation more simply. Can we say sleep causes a lower heart rate?',
    'Make a pie chart for age and explain how to read it.'
]:
    started = time.monotonic()
    response = client.post('/api/chat', json={'question': question})
    data = response.get_json()
    if response.status_code != 200:
        print('Request failed:', data.get('error'), flush=True)
        raise SystemExit(1)
    result = data['result']
    entry = {'question': question, 'elapsed_seconds': round(time.monotonic() - started, 2),
             'answer': result['answer'], 'charts': [c['type'] for c in result['charts']],
             'partial': result['partial'], 'evidence': result['evidence'], 'trace': data['trace']}
    report.append(entry)
    print(json.dumps(entry, ensure_ascii=True), flush=True)
    if len(report) == 1:
        assert result['charts'][0]['type'] == 'scatter'
        assert len(result['answer']) > 200
    if len(report) == 3:
        assert result['charts'][0]['type'] == 'pie'
        assert sum(result['charts'][0]['values']) == 180
Path('artifacts/education-live-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Live educational conversation verified.', flush=True)
