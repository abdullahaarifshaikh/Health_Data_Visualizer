"""Opt-in live API smoke check using only synthetic classroom data."""
import base64
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web

client = web.app.test_client()
sample = client.post('/api/sample').get_json()
print('Synthetic dataset:', sample['profile']['rows'], 'rows', flush=True)
expected = sum(row[1] for row in web.datasets[next(iter(web.datasets))]['frame'].itertuples(index=False, name=None)) / sample['profile']['rows']
for question in ['What is the mean age_years? Return the numeric mean.',
                 'Create a histogram of age_years using matplotlib. Label the x axis Age (years) and y axis Record count.']:
    response = client.post('/api/chat', json={'question': question})
    payload = response.get_json()
    print('Request status:', response.status_code, flush=True)
    if response.status_code != 200:
        print('App error:', payload.get('error'), flush=True)
        break
    result = payload['result']
    print('Result kind:', result['kind'], flush=True)
    print('Workflow:', ' -> '.join(payload['trace']), flush=True)
    if result['kind'] == 'chart':
        out = Path('artifacts/live-age-histogram.png')
        out.write_bytes(base64.b64decode(result['image'].split(',', 1)[1]))
        print('Chart saved:', out, flush=True)
    else:
        print('Calculated answer:', result['answer'], flush=True)
        print('Expected mean age:', expected, flush=True)
