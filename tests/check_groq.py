"""Verify Groq access without printing credentials or provider request headers."""
import os
import requests
from dotenv import load_dotenv
load_dotenv()
response = requests.get('https://api.groq.com/openai/v1/models',
                        headers={'Authorization': 'Bearer ' + os.environ['GROQ_API_KEY']}, timeout=20)
print('Groq model listing HTTP:', response.status_code, flush=True)
if response.ok:
    ids = [model['id'] for model in response.json()['data']]
    print('Available reasoning model candidates:', ', '.join(model for model in ids if any(t in model for t in ('gpt-oss', 'qwen', 'llama-3.3'))), flush=True)
else:
    print('Groq error category:', response.json().get('error', {}).get('type'), flush=True)
