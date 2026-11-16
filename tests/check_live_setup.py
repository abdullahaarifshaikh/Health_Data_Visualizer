"""Print setup status without printing credentials or provider error bodies."""
import os
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
print('API key present:', bool(key), flush=True)
if key:
    try:
        response = requests.get('https://generativelanguage.googleapis.com/v1beta/models',
                                headers={'x-goog-api-key': key}, timeout=20)
        print('Google model listing HTTP status:', response.status_code, flush=True)
        if response.ok:
            names = [m['name'].removeprefix('models/') for m in response.json().get('models', [])]
            target = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
            print('Configured model:', target, flush=True)
            print('Configured model available:', target in names, flush=True)
            print('Available Flash models:', ', '.join(n for n in names if 'flash' in n and 'image' not in n and 'tts' not in n), flush=True)
            generated = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{target}:generateContent',
                                      headers={'x-goog-api-key': key}, timeout=45,
                                      json={'contents': [{'parts': [{'text': 'For a software smoke test, what is the mean of 20 and 40? Return only the number.'}]}]})
            print('Gemini generation HTTP status:', generated.status_code, flush=True)
            if generated.ok:
                parts = generated.json().get('candidates', [{}])[0].get('content', {}).get('parts', [])
                print('Gemini test answer:', ' '.join(p.get('text', '') for p in parts), flush=True)
            else:
                print('Gemini error status:', generated.json().get('error', {}).get('status'), flush=True)
    except Exception as exc:
        print('Google connectivity:', type(exc).__name__, flush=True)
try:
    result = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                            capture_output=True, text=True, timeout=10)
    print('Docker ready:', result.returncode == 0, flush=True)
except Exception as exc:
    print('Docker readiness:', type(exc).__name__, flush=True)
