from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as web
import json,time
client=web.app.test_client()
client.post('/api/sample')
started=time.monotonic()
r=client.post('/api/chat',json={'question':'What is the relationship between heart rate and sleep hours? Explain the evidence and show a chart.'})
data=r.get_json()
Path('artifacts/refactor-live.json').write_text(json.dumps(data,indent=2))
result=data.get('result',{})
print(json.dumps({'status':r.status_code,'seconds':round(time.monotonic()-started,1),'trace':data.get('trace'),'partial':result.get('partial'),'charts':[c['type'] for c in result.get('charts',[])],'followups':result.get('followups'),'answer':result.get('answer')},ensure_ascii=True),flush=True)
