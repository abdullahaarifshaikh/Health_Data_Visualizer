import io,json,random,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as web
import pandas as pd
rng=random.Random(23)
rows=[]
for i in range(120):
 age=rng.randint(20,75); bmi=round(rng.uniform(19,38),1); glucose=round(70+age*.7+bmi*.6+rng.gauss(0,12),1)
 rows.append({'age':age,'bmi':bmi,'glucose':glucose,'insulin':round(10+glucose*.6+rng.gauss(0,15),1) if i%11 else None,'diabetes':int(i%3==0)})
client=web.app.test_client()
client.post('/api/upload',data={'file':(io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode()),'synthetic_metabolic_classroom.csv')})
started=time.monotonic()
r=client.post('/api/chat',json={'question':'Explore relationships among glucose, bmi, age, insulin and diabetes using pairwise correlations. Include a correlation heatmap and explain limitations. No regression.'})
data=r.get_json(); result=data.get('result',{})
Path('artifacts/refactor-multivariable-live.json').write_text(json.dumps(data,indent=2))
print(json.dumps({'status':r.status_code,'seconds':round(time.monotonic()-started,1),'partial':result.get('partial'),'charts':[c['type'] for c in result.get('charts',[])],'followups':result.get('followups'),'answer':result.get('answer')},ensure_ascii=True),flush=True)
