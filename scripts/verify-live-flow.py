from pathlib import Path
import json,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as web
client=web.app.test_client()
client.post('/api/sample')
cases=[('greeting','Hi!'),('overview','Help me understand this dataset.'),('grouped_bar','Show a bar chart of mean heart_rate_bpm by group and explain it.'),('matrix','Calculate the pairwise Pearson correlations among age_years, heart_rate_bpm, systolic_bp_mmhg and sleep_hours, show a heatmap and explain the main associations. No regression.')]
for label,question in cases:
 start=time.monotonic();response=client.post('/api/chat',json={'question':question});data=response.get_json();result=data.get('result',{})
 Path('artifacts/live-'+label+'.json').write_text(json.dumps(data,indent=2))
 print(json.dumps({'case':label,'status':response.status_code,'seconds':round(time.monotonic()-start,1),'partial':result.get('partial'),'charts':[c['type'] for c in result.get('charts',[])],'answer':result.get('answer'),'followups':result.get('followups'),'trace':data.get('trace')},ensure_ascii=True),flush=True)
 assert response.status_code==200 and not result.get('partial'),label
 if label=='greeting': assert 'plan_analysis' not in data['trace']
 if label=='overview': assert any(f.get('operation')=='overview' for f in result['evidence'])
 if label=='grouped_bar': assert any(c['type']=='bar' for c in result['charts'])
 if label=='matrix': assert any(c['type']=='heatmap' for c in result['charts'])
print('Live greeting, dataset overview, grouped bar chart and multi-variable heatmap passed.',flush=True)
