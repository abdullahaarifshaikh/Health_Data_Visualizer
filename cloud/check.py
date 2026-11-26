"""Cloud smoke test without an API key or persistent test data."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from core.education import build_graph, unpack, prompt_context
from core.storage import Workspace
from core.tools import ToolContext
class Model:
 def bind_tools(self,tools,**kwargs): return self
 def invoke(self,messages):
  human=next(m for m in reversed(messages) if isinstance(m,HumanMessage))
  current=messages[messages.index(human):]
  outputs=[json.loads(m.content) for m in current if isinstance(m,ToolMessage)]
  def call(name,args,id): return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':id}])
  if not outputs: return call('calculate_statistics',{'operation':'pearson','columns':['heart_rate_bpm','sleep_hours']},'corr')
  if not any(o.get('chart_type')=='scatter' for o in outputs): return call('scatter_plot',{'x':'heart_rate_bpm','y':'sleep_hours'},'chart')
  return AIMessage(content=json.dumps({'answer':'The calculated correlation is -1 using three complete pairs. This is not evidence of cause.','followups':[]}))
with tempfile.TemporaryDirectory() as folder:
 workspace=Workspace('cloud_check_owner_12345678',root=folder,remote='')
 try:
  context=ToolContext(pd.DataFrame({'heart_rate_bpm':[60,70,80],'sleep_hours':[8,7,6]}),workspace,'ds','chat')
  result=build_graph(Model(),workspace.checkpointer,finalizer=lambda state,payload: dict(zip(("answer","followups"),unpack(Model().invoke(prompt_context(state)[0])))),planner=lambda s: {"intent":"relationship","goal":s["question"],"steps":["calculate_statistics","scatter_plot"],"requires_calculation":True,"requires_chart":True}).invoke({'question':'What is the relationship?'},{'configurable':{'thread_id':'ds:chat'}},context=context)['result']
  assert result['charts'][0]['type']=='scatter'
  assert result['evidence'][0]['rows_used']==3
  assert not result['partial']
 finally: workspace.close()
print('Cloud independent tools, runtime context, checkpoint and chart data verified.')
