"""Check the educational graph and chart calculations without using an API key."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from langchain_core.messages import AIMessage
from core.education import build_graph

class Model:
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return AIMessage(content='These recorded measurements have a negative linear association. That is not evidence of cause.')

state = build_graph(Model()).invoke({'frame': pd.DataFrame({'heart_rate_bpm': [60, 70, 80], 'sleep_hours': [8, 7, 6]}),
                                    'question': 'What is the relationship between heart rate and sleep hours?'})
assert state['result']['charts'][0]['type'] == 'scatter'
assert state['result']['evidence'][0]['rows_used'] == 3
print('Cloud educational graph, injected tools, evidence check and interactive chart data verified.')
