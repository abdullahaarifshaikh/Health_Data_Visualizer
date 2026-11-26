"""Verify private hosted backup/restore with an isolated synthetic workspace."""
import secrets
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from langchain_core.messages import AIMessage
from core.education import build_graph
from core.storage import Workspace
from core.tools import ToolContext

owner = 'recovery_' + secrets.token_urlsafe(20)
frame = pd.DataFrame({'age': [20, 40], 'value': [3, 7]})
data = {'name': 'Synthetic recovery check', 'frame': frame, 'active_chat': 'chat',
        'chats': {'chat': {'responses': [], 'history': [], 'turns': []}}}
config = {'configurable': {'thread_id': 'ds:chat'}}
plan = lambda state: {'intent': 'concept', 'goal': 'Explain a concept', 'steps': [], 'requires_calculation': False, 'requires_chart': False, 'chart_tools': []}
final = lambda state, payload: {'answer': 'This is a synthetic teaching example.', 'followups': []}

with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
    source = Workspace(owner, root=first)
    restored = None
    try:
        source.save_dataset('ds', data)
        graph = build_graph(planner=plan, finalizer=final, checkpointer=source.checkpointer)
        graph.invoke({'question': 'Remember this teaching example.'}, config, context=ToolContext(frame, source, 'ds', 'chat'))
        source.backup()
        source.close()
        source = None
        restored = Workspace(owner, root=second)
        assert restored.load_dataset('ds')['frame']['value'].tolist() == [3, 7]
        state = build_graph(checkpointer=restored.checkpointer).get_state(config).values
        assert state['messages'][0].content == 'Remember this teaching example.'
        restored.delete_dataset('ds', ['chat'])
        restored.backup()
        print('Actual private hosted backup, fresh-directory restore and native checkpoint recovery verified.')
    finally:
        if source is not None: source.close()
        if restored is not None: restored.close()
