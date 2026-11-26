import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
import app as web
from core.storage import Workspace


class BackupWorkspace(Workspace):
    blobs = {}
    def request(self, key, body=None):
        if body is None:
            return self.blobs.get(key)
        self.blobs[key] = body
        return b'{"ok":true}'


class StorageTests(unittest.TestCase):
    def test_private_backup_restores_dataset_and_native_checkpoint(self):
        with tempfile.TemporaryDirectory(dir='artifacts') as first, tempfile.TemporaryDirectory(dir='artifacts') as second:
            BackupWorkspace.blobs = {}
            owner = 'backup_test_owner_123456789'
            source = BackupWorkspace(owner, root=first, remote='mock')
            data = {'name': 'class', 'frame': pd.DataFrame({'age': [20, 40]}), 'active_chat': 'chat',
                    'chats': {'chat': {'responses': [{'question': 'Mean?', 'result': {'answer': '30'}}], 'history': [], 'turns': []}}}
            source.save_dataset('ds', data)
            from core.education import build_graph
            from core.tools import ToolContext
            class Model:
                def bind_tools(self, tools, **kwargs): return self
                def invoke(self, messages): return AIMessage(content='An educational concept explanation.')
            config = {'configurable': {'thread_id': 'ds:chat'}}
            build_graph(Model(), source.checkpointer, finalizer=lambda state,payload: {"answer":"An educational concept explanation.","followups":[]}, planner=lambda s: {"intent":"concept","goal":s["question"],"steps":[],"requires_calculation":False,"requires_chart":False}).invoke({'question': 'Remember this topic.'}, config,
                                                           context=ToolContext(data['frame'], source, 'ds', 'chat'))
            source.backup()
            restored = BackupWorkspace(owner, root=second, remote='mock')
            try:
                self.assertEqual(restored.load_dataset('ds')['frame']['age'].tolist(), [20, 40])
                state = build_graph(Model(), restored.checkpointer).get_state(config).values
                self.assertEqual(state['messages'][0].content, 'Remember this topic.')
                restored.clear_chat('ds', 'chat')
                self.assertIsNone(restored.checkpointer.get_tuple(config))
            finally:
                source.close(); restored.close()

    def test_web_checkpoint_history_survives_cache_and_connection_restart(self):
        client = web.app.test_client()
        sample = client.post('/api/sample').get_json()
        class Model:
            calls = []
            def with_structured_output(self, schema, **kwargs):
                outer = self
                class Structured:
                    def invoke(self, messages):
                        if schema.__name__ == 'AnalysisPlan':
                            return {'intent':'concept','goal':'Explain a teaching concept','steps':[],'requires_calculation':False,'requires_chart':False,'chart_tools':[]}
                        return json.loads(outer.invoke(messages).content)
                return Structured()
            def bind_tools(self, tools, **kwargs): return self
            def invoke(self, messages):
                self.calls.append(messages)
                return AIMessage(content=json.dumps({'answer': 'We are exploring the classroom dataset.', 'followups': []}))
        model = Model()
        with patch('core.education.make_model', return_value=model):
            for i in range(6):
                response = client.post('/api/chat', json={'question': f'Teaching question {i}?'})
                self.assertEqual(response.status_code, 200)
            web.datasets.clear()
            for item in web.workspaces.values(): item.close()
            web.workspaces.clear()
            restored = client.get('/api/dataset').get_json()
            self.assertEqual(len(restored['conversation']), 6)
            response = client.post('/api/chat', json={'question': 'What was teaching question 0 about?'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('Teaching question 0?', str(model.calls[-1]))
            client.post('/api/chats', json={'action': 'clear'})
            web.datasets.clear()
            cleared = client.get('/api/dataset').get_json()
            self.assertEqual(cleared['conversation'], [])
            self.assertEqual(cleared['profile']['rows'], 180)


if __name__ == '__main__': unittest.main()
