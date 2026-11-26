import unittest
from unittest.mock import patch
import app as web
from langchain_core.messages import HumanMessage

class ChatTests(unittest.TestCase):
    def test_chats_isolate_context_and_clear_keeps_dataset(self):
        web.datasets.clear()
        client = web.app.test_client()
        original = client.post('/api/sample').get_json()
        first = original['active_chat']
        data = next(iter(web.datasets.values()))
        data.update(responses=[{'question': 'First question', 'result': {'answer': 'Answer'}, 'trace': []}], history=[{'question': 'First question', 'answer': 'Answer'}], turns=[[HumanMessage(content='context')]])
        fresh = client.post('/api/chats', json={'action': 'new'}).get_json()
        self.assertNotEqual(first, fresh['active_chat'])
        self.assertEqual(fresh['conversation'], [])
        self.assertEqual(data['turns'], [])
        self.assertEqual(len(fresh['chats']), 2)
        restored = client.post('/api/chats', json={'action': 'select', 'id': first}).get_json()
        self.assertEqual(restored['conversation'][0]['question'], 'First question')
        self.assertEqual(data['turns'][0][0].content, 'context')
        cleared = client.post('/api/chats', json={'action': 'clear'}).get_json()
        self.assertEqual(cleared['conversation'], [])
        self.assertEqual(cleared['profile']['rows'], 180)
        self.assertEqual(data['history'], [])
        self.assertEqual(client.get('/api/dataset').get_json()['conversation'], [])
        self.assertEqual(client.post('/api/chats', json={'action': 'select', 'id': 'missing'}).status_code, 404)
        other = web.app.test_client()
        self.assertEqual(other.post('/api/chats', json={'action': 'select', 'id': first}).status_code, 400)
        data['lock'].acquire()
        try:
            self.assertEqual(client.post('/api/chats', json={'action': 'clear'}).status_code, 409)
        finally:
            data['lock'].release()

if __name__ == '__main__':
    unittest.main()
