import base64
import io
import os
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

import app as web
from core.analysis import SetupError, normalize_response
from core.pandasai_workflow import build_graph
from core.query import query_frames
from core.errors import redact_error, describe_failure


class Backend:
    def __init__(self):
        self.calls = []

    def analyze(self, frame, question, history):
        self.calls.append((question, history))
        return {'kind': 'answer', 'answer': str(frame['age'].mean())}


class WorkflowTests(unittest.TestCase):
    def test_diagnostics_redact_key_and_identify_timeout(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'private-test-key'}):
            self.assertNotIn('private-test-key', redact_error('request used private-test-key'))
        class Timeout(Exception):
            pass
        with self.assertLogs('core.errors', level='WARNING'):
            message = describe_failure(Timeout('request timed out'))
        self.assertIn('did not respond in time', message)
        self.assertIn('Error reference:', message)

    def test_sql_callback_computes_and_blocks_external_access(self):
        frame = pd.DataFrame({'age': [20, 40]})
        result = query_frames('SELECT avg(age) AS mean FROM classroom', {'classroom': frame})
        self.assertEqual(result.iloc[0, 0], 30)
        for query in ['DROP TABLE classroom', 'SELECT 1; SELECT 2', "SELECT * FROM read_csv_auto('C:/Windows/win.ini')"]:
            with self.assertRaises(Exception):
                query_frames(query, {'classroom': frame})

    def test_real_graph_calculates_and_routes(self):
        backend = Backend()
        state = build_graph(backend).invoke({'frame': pd.DataFrame({'age': [20, 40]}),
                                              'question': 'Mean age?', 'history': []})
        self.assertEqual(state['result']['answer'], '30.0')
        self.assertEqual(state['trace'], ['profile_data', 'validate_request', 'pandasai_analysis', 'format_result', 'teaching_context'])

    def test_validation_stops_backend(self):
        backend = Backend()
        state = build_graph(backend).invoke({'frame': pd.DataFrame({'age': [20]}), 'question': ''})
        self.assertTrue(state['error'])
        self.assertEqual(backend.calls, [])

    def test_backend_error_stops_formatting(self):
        class MissingSetup:
            def analyze(self, *args):
                raise SetupError('Set up Gemini')
        state = build_graph(MissingSetup()).invoke({'frame': pd.DataFrame({'age': [20]}), 'question': 'Mean?'})
        self.assertEqual(state['error'], 'Set up Gemini')
        self.assertNotIn('format_result', state['trace'])

    def test_table_preserves_missing_and_caps_preview(self):
        result = normalize_response(pd.DataFrame({'x': [1, None] * 150}))
        self.assertEqual(result['total_rows'], 300)
        self.assertEqual(len(result['table']), 200)
        self.assertIsNone(result['table'][1][0])

    def test_chart_response(self):
        buffer = io.BytesIO()
        Image.new('RGB', (20, 20), 'white').save(buffer, format='PNG')
        class Chart:
            type = 'chart'
            value = 'unused'
            def get_base64_image(self):
                return base64.b64encode(buffer.getvalue()).decode()
        result = normalize_response(Chart())
        self.assertTrue(result['image'].startswith('data:image/png;base64,'))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        web.datasets.clear()

    def test_sample_profile_and_isolation(self):
        data = self.client.post('/api/sample').get_json()
        self.assertEqual(data['profile']['rows'], 180)
        self.assertGreater(data['profile']['missing_cells'], 0)
        self.assertEqual(web.app.test_client().get('/api/dataset').get_json(), {})

    def test_csv_and_excel_upload(self):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO(b'age,bpm\n20,70\n40,80\n'), 'class.csv')})
        self.assertEqual(response.status_code, 200)
        workbook = io.BytesIO()
        pd.DataFrame({'age': [20, 40]}).to_excel(workbook, index=False)
        workbook.seek(0)
        response = self.client.post('/api/upload', data={'file': (workbook, 'class.xlsx')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['profile']['rows'], 2)

    def test_invalid_file_and_foreign_origin(self):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO(b'bad'), 'file.exe')})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post('/api/sample', headers={'Origin': 'https://example.com'}).status_code, 403)

    def test_missing_key_is_actionable(self):
        self.client.post('/api/sample')
        with patch.dict(os.environ, {'GOOGLE_API_KEY': '', 'GEMINI_API_KEY': '', 'GROQ_API_KEY': ''}):
            response = self.client.post('/api/chat', json={'question': 'Average age?'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['result']['partial'])
        self.assertIn('couldn’t reach the tutor', response.get_json()['result']['answer'])

    def test_chat_uses_graph_and_tracks_followup(self):
        backend = Backend()
        self.client.post('/api/upload', data={'file': (io.BytesIO(b'age\n20\n40\n'), 'class.csv')})
        with patch.object(web, 'graph', build_graph(backend)):
            first = self.client.post('/api/chat', json={'question': 'Mean age?'})
            second = self.client.post('/api/chat', json={'question': 'Explain that'})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(backend.calls[1][1][0]['question'], 'Mean age?')
        self.client.post('/api/clear')
        self.assertEqual(self.client.get('/api/dataset').get_json(), {})


if __name__ == '__main__':
    unittest.main()
