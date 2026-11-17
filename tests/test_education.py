import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import app as web
from core.analytics import relationship, distribution, direct_relationship
from core.education import build_graph
from core.grounding import unsupported_claims


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def call(name, args, id='test-call'):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': id, 'type': 'tool_call'}])


class TutorTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'heart_rate_bpm': [90, 80, 70, 60, 75], 'sleep_hours': [5, 6, 7, 8, None],
                                   'age_years': [20, 30, 40, 50, 60]})
        self.question = 'What is the relationship between heart rate and sleep hours? What conclusions can we draw?'

    def test_exact_request_has_evidence_explanation_and_chart_without_pandasai(self):
        model = Model([AIMessage(content='The four complete records show a negative linear association (r = -1). One record was excluded. This is not evidence of causation.')])
        with patch('core.analysis.PandasAIBackend.analyze', side_effect=AssertionError('PandasAI must not run')):
            state = build_graph(model).invoke({'frame': self.frame, 'question': self.question})
        result = state['result']
        self.assertFalse(result['partial'])
        self.assertEqual(result['charts'][0]['type'], 'scatter')
        facts = result['evidence'][0]
        self.assertEqual(facts['rows_used'], 4)
        self.assertEqual(facts['rows_excluded'], 1)
        self.assertAlmostEqual(facts['pearson_r'], -1)
        self.assertEqual(facts['x_range'], [60, 90])
        self.assertEqual(facts['y_range'], [5, 8])
        self.assertAlmostEqual(facts['squared_pearson_r'], 1)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any(isinstance(m, ToolMessage) for m in model.calls[0]))

    def test_followup_receives_previous_tool_evidence_and_question(self):
        first = build_graph(Model([AIMessage(content='A negative correlation is present.')])).invoke({'frame': self.frame, 'question': self.question})
        model = Model([AIMessage(content='It means the observed variables move in opposite directions, without establishing cause.')])
        state = build_graph(model).invoke({'frame': self.frame, 'messages': first['messages'], 'question': 'Explain that more simply.'})
        self.assertIn('opposite directions', state['result']['answer'])
        self.assertTrue(any(isinstance(m, ToolMessage) and m.artifact for m in model.calls[0]))
        self.assertEqual(state['result']['charts'], [])

    def test_correlation_is_not_allowed_to_become_an_invented_per_hour_effect(self):
        model = Model([AIMessage(content='Heart rate is about 0.1 bpm less for each extra hour of sleep.')])
        state = build_graph(model).invoke({'frame': self.frame, 'question': self.question})
        self.assertEqual(state['result']['partial_reason'], 'grounding')
        self.assertNotIn('for each extra hour', state['result']['answer'])
        self.assertEqual(state['result']['charts'][0]['type'], 'scatter')
        self.assertNotIn('0.1 bpm less', state['messages'][-1].content)
        self.assertTrue(unsupported_claims('The data are cross-sectional.', [{'operation': 'relationship'}]))

    def test_failed_model_still_returns_educational_relationship_and_chart(self):
        with self.assertLogs('core.errors', level='WARNING'):
            state = build_graph(Model([TimeoutError('simulated timeout')])).invoke({'frame': self.frame, 'question': self.question})
        self.assertTrue(state['result']['partial'])
        self.assertEqual(len(state['result']['charts']), 1)
        self.assertIn('r = -1.000', state['result']['answer'])
        self.assertIn('not evidence', state['result']['answer'])

    def test_pie_uses_counted_ranges_and_never_generated_python(self):
        model = Model([AIMessage(content='These slices represent the share of records in labeled age ranges.')])
        state = build_graph(model).invoke({'frame': self.frame, 'question': 'Make a pie chart for age and explain it.'})
        chart = state['result']['charts'][0]
        self.assertEqual(chart['type'], 'pie')
        self.assertEqual(sum(chart['values']), 5)
        self.assertEqual(len(chart['labels']), 8)
        self.assertEqual(sum(row['count'] for row in state['result']['evidence'][0]['bin_counts']), 5)

    def test_bad_column_returns_tool_error_and_clarification(self):
        model = Model([call('analyze_relationship', {'x': 'unknown', 'y': 'sleep_hours'}),
                       AIMessage(content='Which heart-rate column should I use? The dataset has heart_rate_bpm.')])
        state = build_graph(model).invoke({'frame': self.frame, 'question': 'Compare two things.'})
        self.assertIn('Which heart-rate', state['result']['answer'])
        self.assertTrue(any(isinstance(m, ToolMessage) and m.status == 'error' for m in model.calls[1]))

    def test_tool_loop_is_bounded_and_closes_pending_calls(self):
        model = Model([call('inspect_dataset', {}, str(i)) for i in range(4)])
        state = build_graph(model).invoke({'frame': self.frame, 'question': 'Tell me about the dataset.'})
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(state['result']['partial'])
        self.assertIsInstance(state['messages'][-2], ToolMessage)

    def test_shortcut_does_not_ignore_filters_or_ambiguous_columns(self):
        self.assertIsNone(direct_relationship(self.question + ' Only age above 30.', self.frame))
        self.assertIsNone(direct_relationship('What is the relationship?', self.frame))

    def test_nonfinite_and_constant_measurements(self):
        result = relationship(pd.DataFrame({'x': [1, 1, np.inf], 'y': [2, 3, 4]}), 'x', 'y')
        self.assertIsNone(result['facts']['pearson_r'])
        self.assertEqual(result['facts']['rows_excluded'], 1)
        json.dumps(result, allow_nan=False)

    def test_scatter_sampling_does_not_change_statistics(self):
        frame = pd.DataFrame({'x': np.arange(3000), 'y': np.arange(3000) * 3})
        result = relationship(frame, 'x', 'y')
        self.assertEqual(result['facts']['rows_used'], 3000)
        self.assertEqual(len(result['charts'][0]['points']), 2000)
        self.assertAlmostEqual(result['facts']['pearson_r'], 1)

    def test_streamed_answer_persists_full_context_and_restores_conversation(self):
        web.datasets.clear()
        client = web.app.test_client()
        data = io.BytesIO(self.frame.to_csv(index=False).encode())
        client.post('/api/upload', data={'file': (data, 'classroom.csv')})
        model = Model([AIMessage(content='There is a negative association, not evidence of cause.')])
        with patch.object(web, 'graph', build_graph(model)):
            response = client.post('/api/chat', json={'question': self.question}, headers={'Accept': 'application/x-ndjson'})
            events = [json.loads(line) for line in response.data.decode().splitlines()]
        self.assertTrue(any(e['type'] == 'progress' for e in events))
        self.assertEqual(events[-1]['type'], 'result')
        self.assertEqual(events[-1]['result']['charts'][0]['type'], 'scatter')
        restored = client.get('/api/dataset').get_json()
        self.assertEqual(restored['conversation'][0]['question'], self.question)
        self.assertFalse(next(iter(web.datasets.values()))['lock'].locked())


if __name__ == '__main__':
    unittest.main()
