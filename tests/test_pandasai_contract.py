"""Actual installed PandasAI/LiteLLM integration with provider and Docker mocked.

This validates the adapter contract without an API key or host code execution.
It is deliberately not a live model or Docker test.
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('SCARF_NO_ANALYTICS', 'true')
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('TIKTOKEN_CACHE_DIR', os.path.abspath('.cache/tiktoken'))

import pandas as pd
from pandasai import DataFrame
from pandasai.sandbox import Sandbox
from core.analysis import PandasAIBackend


class RecordingSandbox(Sandbox):
    stopped = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False
        RecordingSandbox.stopped = True

    def _exec_code(self, code, environment):
        assert 'result' in code
        assert callable(environment['execute_sql_query'])
        queries = self._extract_sql_queries_from_code(code)
        result = environment['execute_sql_query'](queries[0])
        return {'type': 'number', 'value': float(result.iloc[0, 0])}


class PandasAIContractTest(unittest.TestCase):
    def test_real_agent_and_litellm_configuration(self):
        frame = pd.DataFrame({'age': [20, 40]})
        name = DataFrame(frame).schema.name
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=f"df = execute_sql_query('SELECT AVG(age) AS mean_age FROM {name}')\nresult = {{'type': 'number', 'value': float(df.iloc[0, 0])}}"))])
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-not-a-real-key'}), \
             patch('pandasai_docker.DockerSandbox', RecordingSandbox), \
             patch('pandasai_litellm.litellm.completion', return_value=response) as completion:
            result = PandasAIBackend().analyze(frame, 'Mean age?', [])
        self.assertEqual(float(result['answer']), 30)
        self.assertEqual(completion.call_args.kwargs['model'], 'gemini/gemini-3.6-flash')
        self.assertTrue(RecordingSandbox.stopped)


if __name__ == '__main__':
    unittest.main()
