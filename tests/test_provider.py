import time
import unittest
from unittest.mock import patch
from core.provider import invoke


class RateLimit(Exception):
    status_code = 429
    response = type('Response', (), {'headers': {'retry-after': '1'}})()


class ProviderTests(unittest.TestCase):
    def test_only_quota_failures_are_retried_once_within_budget(self):
        class Runnable:
            calls = 0
            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimit()
                return 'completed'
        runner = Runnable()
        with patch('core.provider.time.sleep') as sleep:
            self.assertEqual(invoke(runner, [], time.monotonic() + 30), 'completed')
            sleep.assert_called_once_with(1.5)
        self.assertEqual(runner.calls, 2)

    def test_no_retry_after_budget_expires_or_for_other_errors(self):
        class Runnable:
            def invoke(self, messages): raise RateLimit()
        with patch('core.provider.time.sleep') as sleep:
            with self.assertRaises(RateLimit): invoke(Runnable(), [], time.monotonic())
            sleep.assert_not_called()
        class Other:
            def invoke(self, messages): raise ValueError('Invalid input')
        with patch('core.provider.time.sleep') as sleep:
            with self.assertRaises(ValueError): invoke(Other(), [], time.monotonic() + 60)
            sleep.assert_not_called()
