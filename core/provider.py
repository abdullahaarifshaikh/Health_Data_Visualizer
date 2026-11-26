"""Bounded provider backoff: at most two quota retries, no retries for unrelated failures."""
import time
import re


def invoke(runnable, messages, deadline):
    for attempt in range(3):
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 429 or attempt == 2:
                raise
            headers = getattr(getattr(exc, 'response', None), 'headers', {})
            try:
                value = headers.get('retry-after')
                if value is None:
                    hint = re.search(r'try again in\s+([\d.]+)\s*(ms|s)', str(exc), re.IGNORECASE)
                    value = float(hint[1]) / (1000 if hint[2].lower() == 'ms' else 1) if hint else 20
                wait = float(value) + .5
            except (TypeError, ValueError):
                wait = 20.5
            wait = max(1, wait)
            if wait > 30 or time.monotonic() + wait + 5 >= deadline:
                raise
            time.sleep(wait)
