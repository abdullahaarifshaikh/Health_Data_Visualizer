"""Keep useful local diagnostics while removing configured credentials."""
import logging
import os
import re
import uuid


def redact_error(message):
    for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        value = os.getenv(name)
        if value:
            message = message.replace(value, '[REDACTED]')
    return re.sub(r'AIza[\w-]{30,}', '[REDACTED]', message)


def describe_failure(exc):
    reference = uuid.uuid4().hex[:8]
    detail = redact_error(str(exc))[:4000]
    logging.getLogger(__name__).warning('Analysis failure [%s] %s: %s', reference, type(exc).__name__, detail)
    kind = type(exc).__name__.lower()
    if 'timeout' in kind:
        message = 'Gemini did not respond in time. Please retry the request.'
    elif 'ratelimit' in kind:
        message = 'Gemini reported a usage limit. Wait before retrying or check your Google API quota.'
    elif 'authentication' in kind:
        message = 'Google rejected the API credentials. Check the configured key.'
    elif 'ModuleNotFoundError' in detail:
        message = 'The generated chart code needs a plotting package that is not installed in the execution environment.'
    else:
        message = 'The generated analysis could not complete. Its technical error has been recorded for diagnosis.'
    return f'{message} Error reference: {reference}.'
