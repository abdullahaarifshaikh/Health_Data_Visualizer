"""Keep useful local diagnostics while removing configured credentials."""
import logging
import os
import re
import uuid


def redact_error(message):
    for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY'):
        value = os.getenv(name)
        if value:
            message = message.replace(value, '[REDACTED]')
    return re.sub(r'AIza[\w-]{30,}|gsk_[A-Za-z0-9]{20,}', '[REDACTED]', message)


def describe_failure(exc):
    reference = uuid.uuid4().hex[:8]
    detail = redact_error(str(exc))[:4000]
    logging.getLogger(__name__).warning('Analysis failure [%s] %s: %s', reference, type(exc).__name__, detail)
    kind = type(exc).__name__.lower()
    provider = 'Groq' if os.getenv('LLM_PROVIDER') == 'groq' else 'Gemini'
    if 'timeout' in kind:
        message = f'{provider} did not respond in time. Please retry the request.'
    elif 'ratelimit' in kind:
        message = f'{provider} reported a usage limit. Wait before retrying or check your API quota.'
    elif 'authentication' in kind:
        message = f'{provider} rejected the API credentials. Check the configured key.'
    elif 'ModuleNotFoundError' in detail:
        message = 'The generated chart code needs a plotting package that is not installed in the execution environment.'
    else:
        message = 'The tutor could not complete this step. Its technical error has been recorded for diagnosis.'
    return f'{message} Error reference: {reference}.'
