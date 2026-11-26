"""Provider selection is configuration, independent of the teaching graph."""
import os
from core.analysis import SetupError


def provider_name():
    return os.getenv('LLM_PROVIDER', 'gemini').lower()


def model_name():
    return os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b') if provider_name() == 'groq' else os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')


def configured():
    return bool(os.getenv('GROQ_API_KEY')) if provider_name() == 'groq' else bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'))


def make_model(timeout=22, max_tokens=1200):
    if not configured():
        raise SetupError(f'Configure the {provider_name()} API key to enable the tutor.')
    if provider_name() == 'groq':
        from langchain_groq import ChatGroq
        reasoning = {'reasoning_format': 'hidden', 'reasoning_effort': 'low'} if any(name in model_name() for name in ('gpt-oss', 'qwen')) else {}
        return ChatGroq(model=model_name(), api_key=os.environ['GROQ_API_KEY'],
                        temperature=0, max_tokens=max_tokens, timeout=timeout, max_retries=0,
                        **reasoning)
    if provider_name() != 'gemini':
        raise SetupError('Choose LLM_PROVIDER=groq or LLM_PROVIDER=gemini.')
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name(),
                                  api_key=os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'),
                                  thinking_level='low', timeout=timeout, max_retries=0, max_tokens=max_tokens)
