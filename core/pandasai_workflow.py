"""LangGraph owns orchestration; analysis backend is replaceable."""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from core.analysis import PandasAIBackend, SetupError, normalize_response


class State(TypedDict, total=False):
    frame: object
    question: str
    history: list
    profile: dict
    raw_response: object
    result: dict
    error: str
    trace: list


def profile_frame(frame):
    return {'rows': len(frame), 'column_count': len(frame.columns),
            'missing_cells': int(frame.isna().sum().sum()),
            'columns': [{'name': str(c), 'dtype': str(frame[c].dtype),
                         'missing': int(frame[c].isna().sum()),
                         'unique': int(frame[c].nunique())} for c in frame.columns]}


def profile_data(state):
    return {'profile': profile_frame(state['frame']), 'trace': ['profile_data'], 'error': ''}


def validate_request(state):
    error = ''
    if not state.get('question', '').strip():
        error = 'Enter a question about your dataset.'
    elif len(state['question']) > 4000:
        error = 'Keep your question under 4,000 characters.'
    elif state['frame'].empty:
        error = 'Upload a dataset with at least one row.'
    return {'error': error, 'trace': state['trace'] + ['validate_request']}


def build_graph(backend=None, normalizer=normalize_response):
    """Inject any backend exposing analyze(frame, question, history)."""
    backend = backend or PandasAIBackend()

    def analyze(state):
        try:
            response = backend.analyze(state['frame'], state['question'], state.get('history', []))
            return {'raw_response': response, 'trace': state['trace'] + ['pandasai_analysis']}
        except Exception as exc:
            from core.errors import describe_failure
            message = str(exc) if isinstance(exc, SetupError) else describe_failure(exc)
            return {'error': message, 'trace': state['trace'] + ['pandasai_analysis']}

    def format_result(state):
        try:
            result = normalizer(state['raw_response'])
            result['rows_in_dataset'] = len(state['frame'])
            return {'result': result, 'trace': state['trace'] + ['format_result']}
        except Exception:
            return {'error': 'Unsupported analysis result. Try rephrasing your question.',
                    'trace': state['trace'] + ['format_result']}

    def teaching_context(state):
        result = dict(state['result'])
        result['teaching_note'] = 'Rows may represent patients, visits, or measurements. Check the dataset definition before interpreting counts. Patterns show associations, not causes.'
        return {'result': result, 'trace': state['trace'] + ['teaching_context']}

    builder = StateGraph(State)
    for name, fn in [('profile_data', profile_data), ('validate_request', validate_request),
                     ('pandasai_analysis', analyze), ('format_result', format_result), ('teaching_context', teaching_context)]:
        builder.add_node(name, fn)
    builder.add_edge(START, 'profile_data')
    builder.add_edge('profile_data', 'validate_request')
    route = lambda s: 'stop' if s.get('error') else 'next'
    builder.add_conditional_edges('validate_request', route, {'stop': END, 'next': 'pandasai_analysis'})
    builder.add_conditional_edges('pandasai_analysis', route, {'stop': END, 'next': 'format_result'})
    builder.add_conditional_edges('format_result', route, {'stop': END, 'next': 'teaching_context'})
    builder.add_edge('teaching_context', END)
    return builder.compile()


graph = build_graph()
