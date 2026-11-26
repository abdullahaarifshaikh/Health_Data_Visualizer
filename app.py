import io
import os
import secrets
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, session, stream_with_context
from werkzeug.exceptions import HTTPException

load_dotenv()
os.environ.setdefault('SCARF_NO_ANALYTICS', 'true')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).parent / '.cache' / 'tiktoken'))
from core.workflow import graph, profile_frame
from core.education import build_graph
from core.tools import ToolContext
from core.storage import Workspace, session_secret
DEFAULT_GRAPH = graph

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = session_secret()
app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True,
                  SESSION_COOKIE_SAMESITE='Strict', SESSION_COOKIE_SECURE=bool(os.getenv('APP_PROXY_SECRET')))
datasets = {}
workspaces = {}
store_lock = threading.Lock()
TTL = 7200


@app.before_request
def local_requests():
    proxy_secret = os.getenv('APP_PROXY_SECRET')
    if proxy_secret:
        supplied = request.headers.get('X-HealthLens-Proxy', '')
        if not secrets.compare_digest(supplied, proxy_secret):
            return jsonify(error='Use the HealthLens website to access this service.'), 403
        return None
    if request.host.split(':')[0] not in ('127.0.0.1', 'localhost'):
        return jsonify(error='Use localhost to open this app.'), 403
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            return jsonify(error='Cross-origin requests are not allowed.'), 403


def owner():
    if 'owner' not in session:
        session['owner'] = secrets.token_urlsafe(24)
        session.permanent = True
    return session['owner']


def workspace():
    key = owner()
    with store_lock:
        if key not in workspaces:
            workspaces[key] = Workspace(key)
        return workspaces[key]


def persist(data):
    # Save app history and native LangGraph checkpoints in the same workspace DB.
    data['chats'][data['active_chat']] = {k: data[k] for k in ('responses', 'history', 'turns')}
    repository = workspace()
    repository.save_dataset(session['dataset'], data)
    repository.backup()


def get_dataset():
    key = session.get('dataset')
    if not key:
        return None
    repository = workspace()
    with store_lock:
        data = datasets.get(key)
        if data is None:
            data = repository.load_dataset(key)
            if data:
                data.update(updated=time.time(), lock=threading.Lock())
                datasets[key] = data
        if data and data['owner'] == owner():
            data['updated'] = time.time()
            return data
    return None


def save_dataset(frame, name):
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise ValueError('The file has no data rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique after trimming spaces.')
    if len(frame) > 100000 or len(frame.columns) > 150:
        raise ValueError('Use up to 100,000 rows and 150 columns.')
    token = secrets.token_urlsafe(24)
    with store_lock:
        datasets.pop(session.get('dataset'), None)
        if len(datasets) >= 8:
            del datasets[min(datasets, key=lambda k: datasets[k]['updated'])]
        datasets[token] = {'frame': frame, 'name': name, 'history': [], 'turns': [], 'responses': [], 'owner': owner(),
                           'updated': time.time(), 'lock': threading.Lock()}
    session['dataset'] = token
    payload = dataset_payload(datasets[token])
    persist(datasets[token])
    return payload


def dataset_payload(data):
    import json
    frame = data['frame']
    numeric = [str(c) for c in frame.select_dtypes(include='number').columns if not str(c).lower().endswith('_id')]
    suggestions = ['Help me understand this dataset.', 'What do the missing values mean for this analysis?']
    if numeric:
        pair = ['heart_rate_bpm', 'sleep_hours'] if all(c in numeric for c in ['heart_rate_bpm', 'sleep_hours']) else numeric[:2]
        if len(pair) == 2:
            suggestions.insert(0, f'How are {pair[0].replace("_", " ")} and {pair[1].replace("_", " ")} related?')
        suggestions.append(f'Explain the distribution of {numeric[0]}.')
    chats = data.setdefault('chats', {})
    active = data.setdefault('active_chat', secrets.token_urlsafe(12))
    chats[active] = {k: data[k] for k in ('responses', 'history', 'turns')}
    return {'name': data['name'], 'profile': profile_frame(frame),
            'preview': json.loads(frame.head(12).to_json(orient='values', date_format='iso')),
            'columns': list(frame.columns), 'suggestions': suggestions, 'conversation': data['responses'],
            'active_chat': active, 'chats': [{'id': key, 'title': value['responses'][0]['question'][:70] if value['responses'] else 'New chat'} for key, value in reversed(list(chats.items()))]}


@app.post('/api/chats')
def manage_chats():
    data = get_dataset()
    if not data:
        return jsonify(error='Upload a dataset first.'), 400
    if not data['lock'].acquire(blocking=False):
        return jsonify(error='Wait for the current answer before changing chats.'), 409
    try:
        dataset_payload(data)
        body = request.get_json(silent=True) or {}
        action = body.get('action')
        if action == 'select':
            key = body.get('id')
            if key not in data['chats']:
                return jsonify(error='Chat not found.'), 404
            data['active_chat'] = key
            data.update(data['chats'][key])
        elif action == 'clear':
            workspace().clear_chat(session['dataset'], data['active_chat'])
            data.update(responses=[], history=[], turns=[])
        elif action == 'new':
            if data['responses']:
                data['active_chat'] = secrets.token_urlsafe(12)
                data.update(responses=[], history=[], turns=[])
        else:
            return jsonify(error='Unknown chat action.'), 400
        payload = dataset_payload(data)
        persist(data)
        return jsonify(payload)
    finally:
        data['lock'].release()


@app.get('/')
def index():
    return app.send_static_file('index.html')


@app.get('/api/status')
def status():
    import importlib.util
    from core.models import model_name, provider_name, configured
    return jsonify(model=model_name(), provider=provider_name(), configured=configured(),
                   engine=os.getenv('ANALYSIS_BACKEND', 'education'),
                   workflow_version='planned-composable-v2', memory_backend='sqlite+private_blob' if os.getenv('HEALTHLENS_STORAGE_URL') else 'sqlite',
                   pandasai_installed=importlib.util.find_spec('pandasai') is not None)


@app.get('/api/dataset')
def current_dataset():
    data = get_dataset()
    return jsonify(dataset_payload(data) if data else {})


@app.post('/api/upload')
def upload():
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify(error='Choose a CSV or Excel file.'), 400
    suffix = Path(file.filename).suffix.lower()
    try:
        if suffix == '.csv':
            frame = pd.read_csv(file, nrows=100001)
        elif suffix == '.xlsx':
            import zipfile
            raw = file.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError('This workbook expands beyond the supported size.')
            frame = pd.read_excel(io.BytesIO(raw), nrows=100001)
        else:
            return jsonify(error='Supported files: .csv and .xlsx. Excel loads the first worksheet.'), 400
        return jsonify(save_dataset(frame, Path(file.filename).name))
    except ValueError as exc:
        return jsonify(error=str(exc)[:250]), 400
    except Exception:
        return jsonify(error='Unable to read this file. Check its format and headers.'), 400


@app.post('/api/sample')
def sample():
    import random
    rng = random.Random(42)
    rows = []
    for i in range(180):
        age = rng.randint(18, 80)
        rows.append({'record_id': f'S{i+1:03}', 'age_years': age,
                     'group': rng.choice(['Class A', 'Class B', 'Class C']),
                     'heart_rate_bpm': rng.randint(55, 105),
                     'systolic_bp_mmhg': round(100 + age * .35 + rng.gauss(0, 12)),
                     'sleep_hours': round(rng.uniform(4.5, 9), 1) if i % 13 else None})
    return jsonify(save_dataset(pd.DataFrame(rows), 'Synthetic classroom data'))


@app.post('/api/chat')
def chat():
    data = get_dataset()
    if not data:
        return jsonify(error='Upload a dataset first.'), 400
    body = request.get_json(silent=True) or {}
    question = body.get('question', '')
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        return jsonify(error='Enter a question between 1 and 4,000 characters.'), 400
    if not data['lock'].acquire(blocking=False):
        return jsonify(error='An analysis is already running for this dataset.'), 409
    repository = workspace()
    use_checkpoints = graph is DEFAULT_GRAPH and os.getenv('ANALYSIS_BACKEND', 'education') != 'pandasai'
    active_graph = build_graph(checkpointer=repository.checkpointer) if use_checkpoints else graph
    previous = [] if use_checkpoints else [message for turn in data['turns'] for message in turn]
    inputs = {'question': question, 'dataset_id': session['dataset']}
    if not use_checkpoints:
        inputs.update(frame=data['frame'], history=data['history'], messages=previous)
    context = ToolContext(data['frame'], repository, session['dataset'], data['active_chat'])
    config = {'configurable': {'thread_id': session['dataset'] + ':' + data['active_chat']}, 'recursion_limit': 24}

    def remember(state):
        result = state['result']
        data['responses'] = data['responses'] + [{'question': question, 'result': result, 'trace': state.get('trace', [])}]
        data['history'] = data['history'] + [{'question': question, 'answer': result.get('answer', '')}]
        if state.get('messages') and not use_checkpoints:
            data['turns'] = data['turns'] + [state['messages'][len(previous):]]
        persist(data)

    if request.headers.get('Accept') == 'application/x-ndjson':
        import json
        labels = {'understand_question': 'Understanding your question', 'plan_analysis': 'Planning the analysis', 'check_completion': 'Checking that requested work is complete', 'prepare_answer': 'Preparing your answer from the evidence', 'reply_social': 'Replying', 'tutor': 'Choosing the next step', 'tools': 'Calculating or drawing your chart', 'plan_response': 'Checking the data',
                  'calculate_evidence': 'Explaining what the evidence means', 'explain_findings': 'Preparing your answer',
                  'verify_explanation': 'Checking the explanation against the calculations', 'respond': 'Putting your lesson together'}

        @stream_with_context
        def events():
            try:
                yield json.dumps({'type': 'progress', 'message': 'Reading your question'}) + '\n'
                last_step = None
                state = None
                for state in active_graph.stream(inputs, config=config, context=context, stream_mode='values'):
                    trace = state.get('trace', [])
                    step = trace[-1] if trace else None
                    if step and step != last_step:
                        yield json.dumps({'type': 'progress', 'message': labels.get(step, 'Working through your question')}) + '\n'
                        last_step = step
                if state is None or state.get('error'):
                    yield json.dumps({'type': 'error', 'error': state.get('error') if state else 'Unable to start the tutor.'}) + '\n'
                else:
                    remember(state)
                    yield json.dumps({'type': 'result', 'result': state['result'], 'trace': state['trace']}, allow_nan=False) + '\n'
            except Exception as exc:
                from core.errors import describe_failure
                yield json.dumps({'type': 'error', 'error': describe_failure(exc)}) + '\n'
            finally:
                data['lock'].release()
        return Response(events(), content_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    try:
        state = active_graph.invoke(inputs, config=config, context=context)
        if state.get('error'):
            return jsonify(error=state['error'], trace=state['trace']), 422
        result = state['result']
        remember(state)
        return jsonify(result=result, trace=state['trace'])
    finally:
        data['lock'].release()


@app.post('/api/clear')
def clear():
    data = get_dataset()
    if data:
        if not data['lock'].acquire(blocking=False):
            return jsonify(error='Wait for the current answer before removing the dataset.'), 409
        try:
            workspace().delete_dataset(session['dataset'], data['chats'])
            workspace().backup()
        finally:
            data['lock'].release()
    with store_lock:
        datasets.pop(session.get('dataset'), None)
    session.pop('dataset', None)
    return jsonify(ok=True)


@app.errorhandler(Exception)
def handle_error(exc):
    if isinstance(exc, HTTPException):
        return jsonify(error='File exceeds the 10 MB limit.' if exc.code == 413 else exc.description), exc.code
    app.logger.warning('Request failed: %s', type(exc).__name__)
    return jsonify(error='Unable to complete the request. Please try again.'), 500


if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')), debug=False)
