"""Per-browser SQLite workspace with authenticated, private off-machine backups."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from langchain_core.messages import messages_from_dict, messages_to_dict
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(os.getenv('HEALTHLENS_DATA_DIR', str(Path(__file__).resolve().parents[1] / '.state')))
ROOT.mkdir(parents=True, exist_ok=True)
CHUNK = 2 * 1024 * 1024


def session_secret():
    configured = os.getenv('SESSION_SECRET')
    if configured:
        return configured
    path = ROOT / 'session.key'
    try:
        with path.open('x') as file:
            file.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    return path.read_text().strip()


class Workspace:
    def __init__(self, owner, root=None, remote=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{20,80}', owner):
            raise ValueError('Invalid workspace.')
        self.owner = owner
        self.root = Path(root or ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / (owner + '.sqlite')
        self.remote = remote if remote is not None else os.getenv('HEALTHLENS_STORAGE_URL', '')
        self.lock = threading.RLock()
        if self.remote:
            self.restore()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.execute('PRAGMA busy_timeout=30000')
        self.conn.execute('PRAGMA secure_delete=ON')
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, dataset TEXT NOT NULL, chat TEXT NOT NULL, payload TEXT NOT NULL);
        ''')
        self.conn.commit()
        self.checkpoint_conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.checkpoint_conn.execute('PRAGMA secure_delete=ON')
        self.checkpointer = SqliteSaver(self.checkpoint_conn)
        self.checkpointer.setup()

    def request(self, key, body=None):
        url = self.remote + ('&' if '?' in self.remote else '?') + urllib.parse.urlencode({'key': self.owner + '/' + key})
        req = urllib.request.Request(url, data=body, method='POST' if body is not None else 'GET',
             headers={'X-HealthLens-Storage': os.environ.get('APP_PROXY_SECRET', ''), 'Content-Type': 'application/octet-stream'})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                return response.read(CHUNK + 65536)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and body is None:
                return None
            raise RuntimeError('Persistent memory storage is unavailable. Please retry.') from None

    def restore(self):
        manifest = self.request('manifest')
        if manifest is None:
            return
        info = json.loads(manifest)
        if not re.fullmatch(r'[a-f0-9]{32}', info['version']) or not 1 <= info['parts'] <= 128:
            raise RuntimeError('Invalid saved workspace.')
        chunks = [self.request(f'{info["version"]}/{i}') for i in range(info['parts'])]
        if any(chunk is None for chunk in chunks):
            raise RuntimeError('Saved workspace is incomplete. Please retry.')
        packed = b''.join(chunks)
        if hashlib.sha256(packed).hexdigest() != info['sha256']:
            raise RuntimeError('Saved workspace checksum mismatch.')
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(256 * 1024 * 1024 + 1)
        if len(raw) > 256 * 1024 * 1024 or not raw.startswith(b'SQLite format 3'):
            raise RuntimeError('Saved workspace exceeds the supported size.')
        self.path.write_bytes(raw)

    def backup(self):
        if not self.remote:
            return
        with self.lock:
            target = sqlite3.connect(':memory:')
            try:
                self.conn.backup(target)
                packed = gzip.compress(target.serialize(), compresslevel=3)
            finally:
                target.close()
            parts = [packed[i:i + CHUNK] for i in range(0, len(packed), CHUNK)]
            version = secrets.token_hex(16)
            for index, part in enumerate(parts):
                self.request(f'{version}/{index}', part)
            info = {'version': version, 'parts': len(parts), 'sha256': hashlib.sha256(packed).hexdigest()}
            self.request('manifest', json.dumps(info).encode())

    def save_dataset(self, dataset_id, data):
        chats = {}
        for key, chat in data.get('chats', {}).items():
            chats[key] = {k: chat.get(k, []) for k in ('responses', 'history')}
            chats[key]['turns'] = [messages_to_dict(turn) for turn in chat.get('turns', [])]
        payload = {'name': data['name'], 'frame': data['frame'].to_json(orient='table', date_format='iso'),
                   'active_chat': data['active_chat'], 'chats': chats}
        with self.lock:
            self.conn.execute('INSERT OR REPLACE INTO datasets VALUES (?, ?)', (dataset_id, json.dumps(payload, allow_nan=False)))
            self.conn.commit()

    def load_dataset(self, dataset_id):
        with self.lock:
            row = self.conn.execute('SELECT payload FROM datasets WHERE id=?', (dataset_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        chats = payload['chats']
        for chat in chats.values():
            chat['turns'] = [messages_from_dict(turn) for turn in chat.get('turns', [])]
        active = chats[payload['active_chat']]
        return {'frame': pd.read_json(io.StringIO(payload['frame']), orient='table'), 'name': payload['name'],
                'owner': self.owner, 'chats': chats, 'active_chat': payload['active_chat'], **active}

    def put_artifact(self, dataset, chat, artifact):
        key = secrets.token_hex(12)
        with self.lock:
            self.conn.execute('INSERT INTO artifacts VALUES (?, ?, ?, ?)',
                (key, dataset, chat, json.dumps(artifact, allow_nan=False, default=str)))
            self.conn.commit()
        return key

    def artifact(self, key, dataset, chat):
        with self.lock:
            row = self.conn.execute('SELECT payload FROM artifacts WHERE id=? AND dataset=? AND chat=?',
                                     (key, dataset, chat)).fetchone()
        if not row:
            raise ValueError('Result not found in this chat. Calculate it first.')
        return json.loads(row[0])

    def clear_chat(self, dataset, chat):
        self.checkpointer.delete_thread(dataset + ':' + chat)
        with self.lock:
            self.conn.execute('DELETE FROM artifacts WHERE dataset=? AND chat=?', (dataset, chat))
            self.conn.commit()

    def delete_dataset(self, dataset, chats):
        for chat in chats:
            self.clear_chat(dataset, chat)
        with self.lock:
            self.conn.execute('DELETE FROM datasets WHERE id=?', (dataset,))
            self.conn.commit()

    def close(self):
        self.checkpoint_conn.close()
        self.conn.close()
