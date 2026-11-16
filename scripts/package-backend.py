"""Allowlist backend source; never package credentials, datasets, or caches."""
import base64
import io
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = [root / 'app.py', root / 'requirements.txt']
files += list((root / 'core').glob('*.py'))
files += list((root / 'cloud').glob('*'))
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for file in files:
        if file.is_file():
            archive.add(file, arcname='healthlens/' + file.relative_to(root).as_posix())
print(base64.b64encode(buffer.getvalue()).decode())
