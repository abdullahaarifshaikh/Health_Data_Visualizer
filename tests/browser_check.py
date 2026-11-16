"""Local browser smoke test. Run after starting app.py; not an API model test."""
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server
from app import app
import os
# Keep this UI check deterministic and free of live model calls, even after setup.
os.environ['GOOGLE_API_KEY'] = ''
os.environ['GEMINI_API_KEY'] = ''
from playwright.sync_api import sync_playwright

out = Path('artifacts')
out.mkdir(exist_ok=True)
server = make_server('127.0.0.1', 5019, app, threaded=True)
threading.Thread(target=server.serve_forever, daemon=True).start()
with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:5019', wait_until='domcontentloaded')
    page.get_by_role('heading', name='Make sense of your data.').wait_for()
    page.screenshot(path=str(out / 'welcome-desktop.png'), full_page=True)
    page.get_by_role('button', name='Explore sample data').click()
    page.locator('#overview').wait_for(state='visible')
    assert page.locator('#rows').inner_text() == '180'
    page.locator('summary').filter(has_text='Dataset preview').click()
    assert page.locator('#preview-table tbody tr').count() == 12
    page.screenshot(path=str(out / 'dataset-desktop.png'), full_page=True)
    # Check the actual unconfigured-key failure in the UI.
    page.locator('#question').fill('What is the average age_years?')
    page.locator('#send').click()
    page.locator('.response.error').wait_for()
    assert 'GOOGLE_API_KEY' in page.locator('.response.error').inner_text()
    page.set_viewport_size({'width': 390, 'height': 844})
    page.screenshot(path=str(out / 'dataset-mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors, errors
    browser.close()
server.shutdown()
print('Browser checks passed: welcome, sample, preview, missing-key error, mobile overflow.')
