"""Browser acceptance check with real calculations and a simulated language model."""
import json
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import app as web
from core.education import build_graph
from playwright.sync_api import sync_playwright


class BrowserModel:
    def bind_tools(self, tools): return self
    def invoke(self, messages):
        question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        def call(name,args,id):
            return AIMessage(content='', tool_calls=[{'name':name,'args':args,'id':id}])
        current = messages[messages.index(next(m for m in reversed(messages) if isinstance(m, HumanMessage))):]
        outputs = [json.loads(m.content) for m in current if isinstance(m,ToolMessage)]
        if 'matrix' in question.lower():
            if not outputs: return call('calculate_statistics', {'operation':'pearson','columns':['heart_rate_bpm','sleep_hours','age_years']}, 'matrix')
            if not any(o.get('chart_type')=='heatmap' for o in outputs): return call('correlation_heatmap', {'source_result_id':outputs[0]['result_id']}, 'heatmap')
            return AIMessage(content=json.dumps({'answer':'This heatmap shows calculated pairwise correlations; it does not adjust for other variables.', 'followups':['How does the age and heart rate correlation compare across groups?']}))
        if 'pie' in question.lower():
            if not outputs: return call('pie_chart',{'x':'age_years'},'pie')
            return AIMessage(content=json.dumps({'answer':'The pie shows the **share of records** in each labeled age range. These are equal-width bins, not clinical categories.', 'followups':['How does age vary between the dataset groups?']}))
        if 'relationship' in question.lower():
            if not outputs: return call('calculate_statistics',{'operation':'pearson','columns':['heart_rate_bpm','sleep_hours']},'stats')
            if not any(o.get('chart_type')=='scatter' for o in outputs): return call('scatter_plot',{'x':'heart_rate_bpm','y':'sleep_hours'},'scatter')
            facts=outputs[0]
            return AIMessage(content=json.dumps({'answer':f'**What the data shows**\n\nPearson correlation is **{facts["pearson_r"]:.3f}**, using **{facts["rows_used"]} complete records**. We excluded {facts["rows_excluded"]} records. This does not establish causation. <script>window.UNSAFE_MARKDOWN=true</script>', 'followups':['What does the heart rate correlation mean in plain language?']}))
        return AIMessage(content=json.dumps({'answer':'The previous correlation describes the observed pattern; it does not establish cause.\n\n| Topic | Explanation |\n| --- | --- |\n| Measurements | '+'A long explanation about recorded measurements. '*12+'|\n| Column | '+'very_long_column_name_'*30+' |', 'followups':['How many sleep hours values are missing?']}))


web.graph = build_graph(BrowserModel())
out = Path('artifacts')
out.mkdir(exist_ok=True)
server = make_server('127.0.0.1', 5019, web.app, threaded=True)
threading.Thread(target=server.serve_forever, daemon=True).start()
with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1050}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:5019', wait_until='domcontentloaded')
    assert page.locator('#chat-form').bounding_box()['height'] <= 60
    assert page.locator('.content').bounding_box()['width'] <= 816
    page.locator('#question').fill('First line\nSecond line\nThird line\nFourth line')
    assert page.locator('#question').bounding_box()['height'] > 70
    page.locator('#question').fill('')
    page.get_by_role('button', name='Explore sample data').click()
    page.locator('#view-data').wait_for(state='visible')
    page.wait_for_function("!document.querySelector('#view-data').disabled")
    assert not page.locator('#overview').is_visible()
    page.locator('#view-data').click()
    assert page.locator('#rows').inner_text() == '180'
    page.locator('#close-data').click()
    page.locator('#sidebar-toggle').click()
    assert page.locator('body').evaluate("el => el.classList.contains('sidebar-collapsed')")
    page.locator('#sidebar-toggle').click()
    page.locator('#question').fill('What is the relationship between heart rate and sleep hours? What conclusions can we draw?')
    page.locator('#send').click()
    page.wait_for_function("document.querySelector('.interactive-chart')?.chartInstance != null")
    assert page.locator('.lesson-text strong').count() >= 2
    assert page.locator('.interactive-chart canvas').count() > 0
    assert not page.evaluate('window.UNSAFE_MARKDOWN === true')
    page.screenshot(path=str(out / 'education-desktop.png'), full_page=True)
    page.get_by_role('button', name='What does the heart rate correlation mean in plain language?', exact=True).click()
    page.wait_for_function("document.querySelectorAll('.lesson-text').length === 2")
    assert 'previous correlation' in page.locator('.lesson-text').last.inner_text()
    assert page.locator('.markdown-table').count() == 1
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth')
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert page.locator('.markdown-table').evaluate('el => el.getBoundingClientRect().right <= innerWidth')
    page.screenshot(path=str(out / 'table-overflow-mobile.png'), full_page=True)
    page.set_viewport_size({'width': 1440, 'height': 1050})
    page.reload(wait_until='networkidle')
    assert page.locator('.lesson-text').count() == 2
    page.locator('#question').fill('Make a pie chart for age and explain it.')
    page.locator('#send').click()
    page.wait_for_function("document.querySelectorAll('.interactive-chart canvas').length >= 2")
    page.locator('#new-chat').click()
    page.wait_for_function("document.querySelectorAll('.lesson-text').length === 0")
    assert page.locator('#dataset-name').inner_text() == 'Synthetic classroom data'
    page.locator('#question').fill('Show a correlation matrix for heart rate, sleep hours and age.')
    page.locator('#send').click()
    page.wait_for_function("document.querySelector('.interactive-chart')?.chartInstance != null")
    assert page.locator('.interactive-chart').evaluate("el => el.chartInstance.getOption().series[0].type") == 'heatmap'
    page.locator('#chat-list button').nth(1).click()
    page.wait_for_function("document.querySelectorAll('.lesson-text').length === 3")
    page.locator('#clear').click()
    page.wait_for_function("document.querySelectorAll('.lesson-text').length === 0")
    page.reload(wait_until='networkidle')
    assert page.locator('.lesson-text').count() == 0
    assert page.locator('#dataset-name').inner_text() == 'Synthetic classroom data'
    page.locator('#sidebar-toggle').click()
    page.set_viewport_size({'width': 390, 'height': 844})
    page.screenshot(path=str(out / 'education-mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors, errors
    browser.close()
server.shutdown()
print('Browser checks passed: explanation, interactive scatter/pie, follow-up, reload memory, safe Markdown, mobile overflow.')
