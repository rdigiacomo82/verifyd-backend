"""Smoke test the actual web controls with Chromium. No network needed."""
from pathlib import Path
from playwright.sync_api import sync_playwright

EXT = Path(__file__).resolve().parents[1] / "extension"


def test_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
        page = browser.new_page(viewport={"width": 1140, "height": 800})
        page.set_content('''<!doctype html><html><body style="font-family:Arial;margin:35px">
          <h1>VeriFYD Access Smoke Test</h1>
          <a id="test-link" href="#success">Jump to success</a>
          <button id="test-button" onclick="document.querySelector('#result').textContent='clicked'">Press the page button</button>
          <label for="test-input">Sample input</label><input id="test-input" value="" />
          <label for="test-password">Private password</label><input id="test-password" type="password" value="secret" />
          <label for="test-email">Email input</label><input id="test-email" type="email" value="" />
          <label for="test-checkbox">Mark read</label><input id="test-checkbox" type="checkbox" />
          <p id="result">not clicked</p><div style="height:1400px"></div><p id="success">Bottom content</p>
          </body></html>''')
        page.add_style_tag(path=str(EXT / 'content.css'))
        page.add_script_tag(path=str(EXT / 'navigation-core.js'))
        page.add_script_tag(path=str(EXT / 'gaze-core.js'))
        page.add_script_tag(path=str(EXT / 'content.js'))
        assert page.locator('#verifiyd-access-root .vfa-panel').count() == 1
        assert page.locator('[data-vfa="mode-scan"]').get_attribute('aria-pressed') == 'true'
        assert page.locator('[data-vfa="dwell"]').get_attribute('aria-pressed') == 'false'
        assert page.locator('[data-vfa="gaze-start"]').count() == 1
        assert page.locator('[data-vfa="gaze-calibrate"]').is_disabled()
        page.locator('[data-vfa="gaze-start"]').click()
        assert 'installed extension' in page.locator('[data-vfa-gaze-status]').inner_text()


        page.locator('[data-vfa="next"]').click()
        assert 'Jump to success' in page.locator('[data-vfa-status]').inner_text()
        page.locator('[data-vfa="next"]').click()
        assert 'Press the page button' in page.locator('[data-vfa-status]').inner_text()
        page.locator('[data-vfa="select"]').click()
        assert page.locator('#result').inner_text() == 'clicked'
        page.screenshot(path=str(EXT.parent / 'PREVIEW.png'), full_page=False)
        page.locator('[data-vfa="next"]').click()
        page.locator('[data-vfa="select"]').click()
        assert page.locator('[data-vfa-keyboard]').is_visible()
        page.locator('#verifiyd-access-root .vfa-key').filter(has_text='h').first.click()
        page.locator('#verifiyd-access-root .vfa-key').filter(has_text='i').first.click()
        assert page.locator('#test-input').input_value() == 'hi'
        page.locator('#verifiyd-access-root .vfa-key').filter(has_text='Backspace').first.click()
        assert page.locator('#test-input').input_value() == 'h'
        # Password fields must never be handled by the prototype keyboard.
        page.locator('[data-vfa="next"]').click()
        page.locator('[data-vfa="select"]').click()
        assert 'Sensitive field' in page.locator('[data-vfa-status]').inner_text()
        assert page.locator('#test-password').input_value() == 'secret'
        # Email input types do not support setSelectionRange; they must not crash.
        page.locator('[data-vfa="next"]').click()
        page.locator('[data-vfa="select"]').click()
        page.locator('#verifiyd-access-root .vfa-key').filter(has_text='a').first.click()
        assert page.locator('#test-email').input_value() == 'a'
        page.locator('[data-vfa="next"]').click()
        page.locator('[data-vfa="select"]').click()
        assert page.locator('#test-checkbox').is_checked()
        assert page.evaluate('window.__verifydAccessControllerV1.dispatchIntent({ type: "PAUSE" })') is True
        assert 'Paused' in page.locator('[data-vfa-status]').inner_text()
        assert page.evaluate('window.__verifydAccessControllerV1.dispatchIntent({ type: "RESUME" })') is True

        page.locator('[data-vfa="mode-cursor"]').click()
        assert page.evaluate('window.__verifydAccessControllerV1.dispatchIntent({type: "GAZE_POINT", x: .31, y: .42})')
        assert page.locator('.vfa-pointer').get_attribute('style') is not None

        assert page.locator('.vfa-pointer').evaluate('(e)=>e.classList.contains("vfa-visible")')
        page.locator('[data-vfa="up"]').click()
        assert page.locator('[data-vfa="mode-cursor"]').get_attribute('aria-pressed') == 'true'
        page.locator('[data-vfa="pause"]').click()
        assert 'Paused' in page.locator('[data-vfa-status]').inner_text()
        page.locator('[data-vfa="close"]').click()
        assert page.locator('#verifiyd-access-root').count() == 0
        # Re-injection should create exactly one fresh panel.
        page.add_script_tag(path=str(EXT / 'navigation-core.js'))
        page.add_script_tag(path=str(EXT / 'gaze-core.js'))
        page.add_script_tag(path=str(EXT / 'content.js'))
        assert page.locator('#verifiyd-access-root').count() == 1
        # Re-activation should close existing panel instead of adding another.
        page.add_script_tag(path=str(EXT / 'content.js'))
        assert page.locator('#verifiyd-access-root').count() == 0
        browser.close()


if __name__ == '__main__':
    test_browser()
    print('PASS Chromium browser smoke test')
