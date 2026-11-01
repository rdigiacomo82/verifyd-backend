"""Optional Chromium MV3 integration smoke: installed action + trusted camera popup.
Only uses a local test page. No camera frames requested and no model assets required.
"""
import tempfile, threading, http.server, pathlib, time
from playwright.sync_api import sync_playwright
EXT=pathlib.Path(__file__).resolve().parents[1] / 'extension'
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        data=b'<html><body><h1>Sample local test page</h1><button id="start">Hello</button></body></html>'
        self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
  with tempfile.TemporaryDirectory() as profile:
    with sync_playwright() as p:
      context=p.chromium.launch_persistent_context(profile,headless=True,executable_path='/usr/bin/chromium',
        args=['--no-sandbox',f'--disable-extensions-except={EXT}',f'--load-extension={EXT}'])
      page=context.new_page();page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='load')
      for _ in range(30):
        if context.service_workers:break
        time.sleep(.1)
      if not context.service_workers:raise RuntimeError('Headless Chromium extension service worker not loaded')
      worker=context.service_workers[0]
      active=page.evaluate('window.location.href')
      result=worker.evaluate('''async()=>{
          const all=await chrome.tabs.query({});
          const t=all.find(t=>t.url&&t.url.startsWith('http://127.0.0.1:'));
          if(!t) return 'tab missing';
          await chrome.scripting.insertCSS({target:{tabId:t.id},files:['content.css']});
          await chrome.scripting.executeScript({target:{tabId:t.id},files:['navigation-core.js','gaze-core.js','content.js'],world:'ISOLATED'});
          return 'injected';
      }''')
      assert result=='injected',result
      assert page.locator('[data-vfa="gaze-start"]').count()==1
      assert page.locator('[data-vfa="gaze-calibrate"]').is_disabled()
      page.locator('[data-vfa="gaze-start"]').click()
      # Without the real toolbar click, background has not authorized this tab.
      page.wait_for_timeout(300)
      assert 'Could not open' in page.locator('[data-vfa-gaze-status]').inner_text()
      context.close()
  print('PASS installed Chromium extension page and permission boundary')
finally:server.shutdown()
