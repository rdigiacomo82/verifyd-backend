// Circle 15 isolated Android TWA PostMessage LAB ONLY — no API access, login, or enrollment.
'use strict';
const EXPECTED_ORIGIN = 'https://verifyd-circle15-bridge-lab.onrender.com';
const SCHEMA = 'verifyd.twa.native-bridge.v1';
const PACKAGE = 'com.verifyd.circle.pilot.dev';
const status = document.getElementById('status');
const detail = document.getElementById('detail');
const check = document.getElementById('check');
let port = null;
let pending = null;
function parse(raw) {
  try { return typeof raw === 'string' ? JSON.parse(raw) : raw; }
  catch { return null; }
}
function validHello(value) {
  return value?.schema === SCHEMA && value?.type === 'hello' &&
    value.package_name === PACKAGE &&
    /^[0-9a-f]{8}-[0-9a-f-]{27,40}$/.test(value.installation_id || '') &&
    /^[A-Za-z0-9_-]{100,256}$/.test(value.public_key_spki_b64 || '');
}
function finishPending(message) {
  if (!pending) return;
  clearTimeout(pending.timer);
  pending = null;
  if (message) detail.textContent = message;
}
function checkNativeStatus() {
  if (!port || pending) return;
  const req = 'req_' + Array.from(crypto.getRandomValues(new Uint8Array(12)), b => b.toString(16).padStart(2, '0')).join('');
  const timer = setTimeout(() => {
    if (pending?.req === req) { finishPending('Native response timed out; the bridge is not yet validated end-to-end.'); }
  }, 12000);
  pending = { req, timer };
  try { port.postMessage({ schema: SCHEMA, type: 'status', request_id: req }); }
  catch { finishPending('Unable to send native status request.'); }
}
check.addEventListener('click', checkNativeStatus);
window.addEventListener('message', event => {
  if (location.origin !== EXPECTED_ORIGIN || location.pathname !== '/voice/' && location.pathname !== '/voice') return;
  if (event.origin !== EXPECTED_ORIGIN || event.ports?.length !== 1 || !validHello(parse(event.data))) return;
  if (port) { finishPending(); try { port.close(); } catch { /* best effort */ } }
  port = event.ports[0];
  port.onmessage = messageEvent => {
    const value = parse(messageEvent.data);
    if (!pending || value?.schema !== SCHEMA || value.type !== 'result' ||
        value.action !== 'status' || value.request_id !== pending.req) return;
    finishPending(value.registered === true
      ? 'Android replied. A local device token exists; this does NOT prove backend session binding.'
      : 'Android replied. The native pilot is not currently enrolled.');
  };
  port.start?.();
  status.textContent = 'Verified native message channel received.';
  detail.textContent = 'Use Check local native status to test a native round trip. No login tokens are exchanged.';
  check.disabled = false;
});
