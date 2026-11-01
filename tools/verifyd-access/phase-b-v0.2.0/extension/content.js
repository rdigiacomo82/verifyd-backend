/*
 * VeriFYD Access — Phase A browser layer.
 * Phase B adds opt-in, on-device camera processing using bundled model assets.
 * No image frames leave this extension or reach the visited site.
 * All controls live in the current webpage's ISOLATED extension world.
 */
(function startVeriFYDAccess() {
  'use strict';
  const INSTANCE = '__verifydAccessControllerV1';
  if (globalThis[INSTANCE]) {
    globalThis[INSTANCE].toggle();
    return;
  }
  const nav = globalThis.__verifydAccessNavigationCore;
  const gaze = globalThis.__verifydAccessGazeCore;
  if (!nav || !gaze || !document.body) return;

  const ROOT_ID = 'verifiyd-access-root';
  const SELECTOR = [
    'a[href]', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
    'summary', '[role="button"]', '[role="link"]', '[role="checkbox"]',
    '[role="switch"]', '[role="radio"]', '[role="tab"]', '[role="menuitem"]',
    '[contenteditable="true"]', '[tabindex]:not([tabindex="-1"])',
  ].join(',');
  const state = {
    mode: 'scan',
    scanning: false,
    paused: false,
    dwell: false,
    dwellMs: 3200,
    scanMs: 5000,
    scanChoices: [5000, 3500, 2500],
    step: 25,
    stepChoices: [25, 40, 65],
    speech: false,
    selected: null,
    index: -1,
    input: null,
    shift: false,
    timer: null,
    dwellTimer: null,
    dwellKey: 0,
    location: { x: window.innerWidth / 2, y: window.innerHeight / 2 },
    cameraActive: false,
    blinkMachine: gaze.createBlinkMachine(),
    calibration: null,
    calibrationStep: null,
    calibrationSamples: {},
    smoothPoint: null,
    gazeLastFaceAt: 0,
    gazeLastActionAt: -Infinity,
    confirmGaze: null,
  };

  const root = document.createElement('div');
  root.id = ROOT_ID;
  root.setAttribute('data-verifiyd-access', 'true');
  root.innerHTML = `
    <div class="vfa-highlight" aria-hidden="true"></div>
    <div class="vfa-pointer" aria-hidden="true"><span></span></div>
    <section class="vfa-panel" role="region" aria-label="VeriFYD Access controls">
      <header class="vfa-heading">
        <span class="vfa-logo" aria-hidden="true">✓</span>
        <div class="vfa-heading-text"><strong>VeriFYD Access</strong><small>Website controls · Prototype</small></div>
        <button type="button" data-vfa="close" aria-label="Close VeriFYD Access" title="Close">×</button>
      </header>
      <div class="vfa-status" data-vfa-status role="status" aria-live="polite">Ready. Select Next to locate a page control.</div>
      <div class="vfa-modes" role="group" aria-label="Navigation mode">
        <button type="button" data-vfa="mode-scan" aria-pressed="true">◉ Controls</button>
        <button type="button" data-vfa="mode-cursor" aria-pressed="false">⊕ Pointer</button>
      </div>
      <div class="vfa-actions" role="group" aria-label="Navigation actions">
        <button type="button" data-vfa="next">Next <small>F7</small></button>
        <button type="button" data-vfa="select" class="vfa-primary">Select <small>F8</small></button>
        <button type="button" data-vfa="scan">Auto-scan <small>F9</small></button>
        <button type="button" data-vfa="pause">Pause</button>
      </div>
      <div class="vfa-direction" role="group" aria-label="Move virtual pointer" hidden>
        <button type="button" data-vfa="up" aria-label="Move pointer up">↑</button>
        <div><button type="button" data-vfa="left" aria-label="Move pointer left">←</button><button type="button" data-vfa="right" aria-label="Move pointer right">→</button></div>
        <button type="button" data-vfa="down" aria-label="Move pointer down">↓</button>
      </div>
      <div class="vfa-actions vfa-secondary" role="group" aria-label="Page navigation">
        <button type="button" data-vfa="scroll-up">Scroll ↑</button>
        <button type="button" data-vfa="scroll-down">Scroll ↓</button>
        <button type="button" data-vfa="keyboard">Keyboard</button>
        <button type="button" data-vfa="dwell" aria-pressed="false">Dwell Off</button>
      </div>
      <div class="vfa-actions vfa-secondary" role="group" aria-label="Adaptive speed and narration">
        <button type="button" data-vfa="speed">Scan: Slow</button>
        <button type="button" data-vfa="step">Pointer: Fine</button>
        <button type="button" data-vfa="speech" aria-pressed="false">Speak Off</button>
      </div>
      <div class="vfa-gaze-controls" role="group" aria-label="Camera Eye Gaze">
        <strong>Eye Gaze · Local camera</strong>
        <div class="vfa-actions vfa-secondary">
          <button type="button" data-vfa="gaze-start">Start camera</button>
          <button type="button" data-vfa="gaze-stop" disabled>Camera off</button>
          <button type="button" data-vfa="gaze-calibrate" disabled>Calibrate gaze</button>
          <button type="button" data-vfa="gaze-reset" disabled>Reset calibration</button>
        </div>
        <p data-vfa-gaze-status role="status" aria-live="polite">Camera off. Works without Eye Gaze using the other controls.</p>
        <p class="vfa-gaze-help">After calibration: move pointer with your eyes. Double-blink selects. In Controls mode, single blink advances; double blink selects. Pause or Camera off stops video.</p>
      </div>
      <div class="vfa-calibration" data-vfa-calibration hidden aria-live="polite" aria-label="Eye Gaze calibration">
        <div data-vfa-dot class="vfa-calibration-dot"></div>
        <div data-vfa-calibration-label class="vfa-calibration-label"></div>
      </div>
      <div class="vfa-keyboard" data-vfa-keyboard hidden aria-label="Accessible on-screen keyboard">
        <div class="vfa-kb-header"><strong>On-screen keyboard</strong><button type="button" data-vfa="hide-keyboard" aria-label="Hide keyboard">×</button></div>
        <div class="vfa-keyrows"></div>
      </div>
      <footer class="vfa-footer">F7 Next · F8 Select · F9 Scan · Esc Pause<br />Only this page. Never clicks without selection or opted-in dwell.</footer>
    </section>`;
  document.documentElement.appendChild(root);
  const $ = (selector) => root.querySelector(selector);
  const all = (selector) => [...root.querySelectorAll(selector)];
  const status = $('[data-vfa-status]');
  const marker = $('.vfa-highlight');
  const pointer = $('.vfa-pointer');
  const direction = $('.vfa-direction');
  const keyboard = $('[data-vfa-keyboard]');
  const rows = $('.vfa-keyrows');
  const gazeStatus = $('[data-vfa-gaze-status]');
  const calibrationOverlay = $('[data-vfa-calibration]');
  const calibrationLabel = $('[data-vfa-calibration-label]');
  const calibrationDot = $('[data-vfa-dot]');
  let cameraLoading = false;

  function cameraNote(text) { gazeStatus.textContent = nav.cleanLabel(text, 250); }
  function cameraButtons() {
    $('[data-vfa="gaze-start"]').disabled = state.cameraActive || cameraLoading;
    $('[data-vfa="gaze-stop"]').disabled = !state.cameraActive && !cameraLoading;
    $('[data-vfa="gaze-calibrate"]').disabled = !state.cameraActive || !!state.calibrationStep;
    $('[data-vfa="gaze-reset"]').disabled = !state.cameraActive;
  }
  function resetCalibration() {
    state.calibration = null; state.calibrationStep = null;
    state.calibrationSamples = {}; state.smoothPoint = null;
    state.blinkMachine.reset(); state.confirmGaze = null;
    calibrationOverlay.hidden = true;
    cameraButtons();
  }
  function stopCamera(message = 'Camera off. Switch scanning and keyboard remain available.') {
    cameraLoading = false;
    state.cameraActive = false;
    resetCalibration(); cameraButtons(); cameraNote(message);
    if (globalThis.chrome?.runtime?.sendMessage) {
      chrome.runtime.sendMessage({type:'VFA_CLOSE_CAMERA'}).catch(()=>{});
    }
  }
  function showCalPoint() {
    const point = gaze.POINTS[state.calibrationStep?.index ?? -1];
    if (!point) return;
    calibrationDot.style.left = `${point.x * 100}%`;
    calibrationDot.style.top = `${point.y * 100}%`;
    calibrationLabel.textContent = `Look at the yellow dot: ${point.name} (${state.calibrationStep.index + 1} of 5)`;
  }
  function startCalibration() {
    if (!state.cameraActive || state.calibrationStep) return;
    resetCalibration();
    state.calibrationStep = {index:0, startedAt:performance.now(), samples:[]};
    calibrationOverlay.hidden = false;
    showCalPoint(); cameraButtons();
    cameraNote('Follow the five yellow dots using only your eyes. Keep your head steady.');
  }
  function processCalibration(sample, now) {
    const step = state.calibrationStep;
    if (!step) return;
    const target = gaze.POINTS[step.index];
    if (sample && now - step.startedAt > 550 && !sample.blink) step.samples.push({x:sample.x, y:sample.y});
    if (now - step.startedAt < 2450) return;
    if (step.samples.length < 6) {
      step.startedAt = now; step.samples = [];
      cameraNote(`Face not steady at ${target.name}. Hold still and look at the dot.`);
      return;
    }
    const ordered = step.samples.slice().sort((a,b) => a.x - b.x);
    const x = ordered[Math.floor(ordered.length/2)].x;
    const vertical = step.samples.slice().sort((a,b) => a.y - b.y);
    const y = vertical[Math.floor(vertical.length/2)].y;
    state.calibrationSamples[target.name] = {x, y};
    if (step.index === gaze.POINTS.length - 1) {
      const result = gaze.calibrationFrom(state.calibrationSamples);
      state.calibrationStep = null; calibrationOverlay.hidden = true;
      state.calibration = result; state.blinkMachine.reset(); cameraButtons();
      if (!result) cameraNote('Calibration too small or unstable. Reposition your device and retry.');
      else {
        setMode('cursor');
        cameraNote('Calibrated. Look around to move pointer; double-blink to select. Single blink advances in Controls mode.');
      }
      return;
    }
    step.index++; step.startedAt = now; step.samples = [];
    showCalPoint();
  }
  async function startCamera() {
    if (cameraLoading || state.cameraActive) return;
    if (!globalThis.chrome?.runtime?.sendMessage) {
      cameraNote('Camera feature is available in the installed extension, not this webpage demo.');
      return;
    }
    if (state.paused) resume();
    cameraLoading=true; cameraButtons();
    cameraNote('Opening the trusted VeriFYD camera window; start the camera there.');
    try {
      const result=await chrome.runtime.sendMessage({type:'VFA_OPEN_CAMERA'});
      if (!result?.ok) throw new Error('Extension camera window could not open');
    } catch(_) {cameraNote('Could not open the trusted camera window. Reload the extension and retry.');}
    finally {cameraLoading=false;cameraButtons();}
  }
  function receiveSensorMessage(message) {
    if (!message || !message.type) return;
    if (message.type==='VFA_SENSOR_READY') {
      state.cameraActive=true;
      state.gazeLastFaceAt=performance.now();
      cameraButtons();cameraNote('Camera ready in trusted window. Select Calibrate gaze on this page.');
    } else if (message.type==='VFA_SENSOR_STOPPED') {
      state.cameraActive=false;resetCalibration();cameraButtons();
      cameraNote('Camera stopped. No video is being captured.');
    } else if (message.type==='VFA_SENSOR_SAMPLE' && state.cameraActive && !state.paused) {
      const sample=message.sample;
      if (!sample || !Number.isFinite(sample.x) || !Number.isFinite(sample.y) || typeof sample.blink!=='boolean')return;
      const now=performance.now();
      if (state.calibrationStep) {processCalibration(sample,now);return;}
      if (!state.calibration) return;
      state.gazeLastFaceAt=now;
      const point=gaze.mapGaze(sample,state.calibration);
      if (point && state.mode==='cursor' && !sample.blink) {
        state.smoothPoint=state.smoothPoint
          ? {x:state.smoothPoint.x*.7+point.x*.3,y:state.smoothPoint.y*.7+point.y*.3}
          : point;
        setGazePoint(state.smoothPoint);
      }
      for(const intent of state.blinkMachine.update(sample.blink,now,true)) {
        if (intent==='SELECT')selectCurrent('gaze');
        if (intent==='NEXT' && state.mode==='scan')chooseNext();
      }
    } else if (message.type==='VFA_SENSOR_NO_FACE' && state.cameraActive) {
      state.blinkMachine.reset();
      cameraNote('Face not detected. Recenter in the camera window; selection suspended.');
    }
  }
  function setGazePoint(point) {
    state.location={x:nav.clamp(point.x*window.innerWidth,4,Math.max(4,window.innerWidth-4)),
      y:nav.clamp(point.y*window.innerHeight,4,Math.max(4,window.innerHeight-4))};
    updatePointer();
  }
  function announce(text) {
    status.textContent = nav.cleanLabel(text, 180);
    if (state.speech && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance === 'function') {
      try {
        window.speechSynthesis.cancel();
        const message = new SpeechSynthesisUtterance(status.textContent);
        message.rate = 0.78;
        window.speechSynthesis.speak(message);
      } catch (_) { /* speech is always optional */ }
    }
  }
  function validElement(element) {
    if (!(element instanceof Element) || root.contains(element)) return false;
    if (element.matches(':disabled') || element.closest('[inert],[aria-hidden="true"],[hidden]')) return false;
    const style = window.getComputedStyle(element);
    if (style.visibility === 'hidden' || style.display === 'none' || style.pointerEvents === 'none') return false;
    const r = element.getBoundingClientRect();
    return r.width >= 2 && r.height >= 2;
  }
  function targets() { return [...document.querySelectorAll(SELECTOR)].filter(validElement); }
  function labelOf(el) {
    if (!el) return 'Nothing selected';
    return nav.cleanLabel(el.getAttribute('aria-label') || el.getAttribute('title') ||
      el.getAttribute('placeholder') || el.innerText || el.textContent || el.getAttribute('name') || el.tagName.toLowerCase());
  }
  function hideHighlight() { marker.classList.remove('vfa-visible'); }
  function updateHighlight() {
    if (!state.selected || !state.selected.isConnected || !validElement(state.selected)) {
      hideHighlight(); return;
    }
    const r = state.selected.getBoundingClientRect();
    marker.style.transform = `translate(${Math.round(r.left - 3)}px, ${Math.round(r.top - 3)}px)`;
    marker.style.width = `${Math.max(10, Math.round(r.width + 6))}px`;
    marker.style.height = `${Math.max(10, Math.round(r.height + 6))}px`;
    marker.classList.add('vfa-visible');
  }
  function updatePointer() {
    pointer.style.transform = `translate(${Math.round(state.location.x - 15)}px, ${Math.round(state.location.y - 15)}px)`;
    pointer.classList.toggle('vfa-visible', state.mode === 'cursor');
  }
  function cancelDwell() {
    if (state.dwellTimer) clearTimeout(state.dwellTimer);
    state.dwellTimer = null;
    state.dwellKey++;
  }
  function scheduleDwell() {
    cancelDwell();
    if (!state.dwell || state.paused) return;
    const key = state.dwellKey;
    // Dwell is opt-in and one-shot. It never repeats on an unchanged target.
    state.dwellTimer = setTimeout(() => {
      if (state.dwellKey !== key || state.paused || !state.dwell) return;
      selectCurrent('dwell');
      cancelDwell();
    }, state.dwellMs);
  }
  function setMode(mode) {
    if (state.paused) resume();
    stopScan();
    cancelDwell();
    state.mode = mode;
    state.selected = null;
    hideHighlight();
    direction.hidden = mode !== 'cursor';
    $('[data-vfa="mode-scan"]').setAttribute('aria-pressed', String(mode === 'scan'));
    $('[data-vfa="mode-cursor"]').setAttribute('aria-pressed', String(mode === 'cursor'));
    updatePointer();
    announce(mode === 'cursor' ? 'Pointer mode. Use arrows or direction buttons; Select clicks the page.' :
      'Controls mode. Next highlights a page button or link.');
  }
  function chooseNext() {
    if (state.paused) return;
    if (state.mode !== 'scan') setMode('scan');
    const list = targets();
    if (!list.length) { state.selected = null; hideHighlight(); announce('No accessible page controls found. Try Pointer mode.'); return; }
    const current = list.indexOf(state.selected);
    state.index = nav.nextIndex(current, list.length);
    state.selected = list[state.index];
    // A manual scan step makes the selected target visible before activating it.
    state.selected.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
    updateHighlight();
    announce(`${state.index + 1} of ${list.length}: ${labelOf(state.selected)}`);
    scheduleDwell();
  }
  function scrollPage(directionValue) {
    if (state.paused) return;
    const dist = Math.max(160, Math.round(window.innerHeight * 0.55));
    window.scrollBy({ top: directionValue * dist, behavior: 'smooth' });
    announce(directionValue < 0 ? 'Scrolling upward.' : 'Scrolling downward.');
  }
  function move(directionValue) {
    if (state.paused) return;
    if (state.mode !== 'cursor') setMode('cursor');
    state.location = nav.movePoint(state.location, directionValue, state.step,
      { width: window.innerWidth, height: window.innerHeight });
    updatePointer();
    const hit = document.elementFromPoint(state.location.x, state.location.y);
    announce(hit && !root.contains(hit) ? `Pointer: ${labelOf(hit) || hit.tagName.toLowerCase()}` : 'Pointer moved.');
    scheduleDwell();
  }
  function findActionable(start) {
    if (!start || root.contains(start)) return null;
    const element = start.closest(SELECTOR) || start;
    return validElement(element) ? element : null;
  }
  function performAction(element) {
    if (!element || !validElement(element)) { announce('Selected control is no longer available.'); return; }
    const tag = element.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || element.isContentEditable) {
      // Never offer an on-screen keyboard for secret/password or payment fields.
      const restricted = tag === 'input' && (
        element.type === 'password' || /^(?:cc-|one-time-code)/i.test(element.autocomplete || '')
      );
      const textInput = tag === 'input' && ['text', 'search', 'url', 'tel', 'email'].includes(element.type);
      state.input = restricted ? null : element;
      element.focus({ preventScroll: true });
      if (restricted) {
        keyboard.hidden = true;
        announce('Sensitive field selected. Use the browser or operating-system secure input method.');
      } else if (textInput || tag === 'textarea' || element.isContentEditable) {
        keyboard.hidden = false;
        announce(`Typing into ${labelOf(element)}. Keyboard ready.`);
      } else if (tag === 'input' && ['checkbox', 'radio', 'button', 'submit', 'reset'].includes(element.type)) {
        element.click();
        announce(`Selected ${labelOf(element)}.`);
      } else {
        keyboard.hidden = true;
        announce(`Selected ${labelOf(element)}. Use its native page control.`);
      }
      return;
    }
    try {
      if (typeof element.click === 'function') element.click();
      else element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
      announce(`Selected ${labelOf(element)}.`);
    } catch (_) { announce('This page did not accept the selection. Try its native keyboard navigation.'); }
  }
  function requiresGazeConfirmation(el) {
    if (!el) return false;
    const label = labelOf(el).toLowerCase();
    if (/\b(buy|purchase|checkout|pay|send|submit|delete|remove|transfer|place order|confirm|sign|authorize)\b/i.test(label)) return true;
    return el.matches('button[type="submit"],input[type="submit"],form button:not([type]),[role="button"][aria-label*="Delete"]');
  }
  function selectCurrent(source = 'manual') {
    if (state.paused) return;
    cancelDwell();
    const item = state.mode === 'scan' ? state.selected
      : findActionable(document.elementFromPoint(state.location.x,state.location.y));
    if (!item) {announce('No clickable page element at this position.');return;}
    if (source === 'gaze') {
      const now=performance.now();
      if (now-state.gazeLastActionAt < 850) return;
      if (requiresGazeConfirmation(item)) {
        const pending=state.confirmGaze;
        if (!pending || pending.element!==item || now-pending.time>6500) {
          state.confirmGaze={element:item,time:now};
          announce(`Confirm action: ${labelOf(item)}. Double-blink again within six seconds.`);
          state.gazeLastActionAt=now;return;
        }
      }
      state.gazeLastActionAt=now;
    }
    state.confirmGaze=null;
    performAction(item);
    // After a dwell select, stay on the same target but do not repeat its action.
    if (source === 'dwell') announce(`${status.textContent} Dwell activated once.`);
  }
  function stopScan() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    state.scanning = false;
    $('[data-vfa="scan"]').textContent = 'Auto-scan · F9';
  }
  function toggleScan() {
    if (state.paused) resume();
    if (state.scanning) { stopScan(); announce('Auto-scan stopped.'); return; }
    if (state.mode !== 'scan') setMode('scan');
    stopScan();
    chooseNext();
    state.timer = setInterval(() => { if (!state.paused) chooseNext(); }, state.scanMs);
    state.scanning = true;
    $('[data-vfa="scan"]').textContent = 'Stop scan · F9';
  }
  function pause() {
    stopCamera('Paused: camera off. Select Start camera and recalibrate to resume Eye Gaze.');
    state.paused = true;
    stopScan(); cancelDwell(); hideHighlight();
    $('[data-vfa="pause"]').textContent = 'Resume';
    announce('Paused. No page control will be selected.');
  }
  function resume() {
    state.paused = false;
    $('[data-vfa="pause"]').textContent = 'Pause';
    announce('Ready. Use Next or Pointer controls.');
  }
  function togglePause() { state.paused ? resume() : pause(); }

  function writeText(value, backspace = false) {
    let input = state.input;
    if (!input || !input.isConnected || !validElement(input)) {
      announce('Select a text field on the webpage before using the keyboard.'); return;
    }
    const tag = input.tagName.toLowerCase();
    input.focus({ preventScroll: true });
    if (tag === 'input' || tag === 'textarea') {
      if (tag === 'input' && (input.type === 'password' || /^(?:cc-|one-time-code)/i.test(input.autocomplete || ''))) {
        announce('Sensitive field blocked. Use the browser secure input method.'); return;
      }
      if (tag === 'input' && !['text','search','url','tel','email'].includes(input.type)) {
        announce('This field requires its native input control.'); return;
      }
      const start = input.selectionStart == null ? input.value.length : input.selectionStart;
      const end = input.selectionEnd == null ? start : input.selectionEnd;
      let a = start, b = end;
      if (backspace && a === b) a = Math.max(0, a - 1);
      const updated = input.value.slice(0, a) + (backspace ? '' : value) + input.value.slice(b);
      // The native setter is necessary for React-controlled text fields.
      const proto = tag === 'textarea' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const native = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
      if (native) native.call(input, updated);
      else input.value = updated;
      input.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: backspace ? 'deleteContentBackward' : 'insertText', data: backspace ? null : value }));
      try { input.setSelectionRange(a + (backspace ? 0 : value.length), a + (backspace ? 0 : value.length)); }
      catch (_) { /* email inputs and custom input types do not support selection ranges */ }
    } else if (input.isContentEditable) {
      // Local browser editing command preserves native selection in many editors.
      document.execCommand(backspace ? 'delete' : 'insertText', false, backspace ? null : value);
    }
    announce(backspace ? 'Deleted a character.' : `${value === ' ' ? 'Space' : value} entered.`);
  }
  function keyAction(key) {
    if (key === 'Hide') { keyboard.hidden = true; return; }
    if (key === 'Shift') { state.shift = !state.shift; renderKeys(); return; }
    if (key === 'Backspace') { writeText('', true); return; }
    if (key === 'Space') { writeText(' '); return; }
    if (key === 'Enter') { if (state.input?.tagName === 'TEXTAREA') writeText('\n'); else keyboard.hidden = true; return; }
    writeText(state.shift ? key.toUpperCase() : key.toLowerCase());
    if (state.shift) { state.shift = false; renderKeys(); }
  }
  function renderKeys() {
    rows.replaceChildren();
    const layout = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];
    for (const letters of layout) {
      const row = document.createElement('div'); row.className = 'vfa-keyrow';
      for (const char of letters) {
        const key = document.createElement('button');
        key.type = 'button'; key.className = 'vfa-key';
        key.textContent = state.shift ? char : char.toLowerCase();
        key.setAttribute('aria-label', `Type ${char}`);
        // Prevent page input from losing caret position on mousedown.
        key.addEventListener('mousedown', (event) => event.preventDefault());
        key.addEventListener('click', () => keyAction(char));
        row.appendChild(key);
      }
      rows.appendChild(row);
    }
    const extras = document.createElement('div'); extras.className = 'vfa-keyrow vfa-keyrow-extra';
    for (const name of ['Shift', 'Backspace', 'Space', 'Enter', 'Hide']) {
      const key = document.createElement('button'); key.type = 'button';
      key.className = 'vfa-key'; key.textContent = name;
      key.setAttribute('aria-label', name);
      key.addEventListener('mousedown', (event) => event.preventDefault());
      key.addEventListener('click', () => keyAction(name)); extras.appendChild(key);
    }
    rows.appendChild(extras);
  }
  function showKeyboard() {
    if (!state.input || !state.input.isConnected) {
      const current = document.activeElement;
      if (current && !root.contains(current) && nav.isTyping(current)) state.input = current;
    }
    keyboard.hidden = !keyboard.hidden;
    if (!keyboard.hidden && !state.input) announce('Keyboard opened. Choose a webpage text field first.');
    else if (!keyboard.hidden) announce('Keyboard open. Type on the selected webpage field.');
  }
  function handleButton(event) {
    const button = event.target.closest('button[data-vfa]');
    if (!button || !root.contains(button)) return;
    switch (button.dataset.vfa) {
      case 'close': dispose(); break;
      case 'gaze-start': startCamera(); break;
      case 'gaze-stop': stopCamera(); break;
      case 'gaze-calibrate': startCalibration(); break;
      case 'gaze-reset': resetCalibration(); cameraNote('Calibration cleared. Select Calibrate gaze to retry.'); break;
      case 'mode-scan': setMode('scan'); break;
      case 'mode-cursor': setMode('cursor'); break;
      case 'next': chooseNext(); break;
      case 'select': selectCurrent(); break;
      case 'scan': toggleScan(); break;
      case 'pause': togglePause(); break;
      case 'up': case 'down': case 'left': case 'right': move(button.dataset.vfa); break;
      case 'scroll-up': scrollPage(-1); break;
      case 'scroll-down': scrollPage(1); break;
      case 'keyboard': showKeyboard(); break;
      case 'hide-keyboard': keyboard.hidden = true; break;
      case 'speed':
        state.scanMs = state.scanChoices[(state.scanChoices.indexOf(state.scanMs) + 1) % state.scanChoices.length];
        button.textContent = `Scan: ${state.scanMs === 5000 ? 'Slow' : state.scanMs === 3500 ? 'Medium' : 'Fast'}`;
        if (state.scanning) { stopScan(); toggleScan(); }
        announce(`Scanning speed set to ${state.scanMs / 1000} seconds per target.`);
        break;
      case 'step':
        state.step = state.stepChoices[(state.stepChoices.indexOf(state.step) + 1) % state.stepChoices.length];
        button.textContent = `Pointer: ${state.step === 25 ? 'Fine' : state.step === 40 ? 'Medium' : 'Large'}`;
        announce(`Pointer step is ${state.step} pixels.`);
        break;
      case 'speech':
        state.speech = !state.speech;
        button.setAttribute('aria-pressed', String(state.speech));
        button.textContent = state.speech ? 'Speak On' : 'Speak Off';
        if (!state.speech) try { window.speechSynthesis?.cancel(); } catch (_) {}
        announce(state.speech ? 'Spoken guidance enabled.' : 'Spoken guidance disabled.');
        break;
      case 'dwell':
        state.dwell = !state.dwell;
        button.setAttribute('aria-pressed', String(state.dwell));
        button.textContent = state.dwell ? 'Dwell On' : 'Dwell Off';
        state.dwell ? scheduleDwell() : cancelDwell();
        announce(state.dwell ? 'Dwell activated (3.2 seconds). Select with care.' : 'Dwell disabled.');
        break;
      default: break;
    }
  }
  function onKeyDown(event) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const pageTyping = nav.isTyping(document.activeElement) && !root.contains(document.activeElement);
    if (event.key === 'Escape') {
      if (!state.paused) { event.preventDefault(); event.stopPropagation(); pause(); }
      return;
    }
    if (event.key === 'F7') { event.preventDefault(); event.stopPropagation(); chooseNext(); }
    else if (event.key === 'F8') { event.preventDefault(); event.stopPropagation(); selectCurrent(); }
    else if (event.key === 'F9') { event.preventDefault(); event.stopPropagation(); toggleScan(); }
    else if (!pageTyping && !state.paused && state.mode === 'cursor') {
      const map = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
      if (map[event.key]) { event.preventDefault(); event.stopPropagation(); move(map[event.key]); }
    }
  }
  function onScroll() { updateHighlight(); }
  function onResize() {
    state.location = {
      x: nav.clamp(state.location.x, 4, Math.max(4, window.innerWidth - 4)),
      y: nav.clamp(state.location.y, 4, Math.max(4, window.innerHeight - 4)),
    };
    updateHighlight(); updatePointer();
  }
  function dispose() {
    stopCamera();
    stopScan(); cancelDwell();
    if (state.speech) try { window.speechSynthesis?.cancel(); } catch (_) {}
    document.removeEventListener('keydown', onKeyDown, true);
    window.removeEventListener('scroll', onScroll, true);
    window.removeEventListener('resize', onResize);
    document.removeEventListener('visibilitychange', onVisibility);
    root.removeEventListener('click', handleButton);
    if (globalThis.chrome?.runtime?.onMessage) chrome.runtime.onMessage.removeListener(receiveSensorMessage);
    root.remove();
    globalThis[INSTANCE] = null;
  }
  function onVisibility() { if (document.hidden) pause(); }
  function toggle() { dispose(); }
  // Future camera / switch adapters call this intent API from the extension's
  // isolated world. The host webpage cannot call it or send arbitrary messages.
  function dispatchIntent(intent) {
    if (!intent || typeof intent !== 'object') return false;
    switch (intent.type) {
      case 'NEXT': chooseNext(); return true;
      case 'SELECT': selectCurrent(); return true;
      case 'SCAN': toggleScan(); return true;
      case 'PAUSE': pause(); return true;
      case 'RESUME': resume(); return true;
      case 'SCROLL': scrollPage(intent.direction === 'up' ? -1 : 1); return true;
      case 'MODE': if (intent.mode === 'scan' || intent.mode === 'cursor') { setMode(intent.mode); return true; } return false;
      case 'MOVE': if (['left', 'right', 'up', 'down'].includes(intent.direction)) { move(intent.direction); return true; } return false;
      // Trusted isolated-world adapter input. This is not exposed to website JS.
      case 'GAZE_POINT':
        if (typeof intent.x === 'number' && typeof intent.y === 'number' && !state.paused) {
          setGazePoint({x:nav.clamp(intent.x,0,1),y:nav.clamp(intent.y,0,1)});return true;
        }
        return false;
      default: return false;
    }
  }

  root.addEventListener('click', handleButton);
  if (globalThis.chrome?.runtime?.onMessage) chrome.runtime.onMessage.addListener(receiveSensorMessage);
  document.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('scroll', onScroll, true);
  window.addEventListener('resize', onResize);
  document.addEventListener('visibilitychange', onVisibility);
  renderKeys(); updatePointer(); cameraButtons();
  announce('Ready. Use Next, or press F7 to highlight a webpage control.');
  globalThis[INSTANCE] = Object.freeze({ toggle, dispose, dispatchIntent });
})();
