/* VeriFYD Access Phase B: local iris geometry, calibration, and blink intents.
 * Pure logic — safe to unit test without a camera or website.
 */
(function initVeriFYDGazeCore(global) {
  'use strict';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const middle = (a, b) => (a + b) / 2;
  const mean = (a) => a.reduce((sum, x) => sum + x, 0) / a.length;
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const iris = (landmarks, from) => ({
    x: mean(landmarks.slice(from, from + 4).map(p => p.x)),
    y: mean(landmarks.slice(from, from + 4).map(p => p.y)),
  });
  function eyeReading(pts, eye) {
    const a = pts[eye.a], b = pts[eye.b], top = pts[eye.top], bottom = pts[eye.bottom];
    if (!a || !b || !top || !bottom) return null;
    const width = Math.max(.00001, Math.abs(b.x - a.x));
    const height = Math.max(.00001, Math.abs(bottom.y - top.y));
    const originX = Math.min(a.x, b.x);
    const centerX = pts.length >= eye.iris + 4 ? iris(pts, eye.iris).x : middle(a.x, b.x);
    const centerY = pts.length >= eye.iris + 4 ? iris(pts, eye.iris).y : middle(top.y, bottom.y);
    return { x: (centerX - originX) / width, y: (centerY - top.y) / height,
      aspect: height / Math.max(.00001, distance(a, b)) };
  }
  function sampleLandmarks(pts, categories = []) {
    if (!Array.isArray(pts) || pts.length < 478) return null; // Iris refinement required.
    const a = eyeReading(pts, {a: 33, b: 133, top: 159, bottom: 145, iris: 468});
    const b = eyeReading(pts, {a: 263, b: 362, top: 386, bottom: 374, iris: 473});
    if (!a || !b) return null;
    const blinkEntries = categories.filter(c => /eyeBlink(Left|Right)/.test(c.categoryName || ''));
    // Prefer MediaPipe eyeBlink blendshapes. Eye aspect ratio is only a fallback.
    const blink = blinkEntries.length === 2
      ? blinkEntries.every(c => (c.score || 0) >= .56)
      : a.aspect < .13 && b.aspect < .13;
    return {
      x: clamp(mean([a.x, b.x]), -.3, 1.3),
      y: clamp(mean([a.y, b.y]), -.3, 1.3),
      blink,
      face: true,
    };
  }
  const POINTS = Object.freeze([
    {name:'center', x:.5, y:.5}, {name:'left', x:.17, y:.5},
    {name:'right', x:.83, y:.5}, {name:'up', x:.5, y:.18},
    {name:'down', x:.5, y:.82},
  ]);
  function calibrationFrom(samples) {
    if (!samples || !POINTS.every(p => samples[p.name] && Number.isFinite(samples[p.name].x) && Number.isFinite(samples[p.name].y))) return null;
    const c=samples.center, left=samples.left, right=samples.right, up=samples.up, down=samples.down;
    // Reject noisy calibration when gaze movement did not produce detectable displacement.
    const dx=Math.abs(right.x-left.x), dy=Math.abs(down.y-up.y);
    if (dx < .035 || dy < .035) return null;
    return {cx:c.x, cy:c.y, sx: .66 / (right.x - left.x), sy: .64 / (down.y - up.y)};
  }
  function mapGaze(sample, cal) {
    if (!sample || !cal) return null;
    return {x:clamp(.5+(sample.x-cal.cx)*cal.sx,.02,.98),
      y:clamp(.5+(sample.y-cal.cy)*cal.sy,.02,.98)};
  }
  function createBlinkMachine({minClosedMs=65, maxClosedMs=550, doubleWindowMs=680, debounceMs=750}={}) {
    let closedAt = null, pending = null, lastSelected = -Infinity;
    return {
      update(blink, now, ready) {
        if (!ready) { closedAt=null; pending=null; return []; }
        const actions=[];
        if (blink) {
          if (closedAt===null) closedAt=now;
        } else if (closedAt!==null) {
          const duration=now-closedAt;
          closedAt=null;
          if (duration>=minClosedMs && duration<=maxClosedMs) {
            if (pending!==null && now-pending<=doubleWindowMs) {
              pending=null;
              if (now-lastSelected>debounceMs) {actions.push('SELECT');lastSelected=now;}
            } else pending=now;
          }
        }
        if (pending!==null && now-pending>doubleWindowMs) {
          pending=null;
          actions.push('NEXT');
        }
        return actions;
      },
      reset(){ closedAt=null;pending=null; },
    };
  }
  const api=Object.freeze({sampleLandmarks,POINTS,calibrationFrom,mapGaze,createBlinkMachine});
  global.__verifydAccessGazeCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis!=='undefined'?globalThis:this);
