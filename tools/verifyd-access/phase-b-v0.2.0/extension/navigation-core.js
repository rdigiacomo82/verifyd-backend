/* Pure controller utilities shared by the browser overlay and future gaze adapters. */
(function initializeVeriFYDNavigationCore(global) {
  'use strict';
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const nextIndex = (previous, length, step = 1) => length > 0
    ? ((previous + step) % length + length) % length
    : -1;
  const movePoint = (point, direction, step, viewport) => {
    const delta = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[direction];
    if (!delta) return { x: point.x, y: point.y };
    return {
      x: clamp(point.x + delta[0] * step, 4, Math.max(4, viewport.width - 4)),
      y: clamp(point.y + delta[1] * step, 4, Math.max(4, viewport.height - 4)),
    };
  };
  const isTyping = (element) => Boolean(element && (
    element.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName)
  ));
  const cleanLabel = (raw, max = 84) => String(raw || '').replace(/\s+/g, ' ').trim().slice(0, max);
  const api = Object.freeze({ clamp, nextIndex, movePoint, isTyping, cleanLabel });
  global.__verifydAccessNavigationCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
