import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, MessageSquareText, X } from 'lucide-react';

// VeriFYD-owned emergency messaging WORKFLOW SIMULATOR, not an emergency service.
// Never send this to RapidSOS / PSAP / 911 / SMS, or production Circle Distress.
const enabled = () => import.meta.env.DEV
  && import.meta.env.VITE_VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED === 'true'
  && typeof window !== 'undefined'
  && ['localhost', '127.0.0.1'].includes(window.location.hostname);
const base = () => String(import.meta.env.VITE_VERIFYD_EMERGENCY_WORKFLOW_LAB_API || '').replace(/\/+$/, '');
const scenarios = [
  ['medical', 'Medical scenario'],
  ['fire', 'Fire scenario'],
  ['safety', 'Personal safety scenario'],
  ['other', 'Other scenario'],
];
const phrases = [
  'TEST ONLY: I cannot speak clearly.',
  'TEST ONLY: I am using Eye Gaze to type.',
  'TEST ONLY: This is a fictitious medical scenario.',
  'TEST ONLY: Please ask me one question at a time.',
];

export default function TrustVoiceEmergencyWorkflowLab({ open, onClose, sessionToken }) {
  const [scenario, setScenario] = useState('medical');
  const [incident, setIncident] = useState(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(false);
  const closeRef = useRef(null);
  const session = useRef('');
  session.current = sessionToken || '';

  const request = useCallback(async (path, method = 'GET', body) => {
    if (!enabled()) throw new Error('Training is unavailable in this build.');
    const apiBase = base();
    if (!/^https?:\/\//.test(apiBase)) throw new Error('A separate training backend is not configured.');
    const res = await fetch(`${apiBase}/trust-voice/emergency-workflow-lab${path}`, {
      method,
      headers: { Authorization: `Bearer ${session.current}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
      cache: 'no-store',
    });
    const result = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(res.status === 404 ? 'Practice service is not enabled for this test account.' :
      res.status === 401 ? 'Your VeriFYD session needs to be renewed.' :
      `Practice request failed (${res.status}).`);
    return result;
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    setIncident(null); setError(''); setMessage(''); setConnected(false);
    closeRef.current?.focus();
    if (!sessionToken) { setError('A signed-in Circle session is required.'); return undefined; }
    let active = true;
    request('/status').then((v) => {
      if (active) setConnected(v.mode === 'TRAINING_SIMULATION_ONLY' && v.dispatch_connected === false);
    }).catch((e) => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [open, sessionToken, request]);

  async function execute(task) {
    setBusy(true); setError('');
    try { await task(); } catch (e) { setError(e.message || 'Practice operation failed.'); } finally { setBusy(false); }
  }
  const start = () => execute(async () => {
    const v = await request('/incidents', 'POST', { scenario, confirm_training_only: true });
    if (v.dispatch_connected || v.public_emergency_message_delivered || v.service_type !== 'TRAINING_SIMULATION_ONLY') {
      throw new Error('Invalid training response; no further actions were performed.');
    }
    setIncident(v);
  });
  const send = () => execute(async () => {
    if (!incident || !message.trim()) return;
    const v = await request(`/incidents/${encodeURIComponent(incident.id)}/messages`, 'POST', { message: message.trim() });
    if (v.dispatch_connected || v.public_emergency_message_delivered || v.service_type !== 'TRAINING_SIMULATION_ONLY') {
      throw new Error('Invalid training response.');
    }
    setIncident(v); setMessage('');
  });
  const finish = () => execute(async () => {
    if (!incident) return;
    setIncident(await request(`/incidents/${encodeURIComponent(incident.id)}/close`, 'POST'));
  });

  if (!open || !enabled()) return null;
  return (
    <div className="fixed inset-0 z-[100001] overflow-y-auto bg-black/90 p-3 sm:p-6" role="presentation" data-verifyd-emergency-lab="TRAINING_ONLY">
      <section role="dialog" aria-modal="true" aria-labelledby="verifyd-emergency-lab-title"
        className="mx-auto max-w-2xl rounded-3xl border-2 border-amber-400 bg-[#111824] p-4 text-white shadow-2xl sm:p-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 id="verifyd-emergency-lab-title" className="flex items-center gap-2 text-xl font-extrabold text-amber-200">
              <MessageSquareText aria-hidden="true" /> Accessible emergency messages — TRAINING LAB
            </h2>
            <p className="mt-2 text-sm text-amber-100">Practice adaptive phrases, Eye Gaze and two-way messages with an automated simulator.</p>
          </div>
          <button type="button" ref={closeRef} onClick={onClose} aria-label="Close practice messaging"
            className="min-h-12 min-w-12 rounded-xl border border-white/30 p-3 focus-visible:outline-4 focus-visible:outline-amber-300"><X aria-hidden="true" /></button>
        </div>
        <div className="mt-4 rounded-xl border-2 border-red-400 bg-red-950 p-4 text-lg font-black" role="alert">
          <AlertTriangle className="mr-2 inline-block h-6 w-6" aria-hidden="true" />
          NOT 911. NO EMERGENCY SERVICES ARE CONTACTED. DO NOT USE FOR REAL EMERGENCIES.
        </div>
        <p className="mt-3 text-sm text-slate-200">This session contains simulated messages only. It will not notify dispatchers, trusted contacts, emergency providers or your Circle Distress contacts. Do not enter real names, addresses or telephone numbers.</p>
        <p className="mt-2 text-sm font-bold text-amber-100">In a real emergency, call 911 using an available phone; use native Text-to-911 where supported if you cannot call.</p>
        <div className="mt-4" aria-live="polite">
          {error && <p className="rounded-lg border border-red-300 bg-red-900/30 p-3" role="alert">{error}</p>}
          {!connected && <p className="text-amber-100">Practice backend not connected. No messages can be sent.</p>}
        </div>
        {connected && !incident && <div className="mt-5 space-y-3">
          <label htmlFor="emergency-lab-scenario" className="block text-base font-semibold">Choose a fictitious situation</label>
          <select id="emergency-lab-scenario" value={scenario} onChange={(e) => setScenario(e.target.value)}
            className="min-h-14 w-full rounded-xl border border-slate-400 bg-[#202b3a] p-3 text-white">
            {scenarios.map(([value, name]) => <option key={value} value={value}>{name}</option>)}
          </select>
          <button type="button" onClick={start} disabled={busy}
            className="min-h-14 w-full rounded-xl border border-amber-200 bg-amber-500 px-4 py-3 font-bold text-black disabled:opacity-40">
            Start a practice conversation — no dispatch
          </button>
        </div>}
        {connected && incident && <div className="mt-5 space-y-3">
          <p className="text-sm font-semibold text-amber-100">SIMULATION: {incident.status.replaceAll('_', ' ')} — no real emergency response</p>
          <div role="log" aria-label="Practice text conversation" aria-live="polite" aria-relevant="additions"
            className="max-h-64 space-y-3 overflow-y-auto rounded-xl border border-white/20 bg-black/30 p-3">
            {incident.messages?.map((entry, index) => <div key={`${index}-${entry.sender}`} className="rounded-lg bg-white/10 p-3">
              <strong className="block text-xs uppercase tracking-wide text-amber-100">{entry.sender === 'simulated_responder' ? 'SIMULATED REPLY — NOT 911' : entry.sender === 'training_system' ? 'Training system' : 'Your practice text'}</strong>
              <p className="mt-1 whitespace-pre-wrap break-words text-sm">{entry.message}</p>
            </div>)}
          </div>
          {incident.status !== 'simulation_closed' && <>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {phrases.map((phrase) => <button type="button" key={phrase} onClick={() => setMessage(phrase)}
                className="min-h-14 rounded-xl border border-sky-300/60 bg-sky-900/30 p-3 text-left text-sm font-semibold focus-visible:outline-4 focus-visible:outline-amber-300">{phrase}</button>)}
            </div>
            <label htmlFor="emergency-lab-message" className="block font-semibold">Practice message (fictional details only)</label>
            <textarea id="emergency-lab-message" maxLength={400} value={message} rows={3} onChange={(e) => setMessage(e.target.value)}
              className="min-h-28 w-full rounded-xl border border-slate-400 bg-[#202b3a] p-3 text-white" />
            <button type="button" onClick={send} disabled={busy || !message.trim()}
              className="min-h-14 w-full rounded-xl bg-sky-500 px-4 py-3 font-extrabold text-black disabled:opacity-40">Send to simulator ONLY</button>
            <button type="button" onClick={finish} disabled={busy}
              className="min-h-14 w-full rounded-xl border border-white/40 p-3 font-semibold disabled:opacity-40">End practice conversation</button>
          </>}
          {incident.status === 'simulation_closed' && <p className="font-bold text-amber-100">Practice finished. No emergency message was delivered.</p>}
        </div>}
        <button type="button" onClick={onClose} className="mt-6 min-h-12 w-full rounded-xl border border-white/40 px-4 py-3 font-semibold">Close training lab</button>
      </section>
    </div>
  );
}