import { Room, RoomEvent } from 'livekit-client';
// Lab-specific browser UI. NO API secret or SIP admin token is ever sent here.
const status = document.querySelector('#status');
const join = document.querySelector('#join');
const leave = document.querySelector('#leave');
const token = document.querySelector('#token');
const peers = document.querySelector('#peerCount');
const audioWrap = document.querySelector('#remoteAudio');
let room = null;
const playerElements = new Map();
const report = (s) => { status.textContent = s; };
const update = () => { peers.textContent = `Remote audio tracks: ${playerElements.size}`; };
const destroy = () => {
  for (const el of playerElements.values()) el.remove();
  playerElements.clear(); update();
};
join.addEventListener('click', async () => {
  if (room) return;
  const jwt = token.value.trim();
  if (!jwt) { report('Paste a short-lived room token first.'); return; }
  join.disabled = true;
  try {
    room = new Room({ adaptiveStream: true, dynacast: true });
    room.on(RoomEvent.TrackSubscribed, (track, publication) => {
      if (track.kind !== 'audio') return;
      const el = track.attach();
      el.autoplay = true;
      el.controls = true;
      el.setAttribute('aria-label', 'Audio from lab echo SIP participant');
      audioWrap.appendChild(el);
      playerElements.set(publication.trackSid, el); update();
    });
    room.on(RoomEvent.TrackUnsubscribed, (track, publication) => {
      track.detach().forEach(el => el.remove());
      playerElements.delete(publication.trackSid); update();
    });
    room.on(RoomEvent.Disconnected, () => { destroy(); leave.disabled = true; join.disabled = false; room = null; report('Disconnected'); });
    await room.connect('ws://127.0.0.1:7880', jwt);
    await room.localParticipant.setMicrophoneEnabled(true);
    token.value = ''; // don't retain the token on screen
    leave.disabled = false;
    report('Connected. Run `python bridge/phase4_bridge.py call-echo --confirm-private-echo` in the Linux lab terminal, then speak to hear your echo. Use headphones to avoid feedback.');
  } catch (err) {
    report(`Room/microphone error: ${String(err?.message || err)}`);
    room?.disconnect(); room = null; destroy(); join.disabled = false;
  }
});
leave.addEventListener('click', () => {
  room?.disconnect(); room = null; leave.disabled = true; join.disabled = false; destroy(); report('Disconnected');
});