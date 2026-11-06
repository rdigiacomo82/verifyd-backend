"""Ephemeral Linux runner test: WebRTC RTP -> LiveKit -> SIP -> local Asterisk Echo -> LiveKit.

NO emergency / PSTN / telephone dialing. The only permitted target is loopback extension 6000.
A success result requires an authenticated SIP participant AND measurable returned audio.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
import sys

from livekit import rtc

# Runs from package root. Do not import the production VeriFYD backend.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import phase4_bridge as lab

SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_SAMPLES = 960  # 20 milliseconds
SIGNAL_HZ = 523.25
SIGNAL_SECONDS = 5.0
MIN_RETURNED_PEAK = 500
EXPECTED_PEER = 'local-sip-echo'


def pcm_frame(index: int) -> rtc.AudioFrame:
    frame = rtc.AudioFrame.create(SAMPLE_RATE, CHANNELS, FRAME_SAMPLES)
    data = frame.data
    for i in range(FRAME_SAMPLES):
        # Sine wave, 16-bit PCM. Contains no speech, personal details or call information.
        data[i] = int(9000 * math.sin(2 * math.pi * SIGNAL_HZ * (index * FRAME_SAMPLES + i) / SAMPLE_RATE))
    return frame


async def run_echo() -> dict:
    lab._only_loopback()
    room = rtc.Room()
    subscribed = asyncio.Event()
    received = {"frames": 0, "peak": 0, "non_silent": 0}
    reader_tasks = []

    async def read_audio(track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
        try:
            async for event in stream:
                samples = event.frame.data
                peak = max((abs(int(sample)) for sample in samples), default=0)
                received['frames'] += 1
                received['peak'] = max(received['peak'], peak)
                if peak >= MIN_RETURNED_PEAK:
                    received['non_silent'] += 1
        finally:
            await stream.aclose()

    @room.on('track_subscribed')
    def on_track_subscribed(track, publication, participant) -> None:
        if participant.identity != EXPECTED_PEER or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        reader_tasks.append(asyncio.create_task(read_audio(track)))
        subscribed.set()

    try:
        token = lab.generate_join_token()
        await asyncio.wait_for(room.connect('ws://127.0.0.1:7880', token), timeout=15)
        source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
        audio_track = rtc.LocalAudioTrack.create_audio_track('private-echo-test-audio', source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await asyncio.wait_for(room.local_participant.publish_track(audio_track, options), timeout=15)

        # Initiate only the fixed, pre-approved internal SIP extension.
        # In parallel with the WebRTC connection, not on the async event loop.
        call = await asyncio.wait_for(asyncio.to_thread(lab.call_echo, confirmed=True, timeout=40), timeout=55)
        if not call.get('ok') or call.get('real_911_call_placed') or call.get('real_933_call_placed'):
            raise RuntimeError('SIP call result violated lab invariants')
        await asyncio.wait_for(subscribed.wait(), timeout=18)

        for n in range(int(SIGNAL_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)):
            await source.capture_frame(pcm_frame(n))
        await source.wait_for_playout()
        await asyncio.sleep(1.5)

        if received['non_silent'] < 5 or received['peak'] < MIN_RETURNED_PEAK:
            raise RuntimeError(f'No verified audio echoed from internal SIP peer: {received}')
        return {
            'ok': True,
            'mode': 'CI_WEBRTC_LIVEKIT_SIP_ASTERISK_ECHO',
            'peer': EXPECTED_PEER,
            'extension': lab.ONLY_EXTENSION,
            'audio_track_subscribed': True,
            'returned_audio': received,
            'real_933_call_placed': False,
            'real_911_call_placed': False,
            'carrier_connected': False,
            'personal_audio_recorded': False,
        }
    finally:
        for task in reader_tasks:
            task.cancel()
        if reader_tasks:
            await asyncio.gather(*reader_tasks, return_exceptions=True)
        await room.disconnect()


if __name__ == '__main__':
    try:
        result = asyncio.run(asyncio.wait_for(run_echo(), timeout=105))
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                          'real_911_call_placed': False, 'real_933_call_placed': False}, indent=2), file=sys.stderr)
        raise SystemExit(1)