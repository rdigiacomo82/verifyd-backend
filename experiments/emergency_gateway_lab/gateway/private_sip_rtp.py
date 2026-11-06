"""Local-only SIP INVITE/SDP + bidirectional G.711 μ-law RTP test.

This is a MOCK SIP User-Agent Server, NOT LiveKit, a PBX, a carrier or 911.
Both negotiated RTP addresses and the SIP socket are hard-bound to IPv4 loopback.
Audio is generated entirely in memory as synthetic tones. No microphone,
recordings, real people, outside network access or phone numbers are involved.
"""
from __future__ import annotations

import asyncio
import math
import secrets
import socket
import struct
import threading
from dataclasses import dataclass, field

LOOPBACK = '127.0.0.1'
PACKET_SAMPLES = 160
SAMPLE_RATE = 8000
PACKET_INTERVAL = 0.02
DEFAULT_PACKETS = 25
MAX_PACKETS = 40
RTP_PT_PCMU = 0
_LOCK = threading.Lock()


def pcm16_to_mulaw(sample: int) -> int:
    """G.711 PCMU codec, in pure Python (no removed Python 3.13 audioop)."""
    if not -32768 <= sample <= 32767:
        raise ValueError('invalid_pcm_sample')
    sign = 0x7f if sample < 0 else 0xff
    magnitude = min(32635, abs(sample)) + 0x84
    exponent = max(0, min(7, magnitude.bit_length() - 8))
    mantissa = (magnitude >> (exponent + 3)) & 0x0f
    return (sign ^ ((exponent << 4) | mantissa)) & 0xff


def mulaw_to_pcm16(encoded: int) -> int:
    """Decode standard G.711 μ-law value to signed 16-bit linear PCM."""
    value = (~encoded) & 0xff
    magnitude = (((value & 0x0f) << 3) + 0x84) << ((value >> 4) & 0x07)
    decoded = magnitude - 0x84
    return -decoded if value & 0x80 else decoded


def synthetic_tone(frequency_hz: int, packet_index: int) -> bytes:
    if frequency_hz not in (440, 660):
        raise ValueError('unapproved_synthetic_tone')
    start = packet_index * PACKET_SAMPLES
    return bytes(pcm16_to_mulaw(round(8000 * math.sin(2 * math.pi * frequency_hz *
                                                   (start + i) / SAMPLE_RATE)))
                 for i in range(PACKET_SAMPLES))


def make_rtp_packet(*, sequence: int, ssrc: int, tone_hz: int) -> bytes:
    if sequence < 0 or sequence >= MAX_PACKETS:
        raise ValueError('packet_out_of_bounds')
    return (struct.pack('!BBHII', 0x80, RTP_PT_PCMU, sequence, sequence * PACKET_SAMPLES, ssrc)
            + synthetic_tone(tone_hz, sequence))


def decode_rtp_packet(data: bytes):
    if len(data) != 12 + PACKET_SAMPLES:
        raise ValueError('invalid_rtp_frame')
    flags, payload_type, seq, ts, ssrc = struct.unpack('!BBHII', data[:12])
    if flags != 0x80 or payload_type != RTP_PT_PCMU or ts != seq * PACKET_SAMPLES:
        raise ValueError('invalid_rtp_header')
    decoded = [mulaw_to_pcm16(b) for b in data[12:]]
    energy = math.sqrt(sum(x * x for x in decoded) / len(decoded))
    return seq, ssrc, energy


@dataclass
class RtpSink(asyncio.DatagramProtocol):
    received: dict[int, float] = field(default_factory=dict)
    peer: tuple[str, int] | None = None
    rejected: int = 0

    def datagram_received(self, data, addr):
        if self.peer is None or addr != self.peer or addr[0] != LOOPBACK:
            self.rejected += 1
            return
        try:
            sequence, _, energy = decode_rtp_packet(data)
            if sequence not in self.received:
                self.received[sequence] = energy
        except (ValueError, struct.error):
            self.rejected += 1


class PrivateSIPUAS(asyncio.DatagramProtocol):
    """Single-dialog mock; only sip:lab-audio at 127.0.0.1 is allowed."""
    def __init__(self, *, media_port: int, expected_caller_media_port: int, call_id: str):
        self.media_port = media_port
        self.expected_caller_media_port = expected_caller_media_port
        self.call_id = call_id
        self.transport = None
        self.invited = False
        self.acked = False
        self.ended = False
        self.rejected = 0

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if (addr[0] != LOOPBACK or len(data) > 4096 or
                f'Call-ID: {self.call_id}\r\n'.encode('ascii') not in data):
            self.rejected += 1
            return
        text = data.decode('ascii', errors='replace')
        first = text.split('\r\n', 1)[0]
        if first.startswith('INVITE sip:lab-audio@127.0.0.1:') and first.endswith(' SIP/2.0'):
            if self.invited or f'm=audio {self.expected_caller_media_port} RTP/AVP 0\r\n' not in text:
                self.rejected += 1
                return
            self.invited = True
            sdp = (f'v=0\r\no=- 0 0 IN IP4 {LOOPBACK}\r\ns=VERI-FYD PRIVATE AUDIO TEST\r\n'
                   f'c=IN IP4 {LOOPBACK}\r\nt=0 0\r\nm=audio {self.media_port} RTP/AVP 0\r\n'
                   'a=rtpmap:0 PCMU/8000\r\na=sendrecv\r\n')
            reply = ('SIP/2.0 200 OK\r\nCSeq: 1 INVITE\r\n'
                     f'Call-ID: {self.call_id}\r\nContent-Type: application/sdp\r\n'
                     f'Content-Length: {len(sdp.encode())}\r\n\r\n{sdp}')
        elif first.startswith('ACK sip:lab-audio@127.0.0.1:') and self.invited:
            self.acked = True
            return
        elif first.startswith('BYE sip:lab-audio@127.0.0.1:') and self.acked:
            self.ended = True
            reply = f'SIP/2.0 200 OK\r\nCSeq: 2 BYE\r\nCall-ID: {self.call_id}\r\nContent-Length: 0\r\n\r\n'
        else:
            self.rejected += 1
            reply = 'SIP/2.0 403 Forbidden\r\nContent-Length: 0\r\n\r\n'
        self.transport.sendto(reply.encode('ascii'), addr)


async def _signaling_exchange(sip_port: int, caller_port: int, callee_port: int,
                              call_id: str, uas: PrivateSIPUAS, *, bye: bool = False):
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        uri = f'sip:lab-audio@{LOOPBACK}:{sip_port}'
        base = f'Call-ID: {call_id}\r\n'
        if not bye:
            offer = (f'v=0\r\no=- 0 0 IN IP4 {LOOPBACK}\r\ns=VERI-FYD PRIVATE AUDIO TEST\r\n'
                     f'c=IN IP4 {LOOPBACK}\r\nt=0 0\r\nm=audio {caller_port} RTP/AVP 0\r\n'
                     'a=rtpmap:0 PCMU/8000\r\na=sendrecv\r\n')
            invite = (f'INVITE {uri} SIP/2.0\r\n{base}CSeq: 1 INVITE\r\n'
                      f'Content-Type: application/sdp\r\nContent-Length: {len(offer)}\r\n\r\n{offer}')
            await loop.sock_sendto(sock, invite.encode('ascii'), (LOOPBACK, sip_port))
            response, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 4096), timeout=1.0)
            if addr != (LOOPBACK, sip_port) or not response.startswith(b'SIP/2.0 200 OK\r\n'):
                raise RuntimeError('local_invite_failed')
            if (f'm=audio {callee_port} RTP/AVP 0\r\n'.encode() not in response or
                    b'a=rtpmap:0 PCMU/8000\r\n' not in response):
                raise RuntimeError('wrong_private_sdp_answer')
            await loop.sock_sendto(sock, (f'ACK {uri} SIP/2.0\r\n{base}CSeq: 1 ACK\r\n'
                                          'Content-Length: 0\r\n\r\n').encode(), (LOOPBACK, sip_port))
            await asyncio.sleep(0.005)
            if not uas.acked:
                raise RuntimeError('local_ack_failed')
        else:
            await loop.sock_sendto(sock, (f'BYE {uri} SIP/2.0\r\n{base}CSeq: 2 BYE\r\n'
                                          'Content-Length: 0\r\n\r\n').encode(), (LOOPBACK, sip_port))
            response, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=1.0)
            if addr != (LOOPBACK, sip_port) or not response.startswith(b'SIP/2.0 200 OK\r\n'):
                raise RuntimeError('local_bye_failed')
    finally:
        sock.close()


async def run_private_audio_test(packets: int = DEFAULT_PACKETS) -> dict:
    """Exercise SIP INVITE, SDP, ACK, PCMU RTP both ways and BYE on loopback.

    Never accepts an address, phone number, trunk, transport or microphone input.
    Returned audio statistics are synthetic; NO LiveKit connection is tested.
    """
    if not 4 <= packets <= MAX_PACKETS:
        raise ValueError('invalid_packet_count')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('lab_audio_test_busy')
    transports = []
    try:
        loop = asyncio.get_running_loop()
        a_transport, a_sink = await loop.create_datagram_endpoint(RtpSink, local_addr=(LOOPBACK, 0))
        transports.append(a_transport)
        b_transport, b_sink = await loop.create_datagram_endpoint(RtpSink, local_addr=(LOOPBACK, 0))
        transports.append(b_transport)
        a_addr = a_transport.get_extra_info('sockname')
        b_addr = b_transport.get_extra_info('sockname')
        if a_addr[0] != LOOPBACK or b_addr[0] != LOOPBACK:
            raise RuntimeError('non_loopback_media_address')
        a_sink.peer = b_addr
        b_sink.peer = a_addr
        call_id = secrets.token_hex(12) + '@lab.local'
        uas = PrivateSIPUAS(media_port=b_addr[1], expected_caller_media_port=a_addr[1], call_id=call_id)
        sip_transport, _ = await loop.create_datagram_endpoint(lambda: uas, local_addr=(LOOPBACK, 0))
        transports.append(sip_transport)
        sip_addr = sip_transport.get_extra_info('sockname')
        await _signaling_exchange(sip_addr[1], a_addr[1], b_addr[1], call_id, uas)
        a_ssrc = secrets.randbits(32)
        b_ssrc = secrets.randbits(32)
        for seq in range(packets):
            a_transport.sendto(make_rtp_packet(sequence=seq, ssrc=a_ssrc, tone_hz=440), b_addr)
            b_transport.sendto(make_rtp_packet(sequence=seq, ssrc=b_ssrc, tone_hz=660), a_addr)
            await asyncio.sleep(PACKET_INTERVAL)
        await asyncio.sleep(0.025)
        await _signaling_exchange(sip_addr[1], a_addr[1], b_addr[1], call_id, uas, bye=True)
        def stats(sink: RtpSink) -> dict:
            count = len(sink.received)
            return {'packets_sent': packets, 'packets_received': count,
                    'packet_loss': packets - count, 'rms_audio_energy': round(sum(sink.received.values()) / count, 1) if count else 0,
                    'invalid_packets': sink.rejected}
        left, right = stats(a_sink), stats(b_sink)
        passed = bool(uas.invited and uas.acked and uas.ended and
                      left['packets_received'] == packets and right['packets_received'] == packets and
                      left['rms_audio_energy'] > 100 and right['rms_audio_energy'] > 100 and
                      not left['invalid_packets'] and not right['invalid_packets'])
        return {'ok': passed, 'mode': 'PRIVATE_LOCAL_SIP_RTP_SYNTHETIC_AUDIO',
                'sip_dialog': 'invite_200_ack_bye_completed' if uas.ended else 'incomplete',
                'sdp_address': LOOPBACK, 'codec': 'PCMU/8000', 'packet_duration_ms': 20,
                'client_receives_lab_tone': left, 'lab_receives_client_tone': right,
                'media_transport_connected': passed, 'media_is_synthetic_tones': True,
                'livekit_connected': False, 'sip_trunk_connected': False,
                'carrier_connected': False, 'real_933_call_placed': False,
                'real_911_call_placed': False, 'real_emergency_service_validated': False,
                'recorded_audio': False, 'not_emergency_service': True}
    finally:
        for transport in transports:
            transport.close()
        _LOCK.release()