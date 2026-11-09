"""SIP OPTIONS-only loopback probe. No INVITE, RTP, DTMF or emergency dialling.

Only the literal IPv4 loopback address is allowed; no custom destination host,
port, route, PSTN number, or SIP dialstring is accepted from clients.
"""
from __future__ import annotations

import asyncio
import secrets
import socket
from contextlib import suppress

SIP_HOST = '127.0.0.1'
SIP_PORT = 5099
SIP_TEST_URI = f'sip:lab@{SIP_HOST}:{SIP_PORT}'


def _options_request() -> bytes:
    token = secrets.token_hex(12)
    return ('\r\n'.join([
        f'OPTIONS {SIP_TEST_URI} SIP/2.0',
        f'Via: SIP/2.0/UDP 127.0.0.1:5068;branch=z9hG4bK{token}',
        'Max-Forwards: 1',
        'From: <sip:test@127.0.0.1>;tag=lab',
        'To: <sip:lab@127.0.0.1>',
        f'Call-ID: {token}@127.0.0.1',
        'CSeq: 1 OPTIONS',
        'Content-Length: 0',
        '', '',
    ])).encode('ascii')


async def run_options_probe(timeout: float = 0.75) -> bool:
    """Acknowledged SIP signaling does NOT mean an audio or emergency call."""
    if not 0.05 <= timeout <= 3:
        raise ValueError('Timeout out of lab bounds')
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        await loop.sock_sendto(sock, _options_request(), (SIP_HOST, SIP_PORT))
        data, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 4096), timeout)
        return bool(addr == (SIP_HOST, SIP_PORT) and data.startswith(b'SIP/2.0 200 OK\r\n'))
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        sock.close()


class MockSIPProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        # The only allowed SIP method is OPTIONS; all other methods rejected.
        if data.startswith(b'OPTIONS sip:lab@127.0.0.1:5099 SIP/2.0\r\n') and b'CSeq: 1 OPTIONS\r\n' in data:
            lines = data.decode('ascii', errors='replace').split('\r\n')
            keep = [line for line in lines if line.startswith(('Via:', 'From:', 'To:', 'Call-ID:', 'CSeq:'))]
            response = '\r\n'.join(['SIP/2.0 200 OK', *keep, 'Content-Length: 0', '', '']).encode('ascii')
        else:
            response = b'SIP/2.0 405 Method Not Allowed\r\nAllow: OPTIONS\r\nContent-Length: 0\r\n\r\n'
        if self.transport is not None and addr[0] == SIP_HOST:
            self.transport.sendto(response, addr)


async def run_mock_server(ready: asyncio.Event | None = None):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(MockSIPProtocol, local_addr=(SIP_HOST, SIP_PORT))
    if ready is not None:
        ready.set()
    try:
        await asyncio.Event().wait()
    finally:
        transport.close()


if __name__ == '__main__':
    asyncio.run(run_mock_server())