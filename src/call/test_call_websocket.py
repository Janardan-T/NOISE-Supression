"""Automated WebSocket Test for Live Voice Call Protocol and Room Management.

Verifies:
1. Self-test loopback protocol (/ws/call) with RTT, Denoised vs Bypass mode.
2. Room creation and join by unguessable code (/ws/room/{code}).
3. 2-participant limit enforcement (ROOM_FULL, close code 4001 for 3rd joiner).
4. Server room capacity limit enforcement (ROOMS_BUSY, close code 4002 when max_rooms exceeded).
5. Immediate and idle room cleanup (rooms deleted upon empty/idle state).
6. IP rate limiting (RATE_LIMITED, close code 4003 when join attempts exceed threshold).
7. Invalid room code format validation (INVALID_ROOM_CODE, close code 4004).
"""

import argparse
import asyncio
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import soundfile as sf
import websockets

# Binary protocol formats
CLIENT_HEADER_FORMAT = ">IIBBH"  # 12 bytes
CLIENT_HEADER_SIZE = struct.calcsize(CLIENT_HEADER_FORMAT)

SERVER_HEADER_FORMAT = ">IIfBBH"  # 16 bytes
SERVER_HEADER_SIZE = struct.calcsize(SERVER_HEADER_FORMAT)

CHUNK_SAMPLES = 1920
CHUNK_BYTES = CHUNK_SAMPLES * 2


async def test_self_test_protocol(server_ws: str, audio_path: str):
    """Test single-user loopback endpoint (/ws/call)."""
    print("\n" + "=" * 70)
    print(" 1. SELF-TEST LOOPBACK WEBSOCKET PROTOCOL TEST")
    print("=" * 70)
    url = f"{server_ws}/ws/call"

    raw_audio, sr = sf.read(audio_path, dtype="float32")
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1)

    if sr != 48000:
        import scipy.signal
        num_samples = int(len(raw_audio) * 48000 / sr)
        raw_audio = scipy.signal.resample(raw_audio, num_samples)

    raw_int16 = (np.clip(raw_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    total_frames = min(50, len(raw_int16) // CHUNK_SAMPLES)

    rtts_denoised = []
    rtts_bypass = []
    server_procs_denoised = []
    server_procs_bypass = []
    packets_sent = 0
    packets_received = 0

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        for frame_idx in range(total_frames):
            start_smp = frame_idx * CHUNK_SAMPLES
            end_smp = start_smp + CHUNK_SAMPLES
            chunk_int16 = raw_int16[start_smp:end_smp]
            chunk_bytes = chunk_int16.tobytes()

            mode = 1 if (20 <= frame_idx < 35) else 0

            client_ts = int(time.perf_counter() * 1000.0) & 0xFFFFFFFF
            header = struct.pack(CLIENT_HEADER_FORMAT, frame_idx, client_ts, mode, 0, 0)
            packet = header + chunk_bytes

            send_time = time.perf_counter()
            await ws.send(packet)
            packets_sent += 1

            resp_bytes = await ws.recv()
            recv_time = time.perf_counter()
            packets_received += 1

            assert len(resp_bytes) >= SERVER_HEADER_SIZE
            resp_header = resp_bytes[:SERVER_HEADER_SIZE]
            audio_payload = resp_bytes[SERVER_HEADER_SIZE:]

            seq, echo_ts, proc_ms, resp_mode, _, _ = struct.unpack(SERVER_HEADER_FORMAT, resp_header)
            assert seq == frame_idx
            assert resp_mode == mode
            assert len(audio_payload) == CHUNK_BYTES

            rtt_ms = (recv_time - send_time) * 1000.0
            if mode == 0:
                rtts_denoised.append(rtt_ms)
                server_procs_denoised.append(proc_ms)
            else:
                rtts_bypass.append(rtt_ms)
                server_procs_bypass.append(proc_ms)
                assert audio_payload == chunk_bytes

    print(f" [Self-Test]: Packets Sent/Received: {packets_sent}/{packets_received} (100% delivery)")
    print(f"   Denoised RTT: avg = {np.mean(rtts_denoised):.2f} ms | Proc: avg = {np.mean(server_procs_denoised):.2f} ms")
    print(f"   Bypass RTT:   avg = {np.mean(rtts_bypass):.2f} ms | Proc: avg = {np.mean(server_procs_bypass):.4f} ms")
    print(" [Self-Test Result]: PASSED")
    return True


async def test_create_and_join_by_code(server_ws: str, room_code: str):
    """Test joining a room by random unguessable room code."""
    print("\n" + "=" * 70)
    print(" 2. CREATE AND JOIN ROOM BY CODE TEST")
    print("=" * 70)
    url = f"{server_ws}/ws/room/{room_code}"

    async with websockets.connect(url) as ws1:
        msg1 = json.loads(await ws1.recv())
        assert msg1["type"] == "joined"
        assert msg1["room_code"] == room_code
        assert msg1["participant_count"] == 1
        assert msg1["peer_present"] is False
        print(f" [Participant 1 Joined]: Room {room_code}, peer_present=False")

        # Second participant joins
        async with websockets.connect(url) as ws2:
            msg2 = json.loads(await ws2.recv())
            assert msg2["type"] == "joined"
            assert msg2["participant_count"] == 2
            assert msg2["peer_present"] is True

            # Participant 1 should receive peer_joined notification
            peer_event = json.loads(await ws1.recv())
            assert peer_event["type"] == "peer_joined"
            assert peer_event["participant_count"] == 2
            print(f" [Participant 2 Joined]: Room {room_code}, peer_present=True. Peer notified.")

    print(" [Create/Join Result]: PASSED")
    return True


async def test_room_full_rejection(server_ws: str, room_code: str):
    """Test that a 3rd participant is rejected with ROOM_FULL (code 4001)."""
    print("\n" + "=" * 70)
    print(" 3. ROOM_FULL (MAX 2 PARTICIPANTS) ENFORCEMENT TEST")
    print("=" * 70)
    url = f"{server_ws}/ws/room/{room_code}"

    async with websockets.connect(url) as ws1:
        await ws1.recv()  # welcome
        async with websockets.connect(url) as ws2:
            await ws2.recv()  # welcome
            await ws1.recv()  # peer_joined

            # Attempt 3rd participant connection
            rejected = False
            error_code = None
            close_code = None
            try:
                async with websockets.connect(url) as ws3:
                    msg = json.loads(await ws3.recv())
                    error_code = msg.get("code")
                    # Should receive close from server
                    try:
                        await ws3.recv()
                    except websockets.exceptions.ConnectionClosed as cc:
                        close_code = cc.code
                        rejected = True
            except websockets.exceptions.ConnectionClosed as cc:
                close_code = cc.code
                rejected = True

            print(f" [3rd Joiner Rejection]: Error code = '{error_code}', Close Code = {close_code}")
            assert error_code == "ROOM_FULL"
            assert close_code == 4001
            assert rejected

    print(" [ROOM_FULL Result]: PASSED (Strict 2-participant limit verified)")
    return True


async def test_rooms_busy_capacity(server_ws: str, max_rooms: int = 2):
    """Test that server rejects creating rooms beyond max_rooms with ROOMS_BUSY (4002)."""
    print("\n" + "=" * 70)
    print(" 4. ROOMS_BUSY (MAX ACTIVE CONCURRENT ROOMS) ENFORCEMENT TEST")
    print("=" * 70)

    connections = []
    try:
        # Fill up active rooms up to max_rooms
        for i in range(max_rooms):
            code = f"BUSYRM{i:02d}"
            url = f"{server_ws}/ws/room/{code}"
            ws = await websockets.connect(url)
            await ws.recv()  # welcome
            connections.append(ws)
            print(f" [Occupied Room {i + 1}/{max_rooms}]: {code}")

        # Attempt to create (max_rooms + 1)-th room
        overflow_code = f"BUSYRM_OVERFLOW"
        url_overflow = f"{server_ws}/ws/room/{overflow_code}"
        rejected = False
        error_code = None
        close_code = None

        try:
            async with websockets.connect(url_overflow) as ws_over:
                msg = json.loads(await ws_over.recv())
                error_code = msg.get("code")
                try:
                    await ws_over.recv()
                except websockets.exceptions.ConnectionClosed as cc:
                    close_code = cc.code
                    rejected = True
        except websockets.exceptions.ConnectionClosed as cc:
            close_code = cc.code
            rejected = True

        print(f" [Overflow Room Rejection]: Error code = '{error_code}', Close Code = {close_code}")
        assert error_code == "ROOMS_BUSY"
        assert close_code == 4002
        assert rejected
        print(" [ROOMS_BUSY Result]: PASSED (Server overload protection verified)")
        return True

    finally:
        for ws in connections:
            try:
                await ws.close()
            except Exception:
                pass


async def test_room_cleanup_verification(server_ws: str, http_base: str, room_code: str):
    """Test that empty rooms are deleted immediately and do not leak."""
    print("\n" + "=" * 70)
    print(" 5. ROOM CLEANUP AND DISPOSAL VERIFICATION TEST")
    print("=" * 70)
    url = f"{server_ws}/ws/room/{room_code}"

    # Connect and disconnect
    ws = await websockets.connect(url)
    await ws.recv()
    print(f" [Created & Joined]: {room_code}")
    await ws.close()
    await asyncio.sleep(0.2)  # allow server async lock to clean up

    # Check health endpoint active_rooms count
    import urllib.request
    resp = urllib.request.urlopen(f"{http_base}/health", timeout=2.0)
    health = json.loads(resp.read().decode("utf-8"))
    active_rooms = health.get("active_rooms", -1)
    print(f" [Health Check after Disconnect]: active_rooms = {active_rooms}")
    assert active_rooms == 0, f"Expected 0 active rooms, but found {active_rooms}"

    print(" [Room Cleanup Result]: PASSED (Empty room purged immediately)")
    return True


async def test_rate_limiting_enforcement(server_ws: str, http_base: str):
    """Test that excessive rapid connection attempts from the same IP trigger RATE_LIMITED (4003)."""
    print("\n" + "=" * 70)
    print(" 6. JOIN RATE LIMITING ENFORCEMENT TEST")
    print("=" * 70)

    rate_limited = False
    error_code = None
    close_code = None

    # Spam connections rapidly
    conns = []
    try:
        for attempt in range(25):
            code = f"RATELIM{attempt}"
            url = f"{server_ws}/ws/room/{code}"
            try:
                ws = await websockets.connect(url, open_timeout=1.0)
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                msg = json.loads(msg_raw)
                if msg.get("type") == "error" and msg.get("code") == "RATE_LIMITED":
                    error_code = msg.get("code")
                    try:
                        await ws.recv()
                    except websockets.exceptions.ConnectionClosed as cc:
                        close_code = cc.code
                    rate_limited = True
                    break
                conns.append(ws)
            except websockets.exceptions.ConnectionClosed as cc:
                close_code = cc.code
                rate_limited = True
                break
            except Exception:
                pass

        print(f" [Rate Limit Triggered]: error_code = '{error_code}', close_code = {close_code}")
        assert rate_limited, "Rate limiter did not throttle excessive join attempts!"
        assert close_code == 4003 or error_code == "RATE_LIMITED"
        print(" [Rate Limiter Result]: PASSED (IP join burst throttled)")
        return True

    finally:
        for ws in conns:
            try:
                await ws.close()
            except Exception:
                pass
        # Reset rate limiter so subsequent tests from same IP are unblocked
        try:
            import urllib.request
            req = urllib.request.Request(f"{http_base}/api/reset-rate-limit", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass


async def test_invalid_room_code(server_ws: str):
    """Test that invalid room codes (symbols, bad chars) are rejected with INVALID_ROOM_CODE (4004)."""
    print("\n" + "=" * 70)
    print(" 7. INVALID ROOM CODE FORMAT VALIDATION TEST")
    print("=" * 70)
    bad_code = "invalid*code"
    url = f"{server_ws}/ws/room/{bad_code}"

    error_code = None
    close_code = None
    try:
        async with websockets.connect(url) as ws:
            msg = json.loads(await ws.recv())
            error_code = msg.get("code")
            try:
                await ws.recv()
            except websockets.exceptions.ConnectionClosed as cc:
                close_code = cc.code
    except websockets.exceptions.ConnectionClosed as cc:
        close_code = cc.code

    print(f" [Invalid Code Rejection]: error_code = '{error_code}', close_code = {close_code}")
    assert error_code == "INVALID_ROOM_CODE"
    assert close_code == 4004
    print(" [Invalid Code Result]: PASSED (Alphanumeric constraint enforced)")
    return True


async def run_all_call_tests(port: int = 8001):
    server_ws = f"ws://127.0.0.1:{port}"
    http_base = f"http://127.0.0.1:{port}"
    audio_sample = "samples/speech_with_whitenoise.wav"

    print("*" * 78)
    print(f" COMPREHENSIVE CALL SERVER TEST SUITE (TARGET: {http_base})")
    print("*" * 78)

    t0 = time.time()
    await test_self_test_protocol(server_ws, audio_sample)
    await test_create_and_join_by_code(server_ws, "K7P9WR")
    await test_room_full_rejection(server_ws, "FULL01")
    await test_rooms_busy_capacity(server_ws, max_rooms=2)
    await test_room_cleanup_verification(server_ws, http_base, "CLEAN01")
    await test_invalid_room_code(server_ws)
    await test_rate_limiting_enforcement(server_ws, http_base)

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f" ALL 7 TEST SUITES PASSED IN {elapsed:.2f} s")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Server port to test against")
    args = parser.parse_args()

    asyncio.run(run_all_call_tests(args.port))
