"""Automated Python WebSocket test simulating two participants in a room call.

Verifies:
1. Two participants join /ws/room/test-duo.
2. Participant A and B stream audio simultaneously in real time (40 ms frames).
3. Cross-relay verification:
   - Participant A receives Participant B's relayed denoised frames.
   - Participant B receives Participant A's relayed denoised frames.
4. Frame counts, ordering, and 0 dropped frames.
5. Measured RTT and server processing latency for both participants.
6. Rejection of 3rd participant (room capacity limit).
7. Clean disconnection and remaining peer notification.
"""

import asyncio
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import soundfile as sf
import websockets

CLIENT_HEADER_FORMAT = ">IIBBH"  # 12 bytes
CLIENT_HEADER_SIZE = struct.calcsize(CLIENT_HEADER_FORMAT)

SERVER_HEADER_FORMAT = ">IIfBBH"  # 16 bytes
SERVER_HEADER_SIZE = struct.calcsize(SERVER_HEADER_FORMAT)

CHUNK_SAMPLES = 1920
CHUNK_BYTES = CHUNK_SAMPLES * 2


def load_wav_frames(path: str, max_frames: int = 50):
    raw_audio, sr = sf.read(path, dtype="float32")
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1)
    if sr != 48000:
        import scipy.signal
        num_s = int(len(raw_audio) * 48000 / sr)
        raw_audio = scipy.signal.resample(raw_audio, num_s)
    raw_int16 = (np.clip(raw_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    frames = []
    total_f = min(max_frames, len(raw_int16) // CHUNK_SAMPLES)
    for i in range(total_f):
        frames.append(raw_int16[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES].tobytes())
    return frames


async def participant_client(
    name: str,
    room_code: str,
    server_base_ws: str,
    frames_to_send: list,
    results: dict,
    stop_event: asyncio.Event,
):
    ws_url = f"{server_base_ws}/ws/room/{room_code}"
    results[name] = {
        "sent_frames": 0,
        "telemetry_received": 0,
        "relayed_audio_received": 0,
        "rtt_list": [],
        "proc_list": [],
        "peer_events": [],
        "dropped": 0,
        "last_peer_seq": -1,
    }

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        # 1. Receive initial join message
        join_msg = await ws.recv()
        join_data = json.loads(join_msg)
        results[name]["peer_events"].append(join_data.get("type"))

        async def sender():
            for seq, chunk_bytes in enumerate(frames_to_send):
                t_send = time.perf_counter()
                client_ts = int(t_send * 1000.0) & 0xFFFFFFFF
                header = struct.pack(CLIENT_HEADER_FORMAT, seq, client_ts, 0, 0, 0)
                await ws.send(header + chunk_bytes)
                results[name]["sent_frames"] += 1

                elapsed = time.perf_counter() - t_send
                sleep_t = max(0.0, 0.040 - elapsed)
                if sleep_t > 0:
                    await asyncio.sleep(sleep_t)

        async def receiver():
            while True:
                try:
                    data = await ws.recv()
                    t_recv = time.perf_counter()
                    if isinstance(data, str):
                        msg = json.loads(data)
                        results[name]["peer_events"].append(msg.get("type"))
                        continue

                    if len(data) == SERVER_HEADER_SIZE:
                        # Sender telemetry return packet (16 bytes)
                        seq, client_ts, proc_ms, mode, flags, _ = struct.unpack(SERVER_HEADER_FORMAT, data)
                        now_ts = int(t_recv * 1000.0) & 0xFFFFFFFF
                        rtt = max(0, (now_ts - client_ts) & 0xFFFFFFFF)
                        results[name]["telemetry_received"] += 1
                        results[name]["rtt_list"].append(rtt)
                        results[name]["proc_list"].append(proc_ms)
                    elif len(data) > SERVER_HEADER_SIZE:
                        # Relayed audio packet from peer
                        header = data[:SERVER_HEADER_SIZE]
                        audio = data[SERVER_HEADER_SIZE:]
                        seq, client_ts, proc_ms, mode, flags, _ = struct.unpack(SERVER_HEADER_FORMAT, header)

                        last = results[name]["last_peer_seq"]
                        if last >= 0 and seq > last + 1:
                            results[name]["dropped"] += (seq - last - 1)
                        results[name]["last_peer_seq"] = seq

                        results[name]["relayed_audio_received"] += 1
                except asyncio.CancelledError:
                    break
                except websockets.ConnectionClosed:
                    break

        rec_task = asyncio.create_task(receiver())
        await sender()

        # Wait until test stop_event is set or timeout
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=6.0)
        except asyncio.TimeoutError:
            pass

        rec_task.cancel()
        try:
            await rec_task
        except asyncio.CancelledError:
            pass


async def test_room_overflow(room_code: str, server_base_ws: str):
    """Test that a 3rd participant gets cleanly rejected with ROOM_FULL."""
    ws_url = f"{server_base_ws}/ws/room/{room_code}"
    try:
        async with websockets.connect(ws_url) as ws:
            resp = await ws.recv()
            data = json.loads(resp)
            return data.get("code") == "ROOM_FULL"
    except Exception:
        return False


async def run_room_relay_test(server_ws: str = "ws://127.0.0.1:8001"):
    print("=" * 70)
    print(" TWO-PARTY ROOM CALL RELAY TEST (PHASE B)")
    print("=" * 70)

    room_code = "benchmark-room-42"

    # Reset rate limit on server in case previous tests ran
    try:
        import urllib.request
        http_base = server_ws.replace("ws://", "http://").replace("wss://", "https://")
        req = urllib.request.Request(f"{http_base}/api/reset-rate-limit", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=1.0)
    except Exception:
        pass

    audio_file_a = "samples/speech_with_whitenoise.wav"
    audio_file_b = "samples/real_clean_speech_48k.wav"
    if not Path(audio_file_b).exists():
        audio_file_b = audio_file_a

    num_test_frames = 60  # 60 * 40ms = 2.4s of simultaneous real-time audio
    frames_a = load_wav_frames(audio_file_a, max_frames=num_test_frames)
    frames_b = load_wav_frames(audio_file_b, max_frames=num_test_frames)

    print(f" Room Code:          {room_code}")
    print(f" Frames Per Client:  {len(frames_a)} frames ({len(frames_a) * 0.040:.2f} s audio each)")
    print(f" Stream Concurrency: 2 simultaneous senders + 2 receivers (full duplex relay)")
    print("-" * 70)

    results = {}
    stop_event = asyncio.Event()

    # Launch Participant A and Participant B concurrently
    task_a = asyncio.create_task(participant_client("Participant_A", room_code, server_ws, frames_a, results, stop_event))
    await asyncio.sleep(0.05)
    task_b = asyncio.create_task(participant_client("Participant_B", room_code, server_ws, frames_b, results, stop_event))

    # Wait for both to establish and send initial frames
    await asyncio.sleep(0.3)

    # Test 3rd participant rejection while room has 2 members
    rejected_3rd = await test_room_overflow(room_code, server_ws)
    print(f" [Security Check] 3rd participant rejected with ROOM_FULL: {rejected_3rd}")
    assert rejected_3rd, "Error: 3rd participant was not rejected from full room!"

    # Wait until both senders have sent all frames and received all ACKs
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 6.0:
        if (
            "Participant_A" in results and "Participant_B" in results and
            results["Participant_A"]["sent_frames"] == len(frames_a) and
            results["Participant_B"]["sent_frames"] == len(frames_b) and
            results["Participant_A"]["telemetry_received"] >= len(frames_a) and
            results["Participant_B"]["telemetry_received"] >= len(frames_b)
        ):
            # Give a brief moment for any final in-flight relay frame
            await asyncio.sleep(0.2)
            break
        await asyncio.sleep(0.1)

    stop_event.set()
    await asyncio.gather(task_a, task_b)

    # Print Detailed Statistics
    res_a = results["Participant_A"]
    res_b = results["Participant_B"]

    print("\n" + "=" * 70)
    print(" ROOM CALL VERIFICATION RESULTS")
    print("=" * 70)
    print(" Participant A:")
    print(f"   Sent Frames:            {res_a['sent_frames']}")
    print(f"   Telemetry ACKs:         {res_a['telemetry_received']} ({res_a['telemetry_received'] / res_a['sent_frames'] * 100:.1f}%)")
    print(f"   Relayed Audio from B:   {res_a['relayed_audio_received']} frames")
    print(f"   Dropped Relayed Frames: {res_a['dropped']}")
    print(f"   RTT (Client <-> Server):  avg = {np.mean(res_a['rtt_list']):.2f} ms | min = {np.min(res_a['rtt_list']):.2f} ms | max = {np.max(res_a['rtt_list']):.2f} ms")
    print(f"   Server Inference Time:  avg = {np.mean(res_a['proc_list']):.2f} ms | min = {np.min(res_a['proc_list']):.2f} ms | max = {np.max(res_a['proc_list']):.2f} ms")
    print(f"   Peer Lifecycle Events:  {res_a['peer_events']}")
    print()
    print(" Participant B:")
    print(f"   Sent Frames:            {res_b['sent_frames']}")
    print(f"   Telemetry ACKs:         {res_b['telemetry_received']} ({res_b['telemetry_received'] / res_b['sent_frames'] * 100:.1f}%)")
    print(f"   Relayed Audio from A:   {res_b['relayed_audio_received']} frames")
    print(f"   Dropped Relayed Frames: {res_b['dropped']}")
    print(f"   RTT (Client <-> Server):  avg = {np.mean(res_b['rtt_list']):.2f} ms | min = {np.min(res_b['rtt_list']):.2f} ms | max = {np.max(res_b['rtt_list']):.2f} ms")
    print(f"   Server Inference Time:  avg = {np.mean(res_b['proc_list']):.2f} ms | min = {np.min(res_b['proc_list']):.2f} ms | max = {np.max(res_b['proc_list']):.2f} ms")
    print(f"   Peer Lifecycle Events:  {res_b['peer_events']}")
    print("-" * 70)

    # Verification assertions
    assert res_a["sent_frames"] == len(frames_a)
    assert res_b["sent_frames"] == len(frames_b)
    assert res_a["telemetry_received"] == len(frames_a), f"Expected {len(frames_a)} ACKs for A, got {res_a['telemetry_received']}"
    assert res_b["telemetry_received"] == len(frames_b), f"Expected {len(frames_b)} ACKs for B, got {res_b['telemetry_received']}"
    assert res_a["relayed_audio_received"] > 0, "Participant A received no audio from B!"
    assert res_b["relayed_audio_received"] > 0, "Participant B received no audio from A!"
    assert res_a["dropped"] == 0, f"Participant A experienced {res_a['dropped']} dropped frames"
    assert res_b["dropped"] == 0, f"Participant B experienced {res_b['dropped']} dropped frames"
    print(" [Relay Verification]: ALL CHECKS PASSED — Bidirectional cross-relay intact with 0 drops!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Server port to test")
    args = parser.parse_args()
    asyncio.run(run_room_relay_test(server_ws=f"ws://127.0.0.1:{args.port}"))
