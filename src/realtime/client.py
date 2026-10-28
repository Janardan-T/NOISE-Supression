"""Real-time Audio Denoising Client.

Supports:
- Listing audio hardware devices (--list-devices)
- Streaming from microphone to WebSocket server and playback to speakers
- Automated test mode using a WAV file instead of microphone (--file)
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import websockets

# Sample rate required by DeepFilterNet
SAMPLE_RATE = 48000
DEFAULT_CHUNK_SIZE = 1920  # 1920 samples = 40 ms at 48 kHz


def list_audio_devices():
    """List available audio input and output devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n" + "=" * 65)
        print(" Available Audio Hardware Devices (sounddevice)")
        print("=" * 65)
        for i, dev in enumerate(devices):
            in_ch = dev.get("max_input_channels", 0)
            out_ch = dev.get("max_output_channels", 0)
            kind = []
            if in_ch > 0:
                kind.append(f"IN: {in_ch} ch")
            if out_ch > 0:
                kind.append(f"OUT: {out_ch} ch")
            kind_str = ", ".join(kind) if kind else "No channels"
            default_mark = ""
            if i == sd.default.device[0]:
                default_mark += " [Default Input]"
            if i == sd.default.device[1]:
                default_mark += " [Default Output]"
            print(f" [{i:2d}] {dev['name']:40s} | {kind_str:16s}{default_mark}")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"Error querying audio devices: {e}", file=sys.stderr)


async def stream_file_over_websocket(
    file_path: str,
    server_url: str = "ws://127.0.0.1:8000/ws",
    output_file: Optional[str] = "outputs/realtime_stream_test.wav",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    atten_lim_db: Optional[float] = None,
):
    """Stream an audio file in 40ms chunks to the WebSocket server and save the response.

    Automated testing mode that simulates a real-time live microphone stream.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Load audio file
    data, sr = sf.read(str(path), dtype="float32")
    # Convert stereo to mono for streaming test
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 48kHz if required
    if sr != SAMPLE_RATE:
        import torchaudio
        import torch
        tensor = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        data = resampler(tensor).squeeze(0).numpy()
        sr = SAMPLE_RATE

    # Convert to int16 PCM
    data_int16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    total_samples = len(data_int16)
    duration_sec = total_samples / SAMPLE_RATE

    print(f"\n--- Starting Automated WebSocket Stream Test ---")
    print(f" Input file:     {path} ({duration_sec:.2f}s, {total_samples} samples)")
    print(f" Target server:  {server_url}")
    print(f" Chunk size:     {chunk_size} samples ({chunk_size / SAMPLE_RATE * 1000:.1f} ms per frame)")
    print(f" Output save:    {output_file}\n")

    # Connect to server
    async with websockets.connect(server_url) as ws:
        # Send optional config
        if atten_lim_db is not None:
            import json
            await ws.send(json.dumps({"atten_lim_db": atten_lim_db}))
            _ = await ws.recv()

        received_chunks = []
        latencies_ms = []

        total_stream_start = time.perf_counter()

        # Stream chunk by chunk
        num_chunks = int(np.ceil(total_samples / chunk_size))
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = data_int16[start_idx:end_idx]

            # Zero-pad final chunk if smaller than chunk_size
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            chunk_bytes = chunk.tobytes()

            t_send = time.perf_counter()
            await ws.send(chunk_bytes)
            resp_bytes = await ws.recv()
            t_recv = time.perf_counter()

            latency_ms = (t_recv - t_send) * 1000.0
            latencies_ms.append(latency_ms)

            resp_int16 = np.frombuffer(resp_bytes, dtype=np.int16)
            received_chunks.append(resp_int16[: end_idx - start_idx])

        total_stream_time = time.perf_counter() - total_stream_start

    # Combine received chunks
    full_enhanced = np.concatenate(received_chunks)
    full_enhanced_float = full_enhanced.astype(np.float32) / 32768.0

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), full_enhanced_float, samplerate=SAMPLE_RATE, subtype="PCM_16")
        print(f"[OK] Denoised stream saved to: {out_path.resolve()}")

    avg_latency = float(np.mean(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    min_latency = float(np.min(latencies_ms))
    rtf = total_stream_time / max(duration_sec, 1e-6)

    print("\n" + "=" * 50)
    print(" WebSocket Stream Test Results")
    print("=" * 50)
    print(f" Total Chunks:        {num_chunks}")
    print(f" Audio Length:        {duration_sec:.2f} s")
    print(f" Total Process Time:  {total_stream_time:.3f} s")
    print(f" Real-Time Factor:    {rtf:.4f} ({1.0/max(rtf, 1e-4):.1f}x real-time)")
    print(f" Round-trip Latency:  avg={avg_latency:.2f}ms | min={min_latency:.2f}ms | max={max_latency:.2f}ms")
    print(f" Status:              PASSED (Bidirectional stream operational)")
    print("=" * 50 + "\n")

    return {
        "num_chunks": num_chunks,
        "duration_sec": duration_sec,
        "avg_latency_ms": avg_latency,
        "rtf": rtf,
    }


def run_live_microphone_stream(
    server_url: str = "ws://127.0.0.1:8000/ws",
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Capture live microphone audio, stream to WebSocket server, and play back to speaker."""
    import sounddevice as sd
    import queue

    mic_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio Status]: {status}", file=sys.stderr)
        mic_queue.put(bytes(indata))

    async def client_loop():
        print(f"\nConnecting to real-time server at {server_url}...")
        async with websockets.connect(server_url) as ws:
            print("Connected! Streaming live microphone audio (Press Ctrl+C to stop)...")

            # Open audio playback stream
            with sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=output_device,
                blocksize=chunk_size,
            ) as speaker:
                # Open audio input stream
                with sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    device=input_device,
                    blocksize=chunk_size,
                    callback=audio_callback,
                ):
                    while True:
                        # Non-blocking get from queue
                        try:
                            raw_chunk = mic_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.005)
                            continue

                        await ws.send(raw_chunk)
                        denoised_bytes = await ws.recv()
                        speaker.write(denoised_bytes)

    try:
        asyncio.run(client_loop())
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Audio Denoising Client")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit")
    parser.add_argument("-f", "--file", type=str, default=None, help="Audio file to stream (for automated testing)")
    parser.add_argument("-o", "--output-file", type=str, default="outputs/realtime_stream_test.wav", help="Path to save denoised output stream")
    parser.add_argument("-u", "--url", type=str, default="ws://127.0.0.1:8000/ws", help="WebSocket server URL")
    parser.add_argument("-i", "--input-device", type=int, default=None, help="Input audio device ID (see --list-devices)")
    parser.add_argument("-d", "--output-device", type=int, default=None, help="Output audio device ID (see --list-devices)")
    parser.add_argument("-c", "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Samples per chunk (default 1920 = 40ms)")
    parser.add_argument("-a", "--atten-lim", type=float, default=None, help="Attenuation limit in dB")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.file:
        asyncio.run(
            stream_file_over_websocket(
                file_path=args.file,
                server_url=args.url,
                output_file=args.output_file,
                chunk_size=args.chunk_size,
                atten_lim_db=args.atten_lim,
            )
        )
    else:
        run_live_microphone_stream(
            server_url=args.url,
            input_device=args.input_device,
            output_device=args.output_device,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
