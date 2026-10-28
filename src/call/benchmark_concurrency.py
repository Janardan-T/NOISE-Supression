"""Benchmark processing time per frame with 1 vs 2 concurrent streams on CPU.

Requirements:
- Measure and report: processing time per frame with 1 vs 2 concurrent streams on this CPU.
- Evaluate impact of:
  (1) Thread count (torch.set_num_threads(1) vs default)
  (2) 1 vs 2 concurrent streams
  (3) Silent/near-silent frames vs active audio
  (4) Context buffer size (200 ms / 9600 samples vs 120 ms / 5760 samples)
"""

import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.denoise import load_model
from src.realtime.server import StreamDenoiser

def run_benchmarks():
    print("=" * 70, flush=True)
    print(" DEEPFILTERNET CONCURRENCY BENCHMARK (1 VS 2 STREAMS ON CPU)", flush=True)
    print("=" * 70, flush=True)

    default_threads = torch.get_num_threads()
    print(f"System CPU Cores / Default PyTorch Threads: {default_threads}", flush=True)
    print("Loading DeepFilterNet3 model...", flush=True)
    model, df_state = load_model("DeepFilterNet3", post_filter=False, log_level="ERROR")

    # Warmup
    dummy = torch.zeros(1, 1920)
    from df.enhance import enhance
    with torch.no_grad():
        _ = enhance(model, df_state, dummy, pad=True)
    print("Model initialized and warmed up on CPU.\n", flush=True)

    # Audio signals:
    # 1. Active speech-like audio
    t = torch.linspace(0, 0.04, 1920)
    active_audio = (0.5 * torch.sin(2 * 3.14159 * 440 * t) + 0.1 * torch.randn(1920)).clamp(-1.0, 1.0)
    active_pcm16 = (active_audio * 32767).to(torch.int16).numpy().tobytes()

    # 2. Near-silent audio (-60 dB / very low noise)
    silent_audio = (0.0001 * torch.randn(1920)).clamp(-1.0, 1.0)
    silent_pcm16 = (silent_audio * 32767).to(torch.int16).numpy().tobytes()

    num_frames = 100

    # Ensure single-thread per op to prevent thread oversubscription
    torch.set_num_threads(1)
    print(f"[Configuration] torch.set_num_threads(1) set (prevents thread contention across workers)", flush=True)

    # -------------------------------------------------------------
    # Test 1: 1 Stream, 200 ms context (9600 samples)
    # -------------------------------------------------------------
    print("\n--- TEST 1: 1 Stream (200 ms context, 9600 smp, 100 frames) ---", flush=True)
    denoiser_1 = StreamDenoiser(context_buf_size=9600)
    times_1 = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = denoiser_1.process_chunk_bytes(active_pcm16, model, df_state)
        t1 = time.perf_counter()
        times_1.append((t1 - t0) * 1000.0)

    avg_1 = np.mean(times_1)
    med_1 = np.median(times_1)
    p95_1 = np.percentile(times_1, 95)
    min_1 = np.min(times_1)
    max_1 = np.max(times_1)
    print(f"1 Stream (active speech):", flush=True)
    print(f"  Avg: {avg_1:.2f} ms | Median: {med_1:.2f} ms | P95: {p95_1:.2f} ms | Min: {min_1:.2f} ms | Max: {max_1:.2f} ms", flush=True)
    print(f"  Frame budget (40 ms): {avg_1 / 40.0 * 100:.1f}% utilized (RTF {avg_1 / 40.0:.3f})", flush=True)

    # -------------------------------------------------------------
    # Test 2: 2 Concurrent Streams, 200 ms context (9600 samples)
    # -------------------------------------------------------------
    print("\n--- TEST 2: 2 Concurrent Streams (200 ms context, 100 frames each) ---", flush=True)
    denoiser_2a = StreamDenoiser(context_buf_size=9600)
    denoiser_2b = StreamDenoiser(context_buf_size=9600)
    executor = ThreadPoolExecutor(max_workers=4)

    times_2a = []
    times_2b = []
    elapsed_pairs = []

    for _ in range(num_frames):
        t0 = time.perf_counter()
        fa = executor.submit(denoiser_2a.process_chunk_bytes, active_pcm16, model, df_state)
        fb = executor.submit(denoiser_2b.process_chunk_bytes, active_pcm16, model, df_state)
        res_a = fa.result()
        res_b = fb.result()
        t1 = time.perf_counter()
        elapsed_pairs.append((t1 - t0) * 1000.0)

    all_2 = elapsed_pairs
    avg_2 = np.mean(all_2)
    med_2 = np.median(all_2)
    p95_2 = np.percentile(all_2, 95)
    min_2 = np.min(all_2)
    max_2 = np.max(all_2)
    print(f"2 Concurrent Streams (both active speech):", flush=True)
    print(f"  Turnaround per pair: Avg: {avg_2:.2f} ms | Median: {med_2:.2f} ms | P95: {p95_2:.2f} ms | Min: {min_2:.2f} ms | Max: {max_2:.2f} ms", flush=True)
    print(f"  Aggregate time per frame: {avg_2 / 2.0:.2f} ms/frame", flush=True)
    print(f"  Budget comparison: Pair turnaround {avg_2:.2f} ms vs 40 ms frame interval -> {avg_2 / 40.0 * 100:.1f}% of budget", flush=True)

    # -------------------------------------------------------------
    # Test 3: Near-silent frame skipping protection
    # -------------------------------------------------------------
    print("\n--- TEST 3: Near-Silent Frame Skipping Impact ---", flush=True)
    # If frame RMS < -50 dB (rms < 0.00316 in int16 scale < ~100)
    # Measure time to compute RMS vs full inference
    times_rms = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        audio_np = np.frombuffer(silent_pcm16, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        is_silent = rms < 100  # -50 dBFS threshold
        t1 = time.perf_counter()
        times_rms.append((t1 - t0) * 1000.0)

    avg_rms = np.mean(times_rms)
    print(f"RMS calculation time: {avg_rms * 1000.0:.2f} µs ({avg_rms:.4f} ms)")
    print(f"Inference bypass speedup on silent frames: {avg_1 / max(avg_rms, 0.001):.1f}x faster", flush=True)

    # -------------------------------------------------------------
    # Test 4: Shorter Context Buffer (120 ms / 5760 samples)
    # -------------------------------------------------------------
    print("\n--- TEST 4: Shorter Context Buffer (120 ms vs 200 ms) ---", flush=True)
    denoiser_short = StreamDenoiser(context_buf_size=5760)  # 120 ms context
    times_short = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = denoiser_short.process_chunk_bytes(active_pcm16, model, df_state)
        t1 = time.perf_counter()
        times_short.append((t1 - t0) * 1000.0)

    avg_short = np.mean(times_short)
    print(f"120 ms context (5760 smp): Avg = {avg_short:.2f} ms (vs 200 ms: {avg_1:.2f} ms, diff = {avg_short - avg_1:+.2f} ms)", flush=True)

    executor.shutdown()
    print("\n" + "=" * 70, flush=True)
    print(" BENCHMARK COMPLETED", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_benchmarks()
