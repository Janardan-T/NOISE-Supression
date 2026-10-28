"""Test thread counts and processing time for 1 vs 2 streams."""

import time
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.denoise import load_model
from src.realtime.server import StreamDenoiser

def test_threads():
    model, df_state = load_model("DeepFilterNet3", post_filter=False, log_level="ERROR")
    dummy = torch.zeros(1, 1920)
    from df.enhance import enhance
    with torch.no_grad():
        _ = enhance(model, df_state, dummy, pad=True)

    t = torch.linspace(0, 0.04, 1920)
    audio_signal = (0.5 * torch.sin(2 * 3.14159 * 440 * t) + 0.1 * torch.randn(1920)).clamp(-1.0, 1.0)
    pcm16 = (audio_signal * 32767).to(torch.int16).numpy().tobytes()

    for n_threads in [1, 2, 4, torch.get_num_threads()]:
        torch.set_num_threads(n_threads)
        print(f"\n--- Testing with torch.set_num_threads({n_threads}) ---", flush=True)

        # 1 stream test
        denoiser = StreamDenoiser(context_buf_size=9600)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = denoiser.process_chunk_bytes(pcm16, model, df_state)
            times.append((time.perf_counter() - t0) * 1000.0)
        print(f"1 stream (30 frames): avg = {sum(times)/len(times):.2f} ms | min = {min(times):.2f} ms | max = {max(times):.2f} ms", flush=True)

        # 2 concurrent streams
        denoiser_a = StreamDenoiser(context_buf_size=9600)
        denoiser_b = StreamDenoiser(context_buf_size=9600)
        executor = ThreadPoolExecutor(max_workers=2)
        times_2 = []
        for _ in range(30):
            t0 = time.perf_counter()
            fa = executor.submit(denoiser_a.process_chunk_bytes, pcm16, model, df_state)
            fb = executor.submit(denoiser_b.process_chunk_bytes, pcm16, model, df_state)
            fa.result()
            fb.result()
            times_2.append((time.perf_counter() - t0) * 1000.0)
        executor.shutdown()
        print(f"2 streams concurrent (30 frames): elapsed per pair = {sum(times_2)/len(times_2):.2f} ms", flush=True)

if __name__ == "__main__":
    test_threads()
