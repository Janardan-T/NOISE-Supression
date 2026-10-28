"""Real-time Audio Denoising WebSocket Server using FastAPI and DeepFilterNet.

Provides a low-latency WebSocket endpoint (/ws) to stream raw audio chunks,
enhance them on CPU via DeepFilterNet, and return the cleaned audio stream.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Ensure project root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.denoise import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealtimeServer")

app = FastAPI(title="DeepFilterNet Realtime Streaming Server")

# Model and state initialized once at server startup
MODEL = None
DF_STATE = None


@app.on_event("startup")
def startup_event():
    """Load and warm up DeepFilterNet model on CPU."""
    global MODEL, DF_STATE
    logger.info("Initializing DeepFilterNet3 for real-time streaming...")
    MODEL, DF_STATE = load_model(model_name="DeepFilterNet3", post_filter=False, log_level="ERROR")
    # Warmup with a dummy 40ms chunk
    dummy = torch.zeros(1, 1920)
    from df.enhance import enhance
    with torch.no_grad():
        _ = enhance(MODEL, DF_STATE, dummy, pad=True)
    logger.info("Model ready! WebSocket endpoint: ws://127.0.0.1:8000/ws")


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "DeepFilterNet Realtime Server",
        "sample_rate": DF_STATE.sr() if DF_STATE else 48000,
        "websocket_url": "ws://127.0.0.1:8000/ws",
    }


class StreamDenoiser:
    """Stateful audio stream denoiser with 200 ms context re-priming buffer."""

    def __init__(self, context_buf_size: int = 9600):
        self.context_buf_size = context_buf_size
        self.rolling_buffer = torch.zeros(1, context_buf_size)

    def process_chunk_bytes(self, raw_bytes: bytes, model, df_state, atten_lim_db: float = None) -> bytes:
        """Process one chunk of int16 PCM audio bytes, return enhanced int16 PCM audio bytes."""
        if len(raw_bytes) == 0:
            return b""

        # Convert int16 PCM bytes to float32 PyTorch tensor in [-1.0, 1.0]
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # [1, chunk_samples]
        chunk_len = audio_tensor.shape[1]

        # Update rolling context buffer
        if chunk_len < self.context_buf_size:
            self.rolling_buffer = torch.cat([self.rolling_buffer[:, chunk_len:], audio_tensor], dim=-1)
        else:
            self.rolling_buffer = audio_tensor[:, -self.context_buf_size:]

        # Run DeepFilterNet enhancement on CPU with context
        from df.enhance import enhance
        with torch.no_grad():
            enhanced_buf = enhance(
                model=model,
                df_state=df_state,
                audio=self.rolling_buffer,
                pad=True,
                atten_lim_db=atten_lim_db,
            )

        # Extract the enhanced frame corresponding to the current chunk
        enhanced_tensor = enhanced_buf[:, -chunk_len:]

        # Convert enhanced tensor back to int16 PCM bytes
        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np, -1.0, 1.0)
        out_int16 = (enhanced_np * 32767.0).astype(np.int16)
        return out_int16.tobytes()

    def update_context_only(self, raw_bytes: bytes):
        """Update the context buffer with incoming audio without running inference (for bypass mode)."""
        if len(raw_bytes) == 0:
            return
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        chunk_len = audio_tensor.shape[1]
        if chunk_len < self.context_buf_size:
            self.rolling_buffer = torch.cat([self.rolling_buffer[:, chunk_len:], audio_tensor], dim=-1)
        else:
            self.rolling_buffer = audio_tensor[:, -self.context_buf_size:]


@app.websocket("/ws")
async def websocket_denoise_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket for low-latency audio chunk processing.

    Protocol:
    - Optional first message: JSON text with configuration, e.g. {"atten_lim_db": 30.0}
    - Subsequent messages: Binary frames containing raw 16-bit PCM audio samples.
    - Server responds with binary frames of denoised 16-bit PCM audio.
    """
    await websocket.accept()
    logger.info("Client connected to real-time audio stream.")

    atten_lim_db = None
    stream_denoiser = StreamDenoiser(context_buf_size=9600)

    try:
        while True:
            # Receive either text config or binary audio chunk
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnected from audio stream.")
                break

            if "text" in message and message["text"]:
                try:
                    config = json.loads(message["text"])
                    if "atten_lim_db" in config:
                        atten_lim_db = config["atten_lim_db"]
                        logger.info(f"Updated attenuation limit: {atten_lim_db} dB")
                    await websocket.send_text(json.dumps({"status": "config_updated", "atten_lim_db": atten_lim_db}))
                except Exception as e:
                    logger.warning(f"Error parsing config message: {e}")
                continue

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if len(raw_bytes) == 0:
                    continue

                out_bytes = stream_denoiser.process_chunk_bytes(raw_bytes, MODEL, DF_STATE, atten_lim_db=atten_lim_db)
                await websocket.send_bytes(out_bytes)

    except WebSocketDisconnect:
        logger.info("Client disconnected from audio stream.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI WebSocket server."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realtime Denoising Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()
    run_server(args.host, args.port)
