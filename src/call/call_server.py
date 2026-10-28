"""FastAPI Live Voice Call Server with DeepFilterNet Speech Enhancement.

Features:
- Web interface at / (landing page with Create/Join room & Self-Test) and /room/{room_code} (two-party room calls).
- Robust room code generator (unguessable alphanumeric without look-alike characters).
- WebSocket /ws/call for single-user self-test loopback.
- WebSocket /ws/room/{room_code} for two-party cross-relayed voice calls (max 2 participants).
- Concurrency limit: configurable max active rooms (--max-rooms) returning ROOMS_BUSY.
- Rate limiting: join attempts per IP throttled with RATE_LIMITED error.
- Room cleanup: empty rooms deleted immediately, background reaper for idle/stale rooms.
- Configurable --host (127.0.0.1 or 0.0.0.0 for LAN) and --public-url for invite link formatting.
- Prominent startup warnings regarding browser microphone HTTPS requirements and authentication limits.
"""

import argparse
import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
import torch
import uvicorn

# Ensure project root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = workspace_root
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.call import database
from src.denoise import load_model
from src.realtime.server import StreamDenoiser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CallServer] [%(levelname)s] %(message)s")
logger = logging.getLogger("CallServer")

app = FastAPI(title="DeepFilterNet Live Call Server")

# Directory paths
CURRENT_DIR = Path(__file__).resolve().parent
STATIC_DIR = CURRENT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Shared model state
MODEL = None
DF_STATE = None
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="df_worker")

# Protocol constants
CLIENT_HEADER_FORMAT = ">IIBBH"  # seq (uint32), client_ts (uint32), mode (uint8), flags (uint8), reserved (uint16)
CLIENT_HEADER_SIZE = struct.calcsize(CLIENT_HEADER_FORMAT)  # 12 bytes

SERVER_HEADER_FORMAT = ">IIfBBH"  # seq (uint32), client_ts (uint32), proc_ms (float32), mode (uint8), flags (uint8), reserved (uint16)
SERVER_HEADER_SIZE = struct.calcsize(SERVER_HEADER_FORMAT)  # 16 bytes

EXPECTED_CHUNK_SAMPLES = 1920  # 40 ms at 48 kHz
EXPECTED_AUDIO_BYTES = EXPECTED_CHUNK_SAMPLES * 2  # 3840 bytes int16

# Flag bits
FLAG_SILENT_FRAME = 0x01      # Inference skipped due to near-silence
FLAG_OVERLOAD_BYPASS = 0x02   # Backlog exceeded, inference bypassed
FLAG_SHORT_CONTEXT = 0x04     # 120 ms context used instead of 200 ms

# Overload protection thresholds
SILENCE_RMS_THRESHOLD = 20.0   # ~ -64 dBFS on 16-bit integer scale
MAX_BACKLOG_FRAMES = 2         # If queue backlog >= 2 frames, bypass inference

def get_lan_ip() -> str:
    """Detect machine's primary LAN IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_ssl_certs(lan_ip: str, cert_path: Path, key_path: Path):
    """Ensure SSL certificate and private key exist; auto-generate via OpenSSL if needed."""
    if cert_path.exists() and key_path.exists():
        return
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    openssl_candidates = [
        "openssl",
        r"C:\Program Files\Git\usr\bin\openssl.exe",
        r"C:\Program Files (x86)\Git\usr\bin\openssl.exe",
    ]
    openssl_cmd = None
    for cand in openssl_candidates:
        try:
            res = subprocess.run([cand, "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                openssl_cmd = cand
                break
        except Exception:
            continue

    if openssl_cmd:
        san = f"subjectAltName=IP:{lan_ip},IP:127.0.0.1,DNS:localhost"
        cmd = [
            openssl_cmd, "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-sha256", "-days", "365", "-nodes",
            "-subj", f"/CN={lan_ip}", "-addext", san
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


_detected_lan_ip = get_lan_ip()
# Configuration parameters (can be set via CLI or programmatic config)
CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,
    "public_url": None,
    "lan_ip": _detected_lan_ip,
    "lan_url": f"http://{_detected_lan_ip}:8001",
    "use_ssl": False,
    "max_rooms": 2,
    "idle_timeout": 300.0,  # 5 minutes
    "rate_limit_max": 20,   # attempts
    "rate_limit_window": 30.0,  # seconds
}

# Unguessable room code character set without look-alikes:
# Excludes: 0, O (zero/letter O), 1, I, L (one / capital I / lowercase L)
# 31 uppercase letters & numbers
SAFE_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
ROOM_CODE_REGEX = re.compile(r"^[A-Za-z0-9_-]{3,32}$")


def generate_room_code(length: int = 6) -> str:
    """Generate a random, unguessable room code without look-alike characters."""
    return "".join(secrets.choice(SAFE_CODE_ALPHABET) for _ in range(length))


def is_valid_room_code(code: str) -> bool:
    """Validate room code format."""
    if not code or not isinstance(code, str):
        return False
    return bool(ROOM_CODE_REGEX.match(code.strip()))


class JoinRateLimiter:
    """In-memory sliding-window rate limiter for join attempts per IP."""

    def __init__(self, max_attempts: int = 20, window_seconds: float = 30.0):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()

    async def is_allowed(self, ip: str) -> bool:
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            history = [t for t in self.attempts.get(ip, []) if t > cutoff]
            if len(history) >= self.max_attempts:
                self.attempts[ip] = history
                return False
            history.append(now)
            self.attempts[ip] = history
            return True

    async def reset(self):
        async with self.lock:
            self.attempts.clear()


RATE_LIMITER = JoinRateLimiter()


class ParticipantSession:
    """Manages audio state, queue, and serialized sending for a connected participant."""

    def __init__(
        self,
        participant_id: str,
        websocket: WebSocket,
        context_samples: int = 9600,
        email: Optional[str] = None,
        name: Optional[str] = None
    ):
        self.participant_id = participant_id
        self.websocket = websocket
        self.email = email
        self.name = name
        self.send_lock = asyncio.Lock()
        self.denoiser = StreamDenoiser(context_buf_size=context_samples)
        self.frame_queue = asyncio.Queue(maxsize=30)
        self.atten_lim_db: Optional[float] = None
        self.use_short_context = (context_samples < 9600)

    def set_context_size(self, context_samples: int):
        if context_samples != self.denoiser.context_buf_size:
            self.denoiser = StreamDenoiser(context_buf_size=context_samples)
            self.use_short_context = (context_samples < 9600)

    async def send_bytes(self, data: bytes):
        """Thread/task-safe serialized write of binary frame."""
        async with self.send_lock:
            await self.websocket.send_bytes(data)

    async def send_text(self, text: str):
        """Thread/task-safe serialized write of JSON text message."""
        async with self.send_lock:
            await self.websocket.send_text(text)


class UserConnectionManager:
    """Tracks active online user WebSockets for real-time signaling & presence."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()

    async def connect(self, email: str, websocket: WebSocket):
        clean_email = email.strip().lower()
        async with self.lock:
            self.connections[clean_email] = websocket
        database.set_user_status(clean_email, "online")
        await self.broadcast_presence(clean_email, "online")

    async def disconnect(self, email: str):
        clean_email = email.strip().lower()
        async with self.lock:
            if clean_email in self.connections:
                del self.connections[clean_email]
        database.set_user_status(clean_email, "offline")
        await self.broadcast_presence(clean_email, "offline")

    async def send_to_user(self, email: str, message: dict) -> bool:
        clean_email = email.strip().lower()
        async with self.lock:
            ws = self.connections.get(clean_email)
        if ws:
            try:
                await ws.send_text(json.dumps(message))
                return True
            except Exception:
                return False
        return False

    async def broadcast_presence(self, email: str, status: str):
        payload = {"type": "USER_PRESENCE", "email": email.strip().lower(), "status": status}
        async with self.lock:
            sockets = list(self.connections.values())
        for ws in sockets:
            try:
                asyncio.create_task(ws.send_text(json.dumps(payload)))
            except Exception:
                pass


USER_MANAGER = UserConnectionManager()


class CallRoom:
    """Represents a two-party call room with activity tracking."""

    def __init__(self, room_code: str):
        self.room_code = room_code
        self.participants: Dict[str, ParticipantSession] = {}
        self.created_at = time.time()
        self.last_activity = time.time()
        self.lock = asyncio.Lock()

    def touch(self):
        self.last_activity = time.time()

    def is_full(self) -> bool:
        return len(self.participants) >= 2

    def is_empty(self) -> bool:
        return len(self.participants) == 0

    def is_idle(self, timeout_seconds: float) -> bool:
        return (time.time() - self.last_activity) > timeout_seconds

    def get_peer(self, participant_id: str) -> Optional[ParticipantSession]:
        for pid, session in self.participants.items():
            if pid != participant_id:
                return session
        return None


# Global active rooms: room_code -> CallRoom
ACTIVE_ROOMS: Dict[str, CallRoom] = {}
ROOMS_LOCK = asyncio.Lock()
CLEANUP_TASK: Optional[asyncio.Task] = None


async def cleanup_idle_rooms_task():
    """Background task to reap empty and abandoned/idle rooms."""
    while True:
        try:
            await asyncio.sleep(10.0)
            timeout = CONFIG["idle_timeout"]
            now = time.time()
            rooms_to_delete: List[str] = []

            async with ROOMS_LOCK:
                for code, room in list(ACTIVE_ROOMS.items()):
                    if room.is_empty():
                        rooms_to_delete.append(code)
                    elif room.is_idle(timeout):
                        logger.info(f"[Room Reaper] Room {code} idle for {now - room.last_activity:.1f}s (> {timeout}s). Closing.")
                        rooms_to_delete.append(code)

                for code in rooms_to_delete:
                    if code in ACTIVE_ROOMS:
                        r = ACTIVE_ROOMS[code]
                        # Notify and close any stale connections
                        async with r.lock:
                            for p in list(r.participants.values()):
                                try:
                                    await p.send_text(json.dumps({
                                        "type": "error",
                                        "code": "ROOM_IDLE_TIMEOUT",
                                        "message": "Room closed due to prolonged inactivity."
                                    }))
                                    await p.websocket.close(code=4000)
                                except Exception:
                                    pass
                            r.participants.clear()
                        del ACTIVE_ROOMS[code]
                        logger.info(f"[Room Reaper] Deleted room: {code}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Room Reaper Error]: {e}")


def get_client_ip(websocket_or_request) -> str:
    """Extract client IP, taking into account forwarded headers if present."""
    client = websocket_or_request.client
    if not client:
        return "127.0.0.1"
    headers = getattr(websocket_or_request, "headers", {})
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return client.host


def render_html_with_config(room_code: Optional[str] = None) -> str:
    """Inject server runtime config directly into index.html for zero-latency client bootstrap."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>Call interface initializing...</h1>"

    html = index_file.read_text(encoding="utf-8")
    server_config_json = json.dumps({
        "public_url": CONFIG["public_url"],
        "lan_ip": CONFIG.get("lan_ip"),
        "lan_url": CONFIG.get("lan_url"),
        "use_ssl": CONFIG.get("use_ssl", False),
        "max_rooms": CONFIG["max_rooms"],
        "idle_timeout": CONFIG["idle_timeout"],
        "sample_rate": DF_STATE.sr() if DF_STATE else 48000,
        "frame_samples": EXPECTED_CHUNK_SAMPLES,
        "preloaded_room_code": room_code,
    })

    script_injection = f"""<script>
window.SERVER_CONFIG = {server_config_json};
</script>
"""
    return html.replace("</head>", f"{script_injection}</head>")


def render_invalid_code_page(code: str) -> str:
    """Render friendly HTML error page for invalid room code syntax."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Invalid Room Code — DeepFilterNet Call Studio</title>
  <style>
    body {{
      background: #0a0e17;
      color: #f9fafb;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 16px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #374151;
      border-radius: 12px;
      padding: 32px;
      max-width: 480px;
      text-align: center;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 12px; color: #ef4444; }}
    p {{ color: #9ca3af; font-size: 0.95rem; margin-bottom: 24px; line-height: 1.5; }}
    code {{ background: #1f2937; padding: 2px 6px; border-radius: 4px; color: #60a5fa; font-family: monospace; }}
    .btn {{
      display: inline-block;
      background: #3b82f6;
      color: white;
      text-decoration: none;
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      transition: background 0.2s;
    }}
    .btn:hover {{ background: #2563eb; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Invalid Room Code</h1>
    <p>The room code <code>{code}</code> is invalid. Room codes must be between 4 and 32 characters using letters, numbers, and hyphens without special symbols.</p>
    <a href="/" class="btn">Return to Home</a>
  </div>
</body>
</html>
"""


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan_handler(app_instance: FastAPI):
    """Load and warm up DeepFilterNet model on CPU at server startup, and initialize SQLite database."""
    global MODEL, DF_STATE, CLEANUP_TASK
    database.init_db()
    torch.set_num_threads(1)
    logger.info("Initializing DeepFilterNet3 for Live Voice Calls (PyTorch threads=1)...")
    MODEL, DF_STATE = load_model(model_name="DeepFilterNet3", post_filter=False, log_level="ERROR")

    # Warmup inference
    dummy = torch.zeros(1, 1920)
    from df.enhance import enhance
    with torch.no_grad():
        _ = enhance(MODEL, DF_STATE, dummy, pad=True)
    logger.info("DeepFilterNet3 model warmed up and ready.")

    # Start room cleanup reaper
    CLEANUP_TASK = asyncio.create_task(cleanup_idle_rooms_task())
    yield
    if CLEANUP_TASK:
        CLEANUP_TASK.cancel()
    EXECUTOR.shutdown(wait=False)

app.router.lifespan_context = lifespan_handler


# ==============================================================================
# HTTP Routes
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve live call web interface landing page."""
    return HTMLResponse(content=render_html_with_config(room_code=None))


@app.get("/room/{room_code}", response_class=HTMLResponse)
async def get_room_page(room_code: str):
    """Serve live call web interface with pre-loaded room code or error if invalid."""
    cleaned_code = room_code.strip()
    if not is_valid_room_code(cleaned_code):
        return HTMLResponse(content=render_invalid_code_page(room_code), status_code=400)
    return HTMLResponse(content=render_html_with_config(room_code=cleaned_code))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/create-room")
async def api_create_room():
    """Generate a random, unguessable room code without look-alike characters."""
    code = generate_room_code(6)
    return {"room_code": code}


@app.get("/api/config")
async def api_config():
    """Public runtime server configuration for clients."""
    return {
        "public_url": CONFIG["public_url"],
        "max_rooms": CONFIG["max_rooms"],
        "idle_timeout": CONFIG["idle_timeout"],
        "active_rooms_count": len(ACTIVE_ROOMS),
        "frame_samples": EXPECTED_CHUNK_SAMPLES,
        "sample_rate": DF_STATE.sr() if DF_STATE else 48000,
    }


@app.post("/api/reset-rate-limit")
async def api_reset_rate_limit():
    """Reset rate limiter state (used in automated test suites)."""
    await RATE_LIMITER.reset()
    return {"status": "ok"}


class LoginRequest(BaseModel):
    email: str
    display_name: str


@app.post("/api/auth/login")
async def api_login(payload: LoginRequest):
    """Authenticate or register user with email and display name."""
    if not payload.email or "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    user = database.register_or_update_user(payload.email, payload.display_name, status="online")
    return {"status": "success", "user": user}


@app.get("/api/users")
async def api_get_users():
    """List registered users with real-time online/offline presence."""
    return {"users": database.list_all_users()}


@app.get("/api/calls/history")
async def api_get_call_history(email: str):
    """Get call records for a user."""
    if not email:
        raise HTTPException(status_code=400, detail="Missing email parameter.")
    return {"calls": database.get_user_call_logs(email)}


@app.delete("/api/calls/history")
async def api_delete_call_history(email: str, call_id: str):
    """Delete a single call record."""
    if not email or not call_id:
        raise HTTPException(status_code=400, detail="Missing email or call_id.")
    success = database.delete_call_log(call_id, email)
    return {"status": "success", "deleted": success}


@app.delete("/api/calls/history/all")
async def api_clear_call_history(email: str):
    """Clear all call history records for a user."""
    if not email:
        raise HTTPException(status_code=400, detail="Missing email parameter.")
    database.clear_user_call_logs(email)
    return {"status": "success", "cleared": True}


@app.get("/api/tunnel")
async def api_tunnel():
    """Retrieve active tunnel URL or LAN connection info."""
    return {
        "status": "online" if CONFIG["public_url"] else "local",
        "tunnel_url": CONFIG["public_url"],
        "lan_url": CONFIG["lan_url"],
        "lan_ip": CONFIG.get("lan_ip", "127.0.0.1"),
        "port": CONFIG.get("port", 8001),
        "use_ssl": CONFIG.get("use_ssl", False),
    }


@app.get("/api/ai/benchmark")
async def api_ai_benchmark():
    """Real-time CPU benchmark of server DeepFilterNet3 neural inference."""
    global MODEL, DF_STATE
    if MODEL is None or DF_STATE is None:
        MODEL, DF_STATE = load_model(model_name="DeepFilterNet3", post_filter=False, log_level="ERROR")

    t0 = time.perf_counter()
    from df.enhance import enhance
    dummy = torch.zeros(1, EXPECTED_CHUNK_SAMPLES)
    with torch.no_grad():
        _ = enhance(MODEL, DF_STATE, dummy, pad=True)
    t1 = time.perf_counter()
    proc_ms = (t1 - t0) * 1000.0
    rtf = proc_ms / 40.0
    return {
        "status": "ok",
        "model": "DeepFilterNet3",
        "sample_rate": DF_STATE.sr() if DF_STATE else 48000,
        "chunk_samples": EXPECTED_CHUNK_SAMPLES,
        "chunk_ms": 40.0,
        "latency_ms": round(proc_ms, 2),
        "rtf": round(rtf, 3),
        "device": "CPU"
    }


@app.get("/api/qr")
async def api_qr_code(data: str = "http://localhost:8001"):
    """Dynamically generate a QR code PNG image on-the-fly for any mobile connection URL."""
    import qrcode
    from io import BytesIO
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ==============================================================================
# User Signaling WebSocket (/ws/user/{email})
# ==============================================================================
@app.websocket("/ws/user/{email}")
async def websocket_user_signaling(websocket: WebSocket, email: str):
    """User-level persistent WebSocket for online presence, dialing, and incoming call alerts."""
    clean_email = email.strip().lower()
    await websocket.accept()
    await USER_MANAGER.connect(clean_email, websocket)
    logger.info(f"[User Signal] Connected: {clean_email}")
    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            mtype = data.get("type")
            target = (data.get("target") or "").strip().lower()

            if mtype == "PING":
                await websocket.send_text(json.dumps({"type": "PONG"}))

            elif mtype == "CALL_INVITE":
                call_id = data.get("call_id") or f"call_{secrets.token_hex(4)}"
                caller_name = data.get("caller_name") or clean_email
                room_code = data.get("room_code") or generate_room_code(6)
                database.record_call_start(call_id, clean_email, target)
                sent = await USER_MANAGER.send_to_user(target, {
                    "type": "INCOMING_CALL",
                    "caller_email": clean_email,
                    "caller_name": caller_name,
                    "call_id": call_id,
                    "room_code": room_code,
                })
                if not sent:
                    await websocket.send_text(json.dumps({
                        "type": "CALL_ERROR",
                        "message": f"User {target} is currently offline."
                    }))

            elif mtype == "CALL_RESPONSE":
                accepted = data.get("accepted", False)
                call_id = data.get("call_id")
                room_code = data.get("room_code")
                if accepted:
                    database.update_call_status(call_id, "connected")
                else:
                    database.update_call_status(call_id, "missed")
                await USER_MANAGER.send_to_user(target, {
                    "type": "CALL_RESPONSE",
                    "callee_email": clean_email,
                    "accepted": accepted,
                    "call_id": call_id,
                    "room_code": room_code
                })

            elif mtype == "CALL_HANGUP":
                call_id = data.get("call_id")
                duration = data.get("duration_sec", 0)
                if call_id:
                    database.update_call_status(call_id, "connected", duration_sec=duration)
                if target:
                    await USER_MANAGER.send_to_user(target, {
                        "type": "CALL_HANGUP",
                        "sender": clean_email,
                        "call_id": call_id
                    })

    except WebSocketDisconnect:
        await USER_MANAGER.disconnect(clean_email)
        logger.info(f"[User Signal] Disconnected: {clean_email}")
    except Exception as e:
        logger.error(f"[User Signal] Error for {clean_email}: {e}")
        await USER_MANAGER.disconnect(clean_email)


@app.get("/health")
def health():
    """Server health status endpoint."""
    return {
        "status": "online",
        "service": "DeepFilterNet Live Call Server",
        "sample_rate": DF_STATE.sr() if DF_STATE else 48000,
        "frame_size_ms": 40.0,
        "frame_samples": 1920,
        "active_rooms": len(ACTIVE_ROOMS),
        "max_rooms": CONFIG["max_rooms"],
        "ws_self_test": f"ws://{CONFIG['host']}:{CONFIG['port']}/ws/call",
        "ws_room_template": f"ws://{CONFIG['host']}:{CONFIG['port']}/ws/room/{{room_code}}",
    }


# ==============================================================================
# Phase A: Single-User Self-Test Endpoint (/ws/call)
# ==============================================================================
@app.websocket("/ws/call")
async def websocket_self_test_endpoint(websocket: WebSocket):
    """Low-latency bidirectional WebSocket for single-user self-test loopback."""
    await websocket.accept()
    client_host = get_client_ip(websocket)
    logger.info(f"[Self-Test] Connected: {client_host}")

    session = ParticipantSession(client_host, websocket, context_samples=9600)
    loop = asyncio.get_running_loop()

    async def audio_processor():
        while True:
            item = await session.frame_queue.get()
            if item is None:
                break

            raw_audio, seq, client_ts, mode, in_flags = item
            out_flags = 0
            if session.use_short_context or (in_flags & FLAG_SHORT_CONTEXT):
                out_flags |= FLAG_SHORT_CONTEXT
                if session.denoiser.context_buf_size != 5760:
                    session.set_context_size(5760)

            t0 = time.perf_counter()

            # Protection (a): Near-silent frame skip
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)) if len(audio_np) > 0 else 0.0

            backlog = session.frame_queue.qsize()

            if mode == 0 and rms < SILENCE_RMS_THRESHOLD:
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                out_flags |= FLAG_SILENT_FRAME
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            elif mode == 0 and backlog >= MAX_BACKLOG_FRAMES:
                # Protection (b): Backlog bypass check
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                out_flags |= FLAG_OVERLOAD_BYPASS
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            elif mode == 0:
                out_audio = await loop.run_in_executor(
                    EXECUTOR,
                    session.denoiser.process_chunk_bytes,
                    raw_audio,
                    MODEL,
                    DF_STATE,
                    session.atten_lim_db,
                )
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            else:
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0

            # Pack response header (16 bytes) + audio payload (3840 bytes)
            resp_header = struct.pack(
                SERVER_HEADER_FORMAT,
                seq,
                client_ts,
                float(proc_ms),
                mode,
                out_flags,
                0,
            )

            try:
                await session.send_bytes(resp_header + out_audio)
            except Exception:
                break

    proc_task = asyncio.create_task(audio_processor())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    mtype = payload.get("type")
                    if mtype == "ping":
                        await session.send_text(json.dumps({"type": "pong", "time": time.time()}))
                    elif mtype == "config":
                        if "atten_lim_db" in payload:
                            session.atten_lim_db = payload["atten_lim_db"]
                        if "short_context" in payload:
                            c_smp = 5760 if payload["short_context"] else 9600
                            session.set_context_size(c_smp)
                        await session.send_text(json.dumps({
                            "type": "config_ack",
                            "atten_lim_db": session.atten_lim_db,
                            "short_context": session.use_short_context
                        }))
                except Exception as e:
                    logger.warning(f"[Self-Test] Control error: {e}")
                continue

            if "bytes" in msg and msg["bytes"]:
                raw_packet = msg["bytes"]
                if len(raw_packet) < CLIENT_HEADER_SIZE:
                    continue

                header_bytes = raw_packet[:CLIENT_HEADER_SIZE]
                raw_audio = raw_packet[CLIENT_HEADER_SIZE:]
                seq, client_ts, mode, in_flags, _ = struct.unpack(CLIENT_HEADER_FORMAT, header_bytes)

                try:
                    session.frame_queue.put_nowait((raw_audio, seq, client_ts, mode, in_flags))
                except asyncio.QueueFull:
                    pass

    except WebSocketDisconnect:
        logger.info(f"[Self-Test] Disconnected: {client_host}")
    except Exception as e:
        logger.error(f"[Self-Test] Error {client_host}: {e}")
    finally:
        await session.frame_queue.put(None)
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


# ==============================================================================
# Phase B: Two-Party Room Call Endpoint (/ws/room/{room_code})
# ==============================================================================
@app.websocket("/ws/room/{room_code}")
async def websocket_room_endpoint(websocket: WebSocket, room_code: str):
    """Bidirectional WebSocket for two-party room voice calls with cross-relay."""
    await websocket.accept()
    client_host = get_client_ip(websocket)
    pid = f"{client_host}_{secrets.token_hex(3)}"
    code_clean = room_code.strip()

    # 1. Validate room code
    if not is_valid_room_code(code_clean):
        logger.warning(f"[Room Connect] Rejected {pid}: Invalid room code format '{room_code}'")
        await websocket.send_text(json.dumps({
            "type": "error",
            "code": "INVALID_ROOM_CODE",
            "message": "Invalid room code format. Room code must be 4-32 alphanumeric characters."
        }))
        await websocket.close(code=4004)
        return

    # 2. Rate limit join attempts per IP
    if not await RATE_LIMITER.is_allowed(client_host):
        logger.warning(f"[Room Connect] Rejected {pid} ({client_host}): Join rate limit exceeded.")
        await websocket.send_text(json.dumps({
            "type": "error",
            "code": "RATE_LIMITED",
            "message": "Too many join attempts. Please slow down and try again shortly."
        }))
        await websocket.close(code=4003)
        return

    logger.info(f"[Room {code_clean}] Participant {pid} attempting to join...")

    # 3. Check Room Existence and Max Active Rooms
    async with ROOMS_LOCK:
        if code_clean not in ACTIVE_ROOMS:
            if len(ACTIVE_ROOMS) >= CONFIG["max_rooms"]:
                logger.warning(f"[Room {code_clean}] Rejected {pid}: Max active rooms ({CONFIG['max_rooms']}) reached.")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "ROOMS_BUSY",
                    "message": f"Server busy: maximum active rooms limit ({CONFIG['max_rooms']}) reached. Please try again later."
                }))
                await websocket.close(code=4002)
                return
            ACTIVE_ROOMS[code_clean] = CallRoom(code_clean)
        room = ACTIVE_ROOMS[code_clean]
        room.touch()

    has_joined = False
    session: Optional[ParticipantSession] = None

    # 4. Enforce 2-Participant Room Limit
    async with room.lock:
        if room.is_full():
            logger.warning(f"[Room {code_clean}] Rejected {pid}: Room full (max 2 participants)")
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "ROOM_FULL",
                "message": "Room is full (maximum 2 participants allowed)."
            }))
            await websocket.close(code=4001)
            return

        email_param = websocket.query_params.get("email")
        name_param = websocket.query_params.get("name")
        session = ParticipantSession(pid, websocket, context_samples=9600, email=email_param, name=name_param)
        room.participants[pid] = session
        room.touch()
        has_joined = True
        peer_session = room.get_peer(pid)
        if peer_session and peer_session.email and session.email:
            database.record_room_call(code_clean, peer_session.email, session.email)

    logger.info(f"[Room {code_clean}] {pid} joined successfully. Total participants in room: {len(room.participants)}")

    # Send welcome acknowledgment
    await session.send_text(json.dumps({
        "type": "joined",
        "room_code": code_clean,
        "participant_id": pid,
        "participant_count": len(room.participants),
        "peer_present": (peer_session is not None)
    }))

    # Notify peer if already present
    if peer_session:
        try:
            await peer_session.send_text(json.dumps({
                "type": "peer_joined",
                "peer_id": pid,
                "participant_count": 2
            }))
        except Exception as e:
            logger.warning(f"[Room {code_clean}] Failed to notify peer of join: {e}")

    loop = asyncio.get_running_loop()

    async def room_audio_processor():
        while True:
            item = await session.frame_queue.get()
            if item is None:
                break

            raw_audio, seq, client_ts, mode, in_flags = item
            room.touch()
            out_flags = 0
            if session.use_short_context or (in_flags & FLAG_SHORT_CONTEXT):
                out_flags |= FLAG_SHORT_CONTEXT
                if session.denoiser.context_buf_size != 5760:
                    session.set_context_size(5760)

            t0 = time.perf_counter()

            # Protection (a): Near-silent frame skip
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)) if len(audio_np) > 0 else 0.0

            backlog = session.frame_queue.qsize()

            if mode == 0 and rms < SILENCE_RMS_THRESHOLD:
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                out_flags |= FLAG_SILENT_FRAME
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            elif mode == 0 and backlog >= MAX_BACKLOG_FRAMES:
                # Protection (b): Backlog queue overload protection
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                out_flags |= FLAG_OVERLOAD_BYPASS
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            elif mode == 0:
                out_audio = await loop.run_in_executor(
                    EXECUTOR,
                    session.denoiser.process_chunk_bytes,
                    raw_audio,
                    MODEL,
                    DF_STATE,
                    session.atten_lim_db,
                )
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
            else:
                session.denoiser.update_context_only(raw_audio)
                out_audio = raw_audio
                t1 = time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0

            # Server response header (16 bytes)
            resp_header = struct.pack(
                SERVER_HEADER_FORMAT,
                seq,
                client_ts,
                float(proc_ms),
                mode,
                out_flags,
                0,
            )

            # 1. Return telemetry ACK to sender
            try:
                await session.send_bytes(resp_header)
            except Exception:
                break

            # 2. Relay audio packet to peer
            async with room.lock:
                current_peer = room.get_peer(pid)

            if current_peer:
                try:
                    await current_peer.send_bytes(resp_header + out_audio)
                except Exception as e:
                    logger.warning(f"[Room {code_clean}] Failed to relay audio to peer: {e}")

    proc_task = asyncio.create_task(room_audio_processor())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Control messages
            if "text" in msg and msg["text"]:
                room.touch()
                try:
                    payload = json.loads(msg["text"])
                    mtype = payload.get("type")
                    if mtype == "ping":
                        await session.send_text(json.dumps({"type": "pong", "time": time.time()}))
                    elif mtype == "config":
                        if "atten_lim_db" in payload:
                            session.atten_lim_db = payload["atten_lim_db"]
                        if "short_context" in payload:
                            c_smp = 5760 if payload["short_context"] else 9600
                            session.set_context_size(c_smp)
                        await session.send_text(json.dumps({
                            "type": "config_ack",
                            "atten_lim_db": session.atten_lim_db,
                            "short_context": session.use_short_context
                        }))
                except Exception as e:
                    logger.warning(f"[Room {code_clean}] Control error: {e}")
                continue

            # Binary audio frame
            if "bytes" in msg and msg["bytes"]:
                raw_packet = msg["bytes"]
                if len(raw_packet) < CLIENT_HEADER_SIZE:
                    continue

                header_bytes = raw_packet[:CLIENT_HEADER_SIZE]
                raw_audio = raw_packet[CLIENT_HEADER_SIZE:]
                seq, client_ts, mode, in_flags, _ = struct.unpack(CLIENT_HEADER_FORMAT, header_bytes)

                try:
                    session.frame_queue.put_nowait((raw_audio, seq, client_ts, mode, in_flags))
                except asyncio.QueueFull:
                    pass

    except WebSocketDisconnect:
        logger.info(f"[Room {code_clean}] Participant {pid} disconnected normally.")
    except Exception as e:
        logger.error(f"[Room {code_clean}] Error for {pid}: {e}")
    finally:
        await session.frame_queue.put(None)
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass

        if has_joined:
            # Clean teardown for admitted participant
            async with room.lock:
                if pid in room.participants:
                    del room.participants[pid]
                remaining_peer = room.get_peer(pid)
                room.touch()

            logger.info(f"[Room {code_clean}] {pid} left. Remaining participants: {len(room.participants)}")

            # Notify remaining peer
            if remaining_peer:
                try:
                    await remaining_peer.send_text(json.dumps({
                        "type": "peer_left",
                        "peer_id": pid,
                        "participant_count": 1
                    }))
                except Exception:
                    pass

            # Clean up empty room immediately
            async with ROOMS_LOCK:
                if code_clean in ACTIVE_ROOMS and room.is_empty():
                    del ACTIVE_ROOMS[code_clean]
                    logger.info(f"[Room {code_clean}] Room empty, cleaned up immediately.")


def print_startup_advisory(
    host: str,
    port: int,
    public_url: Optional[str],
    lan_ip: str,
    lan_url: str,
    use_ssl: bool,
    max_rooms: int,
    idle_timeout: float
):
    """Print comprehensive security and networking advisory at server startup."""
    scheme = "https" if use_ssl else "http"
    print("\n" + "=" * 78)
    print(" DEEPFILTERNET LIVE VOICE CALL SERVER — STARTUP & SECURITY ADVISORY")
    print("=" * 78)
    print(f" [i] Local Binding:       {scheme}://{host}:{port}")
    if host in ("0.0.0.0", "::"):
        print(f" [i] LAN Address (Phone): {lan_url}")
        if not use_ssl and not public_url:
            print("\n [!] CRITICAL BROWSER REQUIREMENT FOR PHONES / SECOND DEVICES:")
            print("     Modern mobile browsers (iOS Safari, Android Chrome) STRICTLY REQUIRE HTTPS")
            print("     for microphone capture (getUserMedia). Plain http:// over LAN will be blocked.")
            print("     TO CONNECT YOUR PHONE:")
            print("     1. Self-Signed HTTPS (LAN only):")
            print(f"        python src/call/call_server.py --host 0.0.0.0 --port {port} --ssl")
            print(f"        Open {lan_url.replace('http:', 'https:')} on your phone and tap 'Advanced -> Proceed'.")
            print("     2. Public HTTPS Tunnel (Any phone/network, zero cert warnings):")
            print(f"        cloudflared tunnel --url http://127.0.0.1:{port}")
            print(f"        python src/call/call_server.py --port {port} --public-url https://your-tunnel.trycloudflare.com")
        elif use_ssl:
            print("\n [i] SELF-SIGNED SSL ACTIVE:")
            print(f"     Open on your phone: {lan_url}")
            print("     Accept the self-signed certificate in your mobile browser (Advanced -> Proceed).")
            print("     Microphone access is enabled because connection is HTTPS.")
    else:
        print(" [i] LOCALHOST BINDING:   Bound to loopback (127.0.0.1). Accessible only from this PC.")
        print("     To allow connections from phone on the same Wi-Fi, pass --host 0.0.0.0 (or use a tunnel).")

    if public_url:
        print(f" [i] Public Invite URL:   {public_url}")
        print("     Invite links generated by the UI will use this public address.")

    print(f" [i] Max Active Rooms:    {max_rooms} concurrent rooms")
    print(f" [i] Room Idle Timeout:   {idle_timeout:.0f} seconds")
    print(" [!] AUTHENTICATION NOTICE:")
    print("     This server relies solely on unguessable random room codes (6-char alphanumeric).")
    print("     There is NO user account, password, or cryptographic signature.")
    print("     Anyone with the room link can join until the 2-participant limit is reached.")
    print("=" * 78 + "\n")


def start_cloudflare_tunnel(local_port: int, max_retries: int = 2) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Spawn cloudflared quick tunnel and extract public HTTPS URL with automatic retry."""
    cf_bin = ROOT_DIR / "cloudflared.exe"
    if not cf_bin.exists():
        cf_bin = ROOT_DIR / "tools" / "cloudflared.exe"
    cf_cmd = str(cf_bin) if cf_bin.exists() else "cloudflared"

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / "cloudflared.log"

    cmd = [cf_cmd, "tunnel", "--protocol", "http2", "--url", f"http://127.0.0.1:{local_port}"]

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"[i] Retrying Cloudflare tunnel request (attempt {attempt}/{max_retries})...")
            time.sleep(2.0)

        log_f = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
            url = None
            t0 = time.time()
            while time.time() - t0 < 22:
                time.sleep(0.5)
                if log_path.exists():
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                        for cand in matches:
                            if "api.trycloudflare.com" not in cand and "www.trycloudflare.com" not in cand and cand != "https://trycloudflare.com":
                                url = cand
                                break
                        if url:
                            return proc, url
                        if "failed to request quick Tunnel" in content or "context deadline exceeded" in content:
                            break
                    except Exception:
                        pass
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not start cloudflared tunnel attempt {attempt}: {e}")

    return None, None


def run_call_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    public_url: Optional[str] = None,
    max_rooms: int = 2,
    idle_timeout: float = 300.0,
    use_ssl: bool = False,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
    auto_tunnel: bool = False,
):
    """Launch the Call Server with configured limits, optional SSL, or automated cloud tunnel."""
    tunnel_proc = None
    if auto_tunnel:
        print("\n[i] Launching automated Cloudflare HTTPS Tunnel for phone/remote access...")
        tunnel_proc, tunnel_url = start_cloudflare_tunnel(port)
        if tunnel_url:
            public_url = tunnel_url
            print(f"[i] Tunnel established: {tunnel_url}")
            print(f"[i] Cloudflared live log: outputs/cloudflared.log")
        else:
            print("[!] Warning: Cloudflare tunnel could not establish a quick URL in time.")

        def cleanup_tunnel():
            if tunnel_proc:
                try:
                    tunnel_proc.terminate()
                except Exception:
                    pass
        atexit.register(cleanup_tunnel)

    lan_ip = get_lan_ip()
    scheme = "https" if use_ssl else "http"
    lan_url = f"{scheme}://{lan_ip}:{port}"

    CONFIG["host"] = host
    CONFIG["port"] = port
    CONFIG["public_url"] = public_url.rstrip("/") if public_url else None
    CONFIG["lan_ip"] = lan_ip
    CONFIG["lan_url"] = lan_url
    CONFIG["use_ssl"] = use_ssl
    CONFIG["max_rooms"] = max_rooms
    CONFIG["idle_timeout"] = idle_timeout

    ssl_certfile = None
    ssl_keyfile = None
    if use_ssl:
        cert_p = Path(ssl_cert) if ssl_cert else ROOT_DIR / "certs" / "cert.pem"
        key_p = Path(ssl_key) if ssl_key else ROOT_DIR / "certs" / "key.pem"
        ensure_ssl_certs(lan_ip, cert_p, key_p)
        ssl_certfile = str(cert_p)
        ssl_keyfile = str(key_p)

    print_startup_advisory(
        host=host,
        port=port,
        public_url=CONFIG["public_url"],
        lan_ip=lan_ip,
        lan_url=lan_url,
        use_ssl=use_ssl,
        max_rooms=max_rooms,
        idle_timeout=idle_timeout,
    )

    uvicorn_kwargs = {
        "app": app,
        "host": host,
        "port": port,
        "log_level": "warning",
    }
    if use_ssl and ssl_certfile and ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    try:
        uvicorn.run(**uvicorn_kwargs)
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFilterNet Live Voice Call Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address (default: 0.0.0.0 for LAN & phone access)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--public-url", default=None, help="Public URL (e.g. https://call.example.com) for invite links")
    parser.add_argument("--tunnel", action="store_true", help="Automatically launch Cloudflare Tunnel so phones can connect via HTTPS with 0 config")
    parser.add_argument("--max-rooms", type=int, default=2, help="Max active concurrent rooms (default: 2)")
    parser.add_argument("--idle-timeout", type=float, default=300.0, help="Idle timeout in seconds before unpopulated/idle room is reaped (default: 300s)")
    parser.add_argument("--ssl", action="store_true", help="Enable self-signed HTTPS/WSS (required for mobile browser mic access on LAN)")
    parser.add_argument("--ssl-cert", default=None, help="Path to SSL certificate (.pem)")
    parser.add_argument("--ssl-key", default=None, help="Path to SSL private key (.pem)")
    args = parser.parse_args()

    run_call_server(
        host=args.host,
        port=args.port,
        public_url=args.public_url,
        max_rooms=args.max_rooms,
        idle_timeout=args.idle_timeout,
        use_ssl=args.ssl,
        ssl_cert=args.ssl_cert,
        ssl_key=args.ssl_key,
        auto_tunnel=args.tunnel,
    )
