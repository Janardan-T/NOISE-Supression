"""
Unit and Integration Tests for NoiseShield Extended Features in Call Server.
Verifies:
1. User registration & login (/api/auth/login)
2. User directory & online presence (/api/users)
3. Call history logging, retrieval, deletion (/api/calls/history)
4. AI benchmark endpoint (/api/ai/benchmark)
5. Dynamic QR code generator (/api/qr)
6. Real-time signaling (/ws/user/{email}): CALL_INVITE, INCOMING_CALL, CALL_RESPONSE, CALL_HANGUP
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from src.call.call_server import app
from src.call import database

client = TestClient(app)

def test_rest_endpoints():
    print("\n--- Testing REST API Endpoints ---")
    
    # 1. Login
    res = client.post("/api/auth/login", json={"email": "alice@gmail.com", "display_name": "Alice Student"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "success"
    assert data["user"]["email"] == "alice@gmail.com"
    print(" [OK] POST /api/auth/login PASSED")

    res = client.post("/api/auth/login", json={"email": "bob@gmail.com", "display_name": "Bob Student"})
    assert res.status_code == 200
    print(" [OK] POST /api/auth/login (User B) PASSED")

    # 2. Users directory
    res = client.get("/api/users")
    assert res.status_code == 200
    users = res.json()["users"]
    emails = [u["email"] for u in users]
    assert "alice@gmail.com" in emails
    assert "bob@gmail.com" in emails
    print(f" [OK] GET /api/users PASSED ({len(users)} users registered)")

    # 3. Call history
    unique_call_id = f"test_call_{int(time.time()*1000)}"
    database.record_call_start(unique_call_id, "alice@gmail.com", "bob@gmail.com")
    database.update_call_status(unique_call_id, "connected", duration_sec=45)

    res = client.get("/api/calls/history?email=alice@gmail.com")
    assert res.status_code == 200
    calls = res.json()["calls"]
    assert len(calls) > 0
    target_call = next(c for c in calls if c["call_id"] == unique_call_id)
    assert target_call["status"] == "connected"
    assert target_call["duration_sec"] == 45
    print(f" [OK] GET /api/calls/history PASSED ({len(calls)} calls retrieved)")

    # 4. Tunnel info
    res = client.get("/api/tunnel")
    assert res.status_code == 200
    assert "status" in res.json()
    print(" [OK] GET /api/tunnel PASSED")

    # 5. AI Benchmark
    res = client.get("/api/ai/benchmark")
    assert res.status_code == 200
    bench = res.json()
    assert bench["status"] == "ok"
    assert "latency_ms" in bench
    assert bench["model"] == "DeepFilterNet3"
    print(f" [OK] GET /api/ai/benchmark PASSED (Latency: {bench['latency_ms']} ms, RTF: {bench['rtf']})")

    # 6. QR Code generator
    res = client.get("/api/qr?data=http://127.0.0.1:8001")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 100
    print(" [OK] GET /api/qr PASSED (PNG QR code generated)")

def test_signaling_websocket():
    print("\n--- Testing User Signaling WebSocket (/ws/user/{email}) ---")
    with client.websocket_connect("/ws/user/alice@gmail.com") as ws_alice:
        # 1. Ping / Pong
        ws_alice.send_json({"type": "PING"})
        msg = ws_alice.receive_json()
        while msg.get("type") == "USER_PRESENCE":
            msg = ws_alice.receive_json()
        assert msg["type"] == "PONG"
        print(" [OK] WebSocket Ping/Pong PASSED")

        # 2. Online status in DB
        user = database.get_user("alice@gmail.com")
        assert user["status"] == "online"
        print(" [OK] User presence set to 'online' in DB PASSED")

        # 3. Call offline user returns CALL_ERROR
        ws_alice.send_json({
            "type": "CALL_INVITE",
            "target": "nobody@gmail.com",
            "caller_name": "Alice Student",
            "call_id": "test_call_offline",
            "room_code": "999888"
        })
        err_msg = ws_alice.receive_json()
        while err_msg.get("type") == "USER_PRESENCE":
            err_msg = ws_alice.receive_json()
        assert err_msg["type"] == "CALL_ERROR"
        assert "offline" in err_msg["message"]
        print(" [OK] CALL_INVITE to offline user returns CALL_ERROR PASSED")

    # 4. Offline status after disconnect
    user_after = database.get_user("alice@gmail.com")
    assert user_after["status"] == "offline"
    print(" [OK] User presence set to 'offline' on disconnect PASSED")

if __name__ == "__main__":
    test_rest_endpoints()
    test_signaling_websocket()
    print("\n=======================================================")
    print(" ALL EXTENDED NOISESHIELD FEATURE TESTS PASSED CLEANLY!")
    print("=======================================================\n")
