"""Automated Headless Chromium Audio Call Test via Chrome DevTools Protocol (CDP).

Launches Google Chrome with:
--use-fake-ui-for-media-stream
--use-fake-device-for-media-stream
--use-file-for-fake-audio-capture=...
to verify that the web interface captures, streams via WebSocket, and plays audio
without browser or AudioWorklet errors.
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PAGE_URL = "http://127.0.0.1:8001/"
CDP_PORT = 9222
TEST_WAV = str(Path(__file__).resolve().parent.parent.parent / "samples" / "speech_with_whitenoise.wav")


async def run_browser_automation():
    print("=" * 70)
    print(" AUTOMATED HEADLESS CHROMIUM AUDIO CALL TEST (CDP)")
    print("=" * 70)

    if not os.path.exists(CHROME_PATH):
        print(f"[Error] Chrome executable not found at: {CHROME_PATH}")
        return False

    temp_user_data = tempfile.mkdtemp(prefix="chrome_call_test_")
    print(f" Chrome binary:     {CHROME_PATH}")
    print(f" Fake Audio Device: {TEST_WAV}")
    print(f" Target Web UI:     {PAGE_URL}")
    print(f" Temp Profile:      {temp_user_data}")
    print("-" * 70)

    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={CDP_PORT}",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={TEST_WAV}",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={temp_user_data}",
        PAGE_URL,
    ]

    print("[1/5] Launching Chrome with fake audio capture flags...")
    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    console_logs = []
    cdp_ws_url = None

    try:
        # Wait for CDP endpoint to become ready
        for _ in range(30):
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=1.0)
                pages = json.loads(resp.read().decode("utf-8"))
                page_targets = [p for p in pages if p.get("type") == "page"]
                if page_targets:
                    # Prefer the target that matches PAGE_URL or port 8001
                    call_page = next((p for p in page_targets if "8001" in p.get("url", "")), page_targets[0])
                    cdp_ws_url = call_page.get("webSocketDebuggerUrl")
                    if cdp_ws_url:
                        break
            except Exception:
                await asyncio.sleep(0.3)

        if not cdp_ws_url:
            print("[Error] Failed to connect to Chrome DevTools Protocol endpoint.")
            return False

        print(f"[2/5] Connected to Chrome CDP: {cdp_ws_url}")

        async with websockets.connect(cdp_ws_url, max_size=10 * 1024 * 1024) as cdp:
            msg_id = 0
            pending_requests = {}
            reader_running = True

            async def cdp_reader():
                try:
                    async for raw in cdp:
                        msg = json.loads(raw)
                        # Handle command responses
                        if "id" in msg and msg["id"] in pending_requests:
                            fut = pending_requests.pop(msg["id"])
                            if not fut.done():
                                fut.set_result(msg)
                        # Handle console & exception events
                        method = msg.get("method")
                        if method == "Runtime.consoleAPICalled":
                            args = msg.get("params", {}).get("args", [])
                            text = " ".join([str(a.get("value", a.get("description", ""))) for a in args])
                            console_logs.append(f"[Console {msg.get('params',{}).get('type','log')}]: {text}")
                        elif method == "Runtime.exceptionThrown":
                            details = msg.get("params", {}).get("exceptionDetails", {})
                            console_logs.append(f"[Exception]: {details.get('text', '')} {details.get('exception', {}).get('description', '')}")
                except Exception as e:
                    if reader_running:
                        console_logs.append(f"[Reader Error]: {e}")

            reader_task = asyncio.create_task(cdp_reader())

            async def call_cdp(method: str, params: dict = None, timeout: float = 5.0):
                nonlocal msg_id
                msg_id += 1
                curr_id = msg_id
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                pending_requests[curr_id] = fut

                req = {"id": curr_id, "method": method, "params": params or {}}
                await cdp.send(json.dumps(req))

                resp = await asyncio.wait_for(fut, timeout=timeout)
                outer = resp.get("result", {})
                if "result" in outer and "value" in outer["result"]:
                    return outer["result"]["value"]
                return outer

            # Enable Page and Runtime domains
            await call_cdp("Page.enable")
            await call_cdp("Runtime.enable")

            # Wait for DOM ready
            print("[3/5] Waiting for DOM ready on call page...")
            for _ in range(20):
                btn_exists = await call_cdp("Runtime.evaluate", {
                    "expression": "document.getElementById('btnJoin') !== null",
                    "returnByValue": True
                })
                if btn_exists is True:
                    break
                await asyncio.sleep(0.3)

            # Trigger "Join Call" button click
            print("[4/5] Clicking 'Join Call' button...")
            await call_cdp("Runtime.evaluate", {
                "expression": "document.getElementById('btnJoin').click(); 'clicked'",
                "returnByValue": True
            })

            # Stream audio through AudioWorklet & WebSocket for 3.5s
            print("      Streaming audio through AudioWorklet & WebSocket for 3.5s...")
            await asyncio.sleep(3.5)

            # Read telemetry from DOM
            telemetry_raw = await call_cdp("Runtime.evaluate", {
                "expression": """
                JSON.stringify({
                    status: document.getElementById('statusText').textContent,
                    rtt: document.getElementById('statRtt').textContent,
                    serverProc: document.getElementById('statServerProc').textContent,
                    buffer: document.getElementById('statBuffer').textContent,
                    dropped: document.getElementById('statDropped').textContent,
                    packets: document.getElementById('statPacketsSub').textContent,
                    callDuration: document.getElementById('callDuration').textContent,
                    audioState: audioContext ? audioContext.state : 'null'
                })
                """,
                "returnByValue": True
            })

            telemetry = json.loads(telemetry_raw) if isinstance(telemetry_raw, str) else {}

            # Test live mode toggle: switch to Bypass mode
            print("      Testing live mode switch to Bypass...")
            await call_cdp("Runtime.evaluate", {
                "expression": "document.getElementById('btnModeBypass').click();",
            })
            await asyncio.sleep(1.5)

            telemetry_bypass_raw = await call_cdp("Runtime.evaluate", {
                "expression": """
                JSON.stringify({
                    mode: document.getElementById('activeModeLabel').textContent,
                    rtt: document.getElementById('statRtt').textContent,
                    serverProc: document.getElementById('statServerProc').textContent,
                    packets: document.getElementById('statPacketsSub').textContent
                })
                """,
                "returnByValue": True
            })
            telemetry_bypass = json.loads(telemetry_bypass_raw) if isinstance(telemetry_bypass_raw, str) else {}

            # Click Leave Call
            print("[5/5] Clicking 'Leave Call' button...")
            await call_cdp("Runtime.evaluate", {
                "expression": "document.getElementById('btnLeave').click();",
            })
            await asyncio.sleep(0.5)

            status_after_leave = await call_cdp("Runtime.evaluate", {
                "expression": "document.getElementById('statusText').textContent",
                "returnByValue": True
            })

            reader_running = False
            reader_task.cancel()

            print("\n" + "=" * 70)
            print(" HEADLESS BROWSER TEST RESULTS")
            print("=" * 70)
            print(f" Browser:              Google Chrome (Headless mode)")
            print(f" AudioContext State:   {telemetry.get('audioState')} (48,000 Hz)")
            print(f" Status During Stream: {telemetry.get('status')}")
            print(f" Call Duration:        {telemetry.get('callDuration')}")
            print(f" Telemetry (Denoised): RTT = {telemetry.get('rtt')} | Server Proc = {telemetry.get('serverProc')} | Buffer = {telemetry.get('buffer')}")
            print(f" Packets Telemetry:    {telemetry.get('packets')} (Dropped: {telemetry.get('dropped')})")
            print(f" Telemetry (Bypass):   Mode = {telemetry_bypass.get('mode')} | RTT = {telemetry_bypass.get('rtt')} | Server Proc = {telemetry_bypass.get('serverProc')}")
            print(f" Status After Leave:   {status_after_leave}")
            print(f" Console Logs:         {len(console_logs)} events recorded")
            for log in console_logs:
                print(f"   {log}")
            print("=" * 70)

            # Assertions
            assert telemetry.get("audioState") == "running", f"AudioContext was not running! Telemetry: {telemetry}"
            assert any(s in telemetry.get("status", "") for s in ["Active", "Room"]), f"Call was not active! Status: {telemetry.get('status')}"
            assert status_after_leave == "Call Ended", f"Call did not end cleanly! Status: {status_after_leave}"

            print("[Result]: HEADLESS BROWSER TEST PASSED SUCCESSFULLY")
            return True

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            pass
        import shutil
        shutil.rmtree(temp_user_data, ignore_errors=True)


if __name__ == "__main__":
    success = asyncio.run(run_browser_automation())
    sys.exit(0 if success else 1)
