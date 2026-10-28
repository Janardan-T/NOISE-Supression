"""
SQLite Database Management for NoiseShield Live Voice Call Studio
==================================================================
Handles user identity, registration, online presence, and call records.
Uses Python's built-in sqlite3 database (100% free, zero external configuration).
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "noiseshield.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema if not already present."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1: Registered Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        status TEXT DEFAULT 'offline',
        last_seen TEXT
    )
    """)

    # Table 2: Call Records
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS call_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT NOT NULL,
        caller_email TEXT NOT NULL,
        callee_email TEXT NOT NULL,
        status TEXT NOT NULL,
        duration_sec INTEGER DEFAULT 0,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

    # Register default presets for instant testing out-of-the-box
    register_or_update_user("alice@gmail.com", "Alice Student", status="offline")
    register_or_update_user("bob@gmail.com", "Bob Student", status="offline")


def register_or_update_user(email: str, display_name: str, status: str = "online") -> dict:
    """Register a new user or update their display name and set presence."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    clean_email = email.strip().lower()
    clean_name = display_name.strip()

    cursor.execute("""
    INSERT INTO users (email, display_name, status, last_seen)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(email) DO UPDATE SET
        display_name = excluded.display_name,
        status = excluded.status,
        last_seen = excluded.last_seen
    """, (clean_email, clean_name, status, now))

    conn.commit()
    conn.close()
    return get_user(clean_email)


def set_user_status(email: str, status: str):
    """Update user online presence ('online' or 'offline')."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE users SET status = ?, last_seen = ? WHERE email = ?",
        (status, now, email.strip().lower())
    )
    conn.commit()
    conn.close()


def get_user(email: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, display_name, status, last_seen FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_all_users() -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, display_name, status, last_seen FROM users ORDER BY status DESC, display_name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_call_start(call_id: str, caller_email: str, callee_email: str):
    """Record start of a call. Initial status is 'missed' until answered."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("SELECT id FROM call_logs WHERE call_id = ?", (call_id,))
    if not cursor.fetchone():
        cursor.execute("""
        INSERT INTO call_logs (call_id, caller_email, callee_email, status, duration_sec, created_at)
        VALUES (?, ?, ?, 'missed', 0, ?)
        """, (call_id, caller_email.strip().lower(), callee_email.strip().lower(), now))
        conn.commit()
    conn.close()


def update_call_status(call_id: str, status: str, duration_sec: int = 0):
    """Update call status to 'connected' or 'missed' with optional duration."""
    if not call_id:
        return
    conn = get_connection()
    cursor = conn.cursor()
    if duration_sec > 0:
        cursor.execute("""
        UPDATE call_logs SET status = ?, duration_sec = ? WHERE call_id = ?
        """, (status, duration_sec, call_id))
    else:
        cursor.execute("""
        UPDATE call_logs SET status = ? WHERE call_id = ?
        """, (status, call_id))
    conn.commit()
    conn.close()


def record_room_call(room_code: str, host_email: str, guest_email: str, duration_sec: int = 0):
    """Record a 6-digit room call as connected."""
    call_id = f"room_{room_code}"
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("SELECT id FROM call_logs WHERE call_id = ?", (call_id,))
    if cursor.fetchone():
        cursor.execute("UPDATE call_logs SET status = 'connected', duration_sec = ? WHERE call_id = ?", (duration_sec, call_id))
    else:
        cursor.execute("""
        INSERT INTO call_logs (call_id, caller_email, callee_email, status, duration_sec, created_at)
        VALUES (?, ?, ?, 'connected', ?, ?)
        """, (call_id, host_email.strip().lower(), guest_email.strip().lower(), duration_sec, now))
    conn.commit()
    conn.close()


def get_user_call_logs(email: str, limit: int = 30) -> List[dict]:
    """Retrieve recent call history for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    clean_email = email.strip().lower()
    cursor.execute("""
    SELECT id, call_id, caller_email, callee_email, status, duration_sec, created_at
    FROM call_logs
    WHERE caller_email = ? OR callee_email = ?
    ORDER BY id DESC
    LIMIT ?
    """, (clean_email, clean_email, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_call_log(call_id: str, email: str) -> bool:
    """Delete a specific call record."""
    conn = get_connection()
    cursor = conn.cursor()
    clean_email = email.strip().lower()
    cursor.execute("""
    DELETE FROM call_logs 
    WHERE call_id = ? AND (caller_email = ? OR callee_email = ?)
    """, (call_id, clean_email, clean_email))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_user_call_logs(email: str) -> bool:
    """Clear all call history records for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    clean_email = email.strip().lower()
    cursor.execute("""
    DELETE FROM call_logs 
    WHERE caller_email = ? OR callee_email = ?
    """, (clean_email, clean_email))
    conn.commit()
    conn.close()
    return True
