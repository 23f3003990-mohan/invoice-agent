"""SQLite persistence: tasks, idempotency records, and the package-decision cache."""
import json
import os
import sqlite3
import threading
import time

DB_PATH = os.getenv("DB_PATH", "/data/agent.db")
_lock = threading.Lock()


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=FULL")
    return c


_db = _conn()
_db.executescript(
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, principal TEXT NOT NULL, body TEXT NOT NULL,
        private TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS tasks_principal ON tasks(principal, updated);
    CREATE TABLE IF NOT EXISTS messages (
        principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL,
        task_id TEXT NOT NULL, PRIMARY KEY (principal, message_id));
    CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, body TEXT NOT NULL);
    """
)


def get_task(principal, task_id):
    with _lock:
        row = _db.execute(
            "SELECT body, private FROM tasks WHERE id=? AND principal=?", (task_id, principal)
        ).fetchone()
    return (json.loads(row[0]), json.loads(row[1])) if row else (None, None)


def list_tasks(principal):
    with _lock:
        rows = _db.execute(
            "SELECT body FROM tasks WHERE principal=? ORDER BY updated DESC", (principal,)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def save_task(principal, task, private, message_record=None):
    """Atomically write the task (and optionally the idempotency record) before responding."""
    now = time.time()
    with _lock:
        _db.execute("BEGIN IMMEDIATE")
        try:
            _db.execute(
                "INSERT INTO tasks(id, principal, body, private, created, updated) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET body=excluded.body, private=excluded.private, updated=excluded.updated",
                (task["id"], principal, json.dumps(task), json.dumps(private), now, now),
            )
            if message_record:
                _db.execute(
                    "INSERT OR IGNORE INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
                    (principal, *message_record),
                )
            _db.execute("COMMIT")
        except Exception:
            _db.execute("ROLLBACK")
            raise


def get_message(principal, message_id):
    with _lock:
        row = _db.execute(
            "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?", (principal, message_id)
        ).fetchone()
    return row


def get_decision(key):
    with _lock:
        row = _db.execute("SELECT body FROM decisions WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put_decision(key, body):
    with _lock:
        _db.execute("INSERT OR REPLACE INTO decisions(key, body) VALUES(?,?)", (key, json.dumps(body)))
