import sqlite3
from config import DB_PATH

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

def save_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO chat_messages(session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content))

def get_history(session_id: str, limit: int = 30):
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]

def clear_history(session_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
