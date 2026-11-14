"""SQLite storage layer.

One database file is shared by every API process (WAL mode, busy timeout).
All writes happen in short IMMEDIATE transactions, so background comparison
never blocks corpus submissions and vice versa.

Tables
------
meta                 single row: current_generation / published_generation
articles             committed corpus (id, side, text, updated_generation)
edges / article_groups  published verification results, keyed by generation
compute_jobs         pending/running background verification jobs (lease based)
"""

import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_generation INTEGER NOT NULL,
    published_generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    side TEXT NOT NULL CHECK (side IN ('train', 'test')),
    text TEXT NOT NULL CHECK (length(text) BETWEEN 5 AND 2000),
    updated_generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    generation INTEGER NOT NULL,
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (generation, a, b)
);

CREATE TABLE IF NOT EXISTS article_groups (
    generation INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    article_id TEXT NOT NULL,
    side TEXT NOT NULL,
    PRIMARY KEY (generation, group_id, article_id)
);

CREATE TABLE IF NOT EXISTS compute_jobs (
    generation INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running')),
    claimed_by TEXT,
    claimed_at REAL
);
"""

# Hard corpus limits (also enforced at the API boundary).
MAX_ARTICLES = 300
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 2000


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection in autocommit mode; transactions are explicit."""
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(db_path: str) -> None:
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta (id, current_generation, published_generation)"
            " VALUES (1, 0, 0)"
        )
        conn.commit()
    finally:
        conn.close()


def get_meta(conn: sqlite3.Connection):
    """Return ``(current_generation, published_generation)``."""
    row = conn.execute(
        "SELECT current_generation, published_generation FROM meta WHERE id = 1"
    ).fetchone()
    return row["current_generation"], row["published_generation"]
