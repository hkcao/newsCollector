"""SQLite 历史存储 —— 用于计算 delta_24h。

设计：
- items:           记录每条信息的元数据 + 首次见到时间
- signal_history:  每次抓取时给已命中的 item 打一个信号快照
- delta_24h = 当前 signals.points - 24h 内最早一次快照的 points
  - 若 item 在 24h 内首次出现：delta=0（用 recency_boost 兜底）
  - 若有 24h 前的快照：用最接近 24h 前的点作 baseline
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    source TEXT,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_history (
    item_id TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    PRIMARY KEY (item_id, snapshot_at)
);
CREATE INDEX IF NOT EXISTS idx_sh_item_time
    ON signal_history(item_id, snapshot_at);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class DB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def baseline_signals(self, item_id: str) -> dict | None:
        """
        返回 delta 计算的 baseline，按优先级：
          1) 24h 之前最近的一次快照（>24h 历史存在）
          2) 24h 内最早的一次快照（item 在 24h 内首次见到）
        没有任何历史 → None
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        row = self.conn.execute(
            "SELECT signals_json FROM signal_history "
            "WHERE item_id=? AND snapshot_at<=? "
            "ORDER BY snapshot_at DESC LIMIT 1",
            (item_id, cutoff),
        ).fetchone()
        if row:
            return json.loads(row[0])

        row = self.conn.execute(
            "SELECT signals_json FROM signal_history "
            "WHERE item_id=? ORDER BY snapshot_at ASC LIMIT 1",
            (item_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def upsert(self, item: dict, now_iso: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO items(id,title,url,source,first_seen) "
            "VALUES (?,?,?,?,?)",
            (item["id"], item["title"], item["url"], item["source"], now_iso),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO signal_history(item_id,snapshot_at,signals_json) "
            "VALUES (?,?,?)",
            (item["id"], now_iso, json.dumps(item.get("signals", {}))),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- meta key/value ----
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()
