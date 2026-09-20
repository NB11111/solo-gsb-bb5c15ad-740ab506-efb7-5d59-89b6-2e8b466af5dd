"""SQLite 持久化与数据库初始化。

plan_id 由规范化请求 JSON 的 SHA256 派生（与时间无关），相同输入天然幂等：
重复提交会命中已存在的方案记录并原样返回，保证“相同输入得到稳定结果”。

切割执行会话（cut_sessions / cut_session_events / cut_session_batches /
cut_session_blade_changes）：
- cut_sessions 记录一次方案执行的进度：version 为乐观锁版本，每完成一刀或
  确认一次换刀 +1；blade_life 为启动时设置的刀片寿命（NULL 表示不检查），
  blade_used 为当前刀片自上次换刀以来的累计切长；
- cut_session_events 以 (session_id, idempotency_key) 为主键记录已应用事件，
  同一键重试直接重放首次响应，同键不同请求视为冲突；version 记录事件应用后
  的会话版本（换刀也占版本号，批次对账据此在版本流中定位）；
- cut_session_batches 以 (session_id, batch_id) 为主键记录已应用批次，
  供设备离线后批量回传：对账一致的前缀仅跳过，其余后缀连续匹配当前方案且
  不越过刀具寿命检查点后一次性推进；同键重试重放首次响应，同键不同请求视为
  冲突；
- cut_session_blade_changes 以 (session_id, idempotency_key) 为主键记录已
  应用的换刀确认；
- 事件/批次/换刀推进走 transaction()（BEGIN IMMEDIATE）：校验与状态推进在
  同一事务中原子完成，并发写者由 SQLite 写锁串行化，旧版本/跳步请求不会推进
  状态，并发换刀与切刀只有一方成功。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get(
    "NESTING_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "nesting.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id        TEXT PRIMARY KEY,
    request_json   TEXT NOT NULL,
    response_json  TEXT NOT NULL,
    status         TEXT NOT NULL,
    panels_used    INTEGER NOT NULL,
    placed_parts   INTEGER NOT NULL,
    unplaced_parts INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_created_at ON plans(created_at DESC);

-- 切割执行会话：一行 = 一个方案的一次执行实例
CREATE TABLE IF NOT EXISTS cut_sessions (
    session_id     TEXT PRIMARY KEY,
    plan_id        TEXT NOT NULL REFERENCES plans(plan_id),
    status         TEXT NOT NULL,              -- active / completed
    version        INTEGER NOT NULL,           -- 乐观锁版本：每完成一刀/换刀 +1
    completed_cuts INTEGER NOT NULL,           -- 已完成刀数
    total_cuts     INTEGER NOT NULL,           -- 启动时快照的总刀数
    blade_life     REAL,                       -- 刀片寿命（mm）；NULL=不检查
    blade_used     REAL NOT NULL DEFAULT 0,    -- 当前刀片累计切长（mm）
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    completed_at   TEXT                        -- 全部完成的时间；未完成为 NULL
);
CREATE INDEX IF NOT EXISTS idx_cut_sessions_plan ON cut_sessions(plan_id);

-- 切割事件幂等记录：同一 (session_id, idempotency_key) 只应用一次
CREATE TABLE IF NOT EXISTS cut_session_events (
    session_id      TEXT NOT NULL REFERENCES cut_sessions(session_id),
    idempotency_key TEXT NOT NULL,
    panel_no        INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    event_version   INTEGER NOT NULL,          -- 事件应用后的会话版本
    request_json    TEXT NOT NULL,             -- 规范化请求（识别同键不同请求）
    response_json   TEXT NOT NULL,             -- 首次应用的响应，重试原样返回
    created_at      TEXT NOT NULL,
    PRIMARY KEY (session_id, idempotency_key)
);

-- 换刀确认幂等记录：同一 (session_id, idempotency_key) 只应用一次
CREATE TABLE IF NOT EXISTS cut_session_blade_changes (
    session_id      TEXT NOT NULL REFERENCES cut_sessions(session_id),
    idempotency_key TEXT NOT NULL,
    from_version    INTEGER NOT NULL,          -- 换刀前的会话版本
    request_json    TEXT NOT NULL,
    response_json   TEXT NOT NULL,             -- 首次应用的响应，重试原样返回
    created_at      TEXT NOT NULL,
    PRIMARY KEY (session_id, idempotency_key)
);

-- 会话版本流：每个版本号一行（切刀事件一刀一行；批次应用的刀各一行；
-- 换刀确认一行 kind='blade'）。批次对账据此把任意 base_version 翻译成
-- 该版本时的已完成刀数（换刀版本不计刀）。
CREATE TABLE IF NOT EXISTS cut_session_ops (
    session_id  TEXT NOT NULL REFERENCES cut_sessions(session_id),
    version     INTEGER NOT NULL,              -- 该操作应用后的会话版本
    kind        TEXT NOT NULL,                 -- event / batch / blade
    panel_no    INTEGER,                       -- kind=event|batch 时所在母板
    step        INTEGER,                       -- kind=event|batch 时刀序号
    PRIMARY KEY (session_id, version)
);

-- 切刀批次幂等记录：同一 (session_id, batch_id) 只应用一次
CREATE TABLE IF NOT EXISTS cut_session_batches (
    session_id   TEXT NOT NULL REFERENCES cut_sessions(session_id),
    batch_id     TEXT NOT NULL,
    base_version INTEGER NOT NULL,           -- 设备离线前看到的会话版本
    cut_count    INTEGER NOT NULL,           -- 批次内刀数
    skipped      INTEGER NOT NULL,           -- 对账一致而跳过的刀数
    applied      INTEGER NOT NULL,           -- 实际推进的刀数
    request_json  TEXT NOT NULL,             -- 规范化请求（识别同键不同请求）
    response_json TEXT NOT NULL,             -- 首次应用的响应，重试原样返回
    created_at   TEXT NOT NULL,
    PRIMARY KEY (session_id, batch_id)
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate(conn: sqlite3.Connection) -> None:
    """对旧版数据库做幂等结构升级（建表后调用）。

    刀具寿命特性新增：cut_sessions.blade_life / blade_used、
    cut_session_events.event_version；旧库逐列 ALTER 补齐，并按事件应用顺序
    回填 event_version（旧版本流中版本与刀数一一相等）。
    """
    cols = _table_columns(conn, "cut_sessions")
    if "blade_life" not in cols:
        conn.execute("ALTER TABLE cut_sessions ADD COLUMN blade_life REAL")
    if "blade_used" not in cols:
        conn.execute(
            "ALTER TABLE cut_sessions ADD COLUMN blade_used REAL NOT NULL "
            "DEFAULT 0")
    ev_cols = _table_columns(conn, "cut_session_events")
    if "event_version" not in ev_cols:
        conn.execute(
            "ALTER TABLE cut_session_events ADD COLUMN event_version INTEGER")
        # 旧库中没有换刀版本节点，事件按应用顺序编号即其应用后的版本
        conn.execute(
            """
            UPDATE cut_session_events
               SET event_version = (
                     SELECT COUNT(*) FROM cut_session_events e2
                      WHERE e2.session_id = cut_session_events.session_id
                        AND (e2.created_at < cut_session_events.created_at
                             OR (e2.created_at = cut_session_events.created_at
                                 AND e2.idempotency_key
                                     <= cut_session_events.idempotency_key)))
             WHERE event_version IS NULL
            """)
    # 旧库版本流由事件 + 批次应用刀构成（无换刀节点）：从幂等记录重建 ops。
    # 逐会话处理：该会话尚无任何 ops 节点时才回填，幂等且不重复。
    event_sessions = conn.execute(
        "SELECT DISTINCT session_id FROM cut_session_events").fetchall()
    batch_sessions = conn.execute(
        "SELECT DISTINCT session_id FROM cut_session_batches").fetchall()
    all_sessions = {r[0] for r in event_sessions} | \
        {r[0] for r in batch_sessions}
    for sid in all_sessions:
        n_ops = conn.execute(
            "SELECT COUNT(*) AS c FROM cut_session_ops WHERE session_id = ?",
            (sid,)).fetchone()[0]
        if n_ops > 0:
            continue
        conn.execute(
            """
            INSERT INTO cut_session_ops (session_id, version, kind,
                                         panel_no, step)
            SELECT session_id, event_version, 'event', panel_no, step
              FROM cut_session_events
             WHERE session_id = ?
            """,
            (sid,),
        )
        # 批次记录只存汇总：应用刀占用的版本为
        # (base_version + skipped, base_version + skipped + applied]；
        # 与事件版本撞号（INSERT OR IGNORE 跳过）只可能出现在历史数据已
        # 损坏的情形，正常旧库中两者恰好首尾相接。
        rows = conn.execute(
            """
            SELECT base_version, skipped, applied
              FROM cut_session_batches
             WHERE session_id = ? AND applied > 0
            """,
            (sid,),
        ).fetchall()
        for base_version, skipped, applied in rows:
            start = base_version + skipped
            conn.executemany(
                "INSERT OR IGNORE INTO cut_session_ops "
                "(session_id, version, kind, panel_no, step) "
                "VALUES (?, ?, 'batch', NULL, NULL)",
                [(sid, start + j) for j in range(1, applied + 1)],
            )


def init_db(path: Optional[str] = None) -> None:
    """创建数据目录、连接并初始化表结构；可重复执行（含旧库结构升级）。"""
    target = path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with sqlite3.connect(target) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.commit()


@contextmanager
def get_conn(path: Optional[str] = None):
    target = path or DB_PATH
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(path: Optional[str] = None):
    """写事务（BEGIN IMMEDIATE … COMMIT），用于“校验 + 推进”原子操作。

    进入即取得 SQLite 写锁：并发写者在库级串行执行，后到者读到的是前一
    事务提交后的最新状态；异常自动回滚。连接使用自动提交模式，busy 超时
    放宽到 30s 以容纳突发并发写。
    """
    target = path or DB_PATH
    conn = sqlite3.connect(target, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK;")
            raise
        else:
            conn.execute("COMMIT;")
    finally:
        conn.close()


def canonical_request(req_dict: dict) -> str:
    """规范化序列化：键排序、空格固定，使同一语义输入得到同一字符串。"""
    return json.dumps(req_dict, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def plan_id_from_request(req_dict: dict) -> str:
    digest = hashlib.sha256(canonical_request(req_dict).encode("utf-8")).hexdigest()
    # 以 UUID 形式呈现（确定性 UUID，仅借用其格式）
    return str(uuid.UUID(digest[:32]))


def save_plan(plan_id: str, request_json: str, response_json: str,
              created_at: str, conn: sqlite3.Connection) -> None:
    resp = json.loads(response_json)
    summary = resp["summary"]
    conn.execute(
        """
        INSERT OR IGNORE INTO plans
          (plan_id, request_json, response_json, status, panels_used,
           placed_parts, unplaced_parts, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (plan_id, request_json, response_json, resp["status"],
         summary["panels_used"], summary["placed_parts"],
         summary["unplaced_parts"], created_at),
    )
    conn.commit()


def get_plan(plan_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT response_json FROM plans WHERE plan_id = ?", (plan_id,)
    ).fetchone()
    return json.loads(row["response_json"]) if row else None


def list_plans(limit: int, offset: int, conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT plan_id, status, created_at, panels_used, placed_parts,
               unplaced_parts
          FROM plans
         ORDER BY created_at DESC, plan_id
         LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def count_plans(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM plans").fetchone()["c"]


# ---------------------------------------------------------------------------
# 切割执行会话
#
# 写函数不自行提交：由调用方决定事务边界（get_conn 场景下显式 conn.commit()，
# 或交由 transaction() 统一提交），保证“校验 + 推进 + 事件记录”可落在同一事务。
# ---------------------------------------------------------------------------

def insert_session(session_id: str, plan_id: str, status: str, version: int,
                   completed_cuts: int, total_cuts: int, created_at: str,
                   updated_at: str, completed_at: Optional[str],
                   conn: sqlite3.Connection,
                   blade_life: Optional[float] = None,
                   blade_used: float = 0.0) -> None:
    conn.execute(
        """
        INSERT INTO cut_sessions
          (session_id, plan_id, status, version, completed_cuts, total_cuts,
           blade_life, blade_used, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, plan_id, status, version, completed_cuts, total_cuts,
         blade_life, blade_used, created_at, updated_at, completed_at),
    )


def get_session(session_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    """读取会话并 JOIN 方案响应 JSON（键 plan_json）；不存在返回 None。"""
    row = conn.execute(
        """
        SELECT s.session_id, s.plan_id, s.status, s.version,
               s.completed_cuts, s.total_cuts,
               s.blade_life, s.blade_used,
               s.created_at, s.updated_at, s.completed_at,
               p.response_json AS plan_json
          FROM cut_sessions s
          JOIN plans p ON p.plan_id = s.plan_id
         WHERE s.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def update_session_progress(session_id: str, expected_version: int,
                            version: int, completed_cuts: int,
                            status: str, updated_at: str,
                            completed_at: Optional[str],
                            conn: sqlite3.Connection,
                            blade_used: Optional[float] = None) -> int:
    """推进版本与进度（乐观条件更新）。

    WHERE 带 version=expected_version：与并发换刀/事件/批次竞争时，只有一个
    事务能把行更新成功（rowcount=1）；败者 rowcount=0，调用方据此抛
    version_conflict 并回滚。blade_used 为 None 时不改动累计切长。
    """
    if blade_used is None:
        cur = conn.execute(
            """
            UPDATE cut_sessions
               SET version = ?, completed_cuts = ?, status = ?,
                   updated_at = ?, completed_at = ?
             WHERE session_id = ? AND version = ?
            """,
            (version, completed_cuts, status, updated_at, completed_at,
             session_id, expected_version),
        )
    else:
        cur = conn.execute(
            """
            UPDATE cut_sessions
               SET version = ?, completed_cuts = ?, status = ?, blade_used = ?,
                   updated_at = ?, completed_at = ?
             WHERE session_id = ? AND version = ?
            """,
            (version, completed_cuts, status, blade_used, updated_at,
             completed_at, session_id, expected_version),
        )
    return cur.rowcount


def reset_blade_used(session_id: str, expected_version: int, version: int,
                     updated_at: str, conn: sqlite3.Connection) -> int:
    """换刀确认：累计切长归零并递增版本（乐观条件更新，返回受影响行数）。

    进度/状态不变；WHERE 带 version=expected_version，与并发切刀竞争时只有
    一方更新成功。
    """
    cur = conn.execute(
        """
        UPDATE cut_sessions
           SET version = ?, blade_used = 0.0, updated_at = ?
         WHERE session_id = ? AND version = ?
        """,
        (version, updated_at, session_id, expected_version),
    )
    return cur.rowcount


def insert_session_event(session_id: str, idempotency_key: str, panel_no: int,
                         step: int, request_json: str, response_json: str,
                         created_at: str, conn: sqlite3.Connection,
                         event_version: int) -> None:
    conn.execute(
        """
        INSERT INTO cut_session_events
          (session_id, idempotency_key, panel_no, step, event_version,
           request_json, response_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, idempotency_key, panel_no, step, event_version,
         request_json, response_json, created_at),
    )


def get_session_event(session_id: str, idempotency_key: str,
                      conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT session_id, idempotency_key, panel_no, step, event_version,
               request_json, response_json, created_at
          FROM cut_session_events
         WHERE session_id = ? AND idempotency_key = ?
        """,
        (session_id, idempotency_key),
    ).fetchone()
    return dict(row) if row else None


def count_events_until_version(session_id: str, version: int,
                               conn: sqlite3.Connection) -> int:
    """base_version 之前（含该版本号）已完成的刀数（切刀版本节点数）。

    版本流中切刀（事件/批次）与换刀确认各占一个版本号；批次对账需要把
    base_version 翻译成“已完成刀数”，故按版本流中的切刀节点统计。
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM cut_session_ops
         WHERE session_id = ? AND version <= ? AND kind IN ('event', 'batch')
        """,
        (session_id, version),
    ).fetchone()
    return row["c"]


def insert_session_op(session_id: str, version: int, kind: str,
                      conn: sqlite3.Connection,
                      panel_no: Optional[int] = None,
                      step: Optional[int] = None) -> None:
    """登记一个版本流节点：kind='event'/'batch'（切刀）或 'blade'（换刀）。"""
    conn.execute(
        """
        INSERT INTO cut_session_ops (session_id, version, kind, panel_no, step)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, version, kind, panel_no, step),
    )


def has_blade_change_after_version(session_id: str, version: int,
                                   conn: sqlite3.Connection) -> bool:
    """base_version 之后（不含该版本号）是否发生过换刀确认。

    离线批次的待应用切刀在设备看来属于 base 时刻的刀片；base 之后服务端若
    已确认换刀，这些切刀的切长不能记到新刀片上，整批必须改以换刀后的版本
    为基线重发。
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM cut_session_ops
         WHERE session_id = ? AND version > ? AND kind = 'blade'
        """,
        (session_id, version),
    ).fetchone()
    return row["c"] > 0


def insert_blade_change(session_id: str, idempotency_key: str,
                        from_version: int, request_json: str,
                        response_json: str, created_at: str,
                        conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO cut_session_blade_changes
          (session_id, idempotency_key, from_version,
           request_json, response_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, idempotency_key, from_version,
         request_json, response_json, created_at),
    )


def get_blade_change(session_id: str, idempotency_key: str,
                     conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT session_id, idempotency_key, from_version,
               request_json, response_json, created_at
          FROM cut_session_blade_changes
         WHERE session_id = ? AND idempotency_key = ?
        """,
        (session_id, idempotency_key),
    ).fetchone()
    return dict(row) if row else None


def count_blade_changes(session_id: str, conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM cut_session_blade_changes "
        "WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]


def count_session_events(session_id: str, conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM cut_session_events WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]


def insert_session_batch(session_id: str, batch_id: str, base_version: int,
                         cut_count: int, skipped: int, applied: int,
                         request_json: str, response_json: str,
                         created_at: str, conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO cut_session_batches
          (session_id, batch_id, base_version, cut_count, skipped, applied,
           request_json, response_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, batch_id, base_version, cut_count, skipped, applied,
         request_json, response_json, created_at),
    )


def get_session_batch(session_id: str, batch_id: str,
                      conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT session_id, batch_id, base_version, cut_count, skipped, applied,
               request_json, response_json, created_at
          FROM cut_session_batches
         WHERE session_id = ? AND batch_id = ?
        """,
        (session_id, batch_id),
    ).fetchone()
    return dict(row) if row else None


def count_session_batches(session_id: str, conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM cut_session_batches WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]
