"""排样业务编排：参数模型 -> 核心算法 -> 响应装配 -> 落库。

切割执行会话：从已保存方案启动会话，按 (panel_no, step) 顺序逐刀推进；
“完成当前刀”事件、换刀确认与离线回传批次在单个 SQLite 写事务中完成幂等
判定、版本/对账/刀具寿命校验、状态推进和事件/批次/换刀记录，并发/重试语义
见 submit_cut_event、confirm_blade_change 与 submit_cut_batch。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import database as db
from .gcode import generate_gcode, validate_plan_cuts
from .models import (
    BLADE_EPS,
    BatchCutIn,
    CutBatchRequest,
    CutEventRequest,
    GcodeRequest,
    PlanRequest,
    StartSessionRequest,
)
from .nesting import build_payload, solve

# 累计切长保留的小数位（mm）：与排样输出精度对齐，避免浮点尾差影响寿命判定
_BLADE_NDIGITS = 6


def _round6(x: float) -> float:
    return round(float(x), _BLADE_NDIGITS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_plan(req: PlanRequest) -> dict:
    """纯计算（不碰数据库），便于单元测试。"""
    parts = [
        {
            "id": p.id,
            "width": p.width,
            "height": p.height,
            "quantity": p.quantity,
            "rotatable": p.rotatable,
        }
        for p in req.parts
    ]
    internal = solve(
        board_w=req.board_width,
        board_h=req.board_height,
        kerf=req.kerf_width,
        max_boards=req.max_boards,
        parts=parts,
    )
    return build_payload(internal)


def create_plan(req_dict: dict) -> dict:
    """幂等创建：相同请求返回同一方案。"""
    request_json = db.canonical_request(req_dict)
    plan_id = db.plan_id_from_request(req_dict)

    with db.get_conn() as conn:
        existing = db.get_plan(plan_id, conn)
        if existing is not None:
            return existing

        payload = compute_plan(PlanRequest.model_validate(req_dict))
        payload["plan_id"] = plan_id
        payload["created_at"] = _now_iso()
        db.save_plan(plan_id, request_json, json.dumps(payload, ensure_ascii=False),
                     payload["created_at"], conn)
        return payload


class PlanGeometryError(Exception):
    """已存方案未通过导出前几何校验；violations 为结构化错误列表。"""

    def __init__(self, violations: list[dict]):
        super().__init__("方案切割线未通过导出前几何校验")
        self.violations = violations


def export_gcode(plan_id: str, req: GcodeRequest) -> Optional[dict]:
    """导出已存方案的 G-code。

    方案不存在返回 None；几何校验失败抛 PlanGeometryError（不产出任何
    部分 G-code）。
    """
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
    if plan is None:
        return None
    violations = validate_plan_cuts(plan)
    if violations:
        raise PlanGeometryError(violations)
    return generate_gcode(
        plan,
        plan_id=plan_id,
        origin=req.origin,
        unit=req.unit,
        safe_z=req.safe_z,
    )


# ---------------------------------------------------------------------------
# 切割执行会话
# ---------------------------------------------------------------------------

def flatten_cuts(plan: dict) -> list[dict]:
    """把方案的切割线展平为全局执行序列（按 panel_no、step 升序）。

    返回项在方案 CutOut 之上附带 panel_no；顺序与 G-code 导出一致：
    先执行的刀把后续刀所在区域切出，母板按编号依次上机。
    """
    seq: list[dict] = []
    for panel in plan.get("panels", []):
        for cut in panel.get("cuts", []):
            seq.append({
                "panel_no": panel["panel_no"],
                "step": cut["step"],
                "axis": cut["axis"],
                "x": cut["x"],
                "y": cut["y"],
                "length": cut["length"],
                "region": cut["region"],
            })
    seq.sort(key=lambda c: (c["panel_no"], c["step"]))
    return seq


def _blade_view(row: dict, cuts: list[dict], done: int) -> dict:
    """计算刀具寿命视图：blade_used/blade_remaining/next_action。

    next_action：已完成 -> done；启用寿命且下一刀会使累计切长越过寿命 ->
    blade_change（恰好等于寿命视为不越过）；否则 cut。
    """
    life = row["blade_life"]
    used = _round6(row["blade_used"])
    if life is None:
        next_action = "done" if done >= len(cuts) else "cut"
        return {"blade_life": None, "blade_used": used,
                "blade_remaining": None, "next_action": next_action}
    life = float(life)
    if done >= len(cuts):
        next_action = "done"
    elif used + float(cuts[done]["length"]) > life + BLADE_EPS:
        next_action = "blade_change"
    else:
        next_action = "cut"
    return {
        "blade_life": life,
        "blade_used": used,
        "blade_remaining": _round6(life - used),
        "next_action": next_action,
    }


def _session_state(row: dict, plan: Optional[dict] = None) -> dict:
    """由会话行（+ 方案）装配对外会话状态。"""
    if plan is None:
        plan = json.loads(row["plan_json"])
    cuts = flatten_cuts(plan)
    done = row["completed_cuts"]
    nxt = cuts[done] if done < len(cuts) else None
    return {
        "session_id": row["session_id"],
        "plan_id": row["plan_id"],
        "status": row["status"],
        "completed": row["status"] == "completed",
        "version": row["version"],
        "completed_cuts": done,
        "total_cuts": row["total_cuts"],
        "current_panel": nxt["panel_no"] if nxt else None,
        "next_cut": nxt,
        **_blade_view(row, cuts, done),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def start_session(plan_id: str,
                  blade: Optional[StartSessionRequest] = None) -> Optional[dict]:
    """从已保存方案启动切割会话；方案不存在返回 None。

    blade 可携带 blade_life（刀具寿命，mm）与 initial_used（启动时已用切长）；
    请求体整体省略时不做刀具寿命检查。无切割线的方案（零件恰好铺满无余料
    切刀，或全部未排）启动即完成。
    """
    blade_life = None
    initial_used = 0.0
    if blade is not None:
        blade_life = blade.blade_life
        initial_used = _round6(blade.initial_used)
        if blade_life is not None:
            blade_life = float(blade_life)
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
        if plan is None:
            return None
        total = len(flatten_cuts(plan))
        now = _now_iso()
        session_id = str(uuid.uuid4())
        done = total == 0
        db.insert_session(
            session_id=session_id,
            plan_id=plan_id,
            status="completed" if done else "active",
            version=0,
            completed_cuts=0,
            total_cuts=total,
            created_at=now,
            updated_at=now,
            completed_at=now if done else None,
            conn=conn,
            blade_life=blade_life,
            blade_used=initial_used,
        )
        conn.commit()
        row = db.get_session(session_id, conn)
    return _session_state(row, plan)


def get_session_state(session_id: str) -> Optional[dict]:
    """查询会话状态；不存在返回 None。"""
    with db.get_conn() as conn:
        row = db.get_session(session_id, conn)
    if row is None:
        return None
    return _session_state(row)


class SessionConflictError(Exception):
    """409 冲突：detail 为结构化错误体（error/message + 上下文）。"""

    def __init__(self, detail: dict):
        super().__init__(detail.get("message", "会话状态冲突"))
        self.detail = detail


def submit_cut_event(session_id: str, req: CutEventRequest) -> Optional[dict]:
    """提交“完成当前刀”事件；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, idempotency_key) 已应用且请求体一致 ->
         原样返回首次响应（状态不重复推进）；同键不同请求 -> 409；
      2. 校验会话未完成、expected_version 等于当前版本、所报
         (panel_no, step) 正是当前刀 —— 任一不满足抛 SessionConflictError，
         事务回滚，状态不推进；
      3. 推进 version/completed_cuts（全部完成时置 completed），记录事件，
         一次性提交。

    BEGIN IMMEDIATE 使并发写者在库级串行：并发同版本请求只有一个能通过
    校验，其余读到推进后的版本并以 version_conflict 拒绝。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同键重试返回首次结果（此时版本已推进，必须先于版本校验）
        evt = db.get_session_event(session_id, req.idempotency_key, conn)
        if evt is not None:
            if evt["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "idempotency_key_reuse",
                    "message": "幂等键已被本会话中的不同请求使用",
                })
            return json.loads(evt["response_json"])

        # 2) 校验：会话未完成 -> 版本一致 -> 所报刀正是当前刀 -> 未越过寿命检查点
        if row["status"] == "completed":
            raise SessionConflictError({
                "error": "session_completed",
                "message": f"会话已完成（共 {row['total_cuts']} 刀），"
                           "不能再提交切割事件",
                "total_cuts": row["total_cuts"],
            })
        if req.expected_version != row["version"]:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           f"{row['version']} 不一致，状态未推进",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })
        plan = json.loads(row["plan_json"])
        cuts = flatten_cuts(plan)
        nxt = cuts[row["completed_cuts"]]
        if req.panel_no != nxt["panel_no"] or req.step != nxt["step"]:
            raise SessionConflictError({
                "error": "cut_mismatch",
                "message": f"请求完成第 {req.panel_no} 板第 {req.step} 刀，"
                           f"但当前刀是第 {nxt['panel_no']} 板第 "
                           f"{nxt['step']} 刀，状态未推进",
                "expected": {"panel_no": nxt["panel_no"], "step": nxt["step"]},
                "received": {"panel_no": req.panel_no, "step": req.step},
            })
        life = row["blade_life"]
        used = float(row["blade_used"])
        new_used = used + float(nxt["length"])
        if life is not None and new_used > float(life) + BLADE_EPS:
            # 寿命检查点：本刀不得越过，先调换刀确认
            raise SessionConflictError({
                "error": "blade_life_exceeded",
                "message": f"下一刀（第 {nxt['panel_no']} 板第 {nxt['step']} 刀，"
                           f"切长 {nxt['length']}）会使累计切长 {_round6(new_used)} "
                           f"超过刀片寿命 {float(life)}，请先确认换刀",
                "blade_life": float(life),
                "blade_used": _round6(used),
                "blade_remaining": _round6(float(life) - used),
                "next_cut": {"panel_no": nxt["panel_no"],
                             "step": nxt["step"],
                             "length": nxt["length"]},
                "next_action": "blade_change",
            })

        # 3) 推进 + 记录事件，同事务提交；乐观条件更新兜住读-写之间的并发
        now = _now_iso()
        new_done = row["completed_cuts"] + 1
        finished = new_done >= row["total_cuts"]
        new_version = row["version"] + 1
        new_used = _round6(new_used) if life is not None else None
        updated = db.update_session_progress(
            session_id=session_id,
            expected_version=row["version"],
            version=new_version,
            completed_cuts=new_done,
            status="completed" if finished else "active",
            updated_at=now,
            completed_at=now if finished else None,
            conn=conn,
            blade_used=new_used,
        )
        if updated == 0:
            # 并发换刀/事件/批次抢先提交：回滚后以版本冲突拒绝
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           "不一致（并发写者抢先推进），状态未推进",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })
        state = _session_state(db.get_session(session_id, conn), plan)
        db.insert_session_event(
            session_id=session_id,
            idempotency_key=req.idempotency_key,
            panel_no=req.panel_no,
            step=req.step,
            request_json=request_json,
            response_json=json.dumps(state, ensure_ascii=False),
            created_at=now,
            conn=conn,
            event_version=new_version,
        )
        db.insert_session_op(session_id, new_version, "event", conn,
                             panel_no=req.panel_no, step=req.step)
        return state


def confirm_blade_change(session_id: str, req) -> Optional[dict]:
    """确认“已换刀”；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, idempotency_key) 已应用且请求体一致 ->
         原样返回首次响应；同键不同请求 -> 409；
      2. 校验会话未完成、expected_version 等于当前版本（会话必须启用了
         blade_life；寿命未启用时换刀无意义 -> 409 blade_life_not_enabled）；
      3. 把已用切长归零、version +1（completed_cuts 不变），记录换刀确认，
         一次性提交。

    与切刀事件共用同一写事务与版本号空间：并发换刀与切刀在库级串行，
    后到者读到推进后的版本并以 version_conflict 拒绝——只有一方成功。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同键重试返回首次结果（必须先于版本校验）
        change = db.get_blade_change(session_id, req.idempotency_key, conn)
        if change is not None:
            if change["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "idempotency_key_reuse",
                    "message": "幂等键已被本会话中的不同请求使用",
                })
            return json.loads(change["response_json"])

        # 2) 校验
        if row["blade_life"] is None:
            raise SessionConflictError({
                "error": "blade_life_not_enabled",
                "message": "本会话启动时未设置 blade_life，无需换刀确认",
            })
        if row["status"] == "completed":
            raise SessionConflictError({
                "error": "session_completed",
                "message": f"会话已完成（共 {row['total_cuts']} 刀），"
                           "不能再确认换刀",
                "total_cuts": row["total_cuts"],
            })
        if req.expected_version != row["version"]:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           f"{row['version']} 不一致，换刀未确认",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })

        # 3) 已用切长归零、版本 +1，同事务提交；乐观条件更新兜住并发切刀
        now = _now_iso()
        new_version = row["version"] + 1
        updated = db.reset_blade_used(session_id, row["version"], new_version,
                                      now, conn)
        if updated == 0:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           "不一致（并发切刀/换刀抢先推进），换刀未确认",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })
        state = _session_state(db.get_session(session_id, conn))
        db.insert_blade_change(
            session_id=session_id,
            idempotency_key=req.idempotency_key,
            from_version=row["version"],
            request_json=request_json,
            response_json=json.dumps(state, ensure_ascii=False),
            created_at=now,
            conn=conn,
        )
        db.insert_session_op(session_id, new_version, "blade", conn)
        return state


def _batch_mismatch_detail(phase: str, index: int,
                           expected: Optional[dict],
                           received: Optional[BatchCutIn],
                           skip: int, n: int) -> dict:
    """组装批次校验失败的结构化错误体（整批不推进）。

    phase="reconcile"：批次前缀与 base_version 之后服务端已完成区间对账失败；
    phase="apply"：其余后缀与当前方案的下一刀序列不匹配；
    phase="blade_checkpoint"：后缀中某一刀会越过刀具寿命检查点。
    index 为批次 cuts 数组下标（从 0 开始）；expected/received 为 None 分别
    表示方案已无剩余切刀 / 批次未覆盖到该位置。
    """
    exp = ({"panel_no": expected["panel_no"], "step": expected["step"]}
           if expected is not None else None)
    got = ({"panel_no": received.panel_no, "step": received.step}
           if received is not None else None)
    if phase == "reconcile" and received is None:
        message = (f"批次仅 {n} 刀，不足以覆盖 base_version 之后服务端已完成的 "
                   f"{skip} 刀，无法逐项对账，整批未推进")
    elif phase == "reconcile":
        message = (f"cuts[{index}] 为第 {got['panel_no']} 板第 {got['step']} 刀，"
                   f"但 base_version 之后服务端已完成的该刀是第 "
                   f"{exp['panel_no']} 板第 {exp['step']} 刀，整批未推进")
    elif expected is None:
        message = (f"cuts[{index}] 超出方案剩余切刀：会话已无可推进的刀，"
                   "整批未推进")
    else:
        message = (f"cuts[{index}] 为第 {got['panel_no']} 板第 {got['step']} 刀，"
                   f"但当前方案的下一刀是第 {exp['panel_no']} 板第 "
                   f"{exp['step']} 刀，整批未推进")
    return {
        "error": "batch_mismatch",
        "message": message,
        "phase": phase,
        "index": index,
        "expected": exp,
        "received": got,
    }


def _blade_checkpoint_detail(index: int, cut: dict, used: float,
                             length: float, life: float) -> dict:
    """组装批次越过刀具寿命检查点的结构化错误体（整批不推进）。"""
    return {
        "error": "blade_life_exceeded",
        "message": f"批次 cuts[{index}]（第 {cut['panel_no']} 板第 "
                   f"{cut['step']} 刀，切长 {length}）会使累计切长 "
                   f"{_round6(used + length)} 超过刀片寿命 {float(life)}；"
                   "请把该刀及其后切刀放到换刀确认之后的批次回传，整批未推进",
        "phase": "blade_checkpoint",
        "index": index,
        "blade_life": float(life),
        "blade_used_before_batch": _round6(used),
        "cut": {"panel_no": cut["panel_no"], "step": cut["step"],
                "length": length},
        "next_action": "blade_change",
    }


def submit_cut_batch(session_id: str, req: CutBatchRequest) -> Optional[dict]:
    """提交设备离线期间完成的切刀批次；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, batch_id) 已应用且请求体一致 ->
         原样返回首次响应（状态不重复推进）；同键不同请求 -> 409；
      2. 基线校验：base_version 超过当前版本（客户端基线比服务端还新，
         无法对账）-> 409；
      3. 对账：base_version 之后服务端已完成的区间与批次前缀逐项比对，
         一致前缀仅跳过；首个不一致 -> 409，整批不推进；
      4. 其余后缀必须连续匹配当前方案的下一刀序列，任一不符 -> 409；
         启用刀具寿命时，后缀中任一刀会使累计切长越过寿命检查点 -> 409
         （blade_life_exceeded，整批不推进，需先经换刀确认再回传后续批次）；
      5. 验证通过后一次性推进 version/completed_cuts/blade_used（全部完成时
         置 completed），记录批次，一次性提交；applied=0 时不改动会话行。

    版本流中换刀确认也占一个版本号，故对账区间刀数由 base_version 之前的
    切刀版本节点数（事件 + 批次应用的刀，不含换刀）确定。与单刀事件/换刀
    确认共用同一写事务机制：批次与单刀、批次与换刀、批次与批次并发时在库级
    串行，后到者基于最新状态重新对账——已被覆盖的批次自然转为“全部跳过”或
    409，不会重复推进。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同 batch_id 重放首次响应（必须先于一切校验）
        batch = db.get_session_batch(session_id, req.batch_id, conn)
        if batch is not None:
            if batch["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "batch_id_reuse",
                    "message": "batch_id 已被本会话中的不同批次请求使用",
                })
            return json.loads(batch["response_json"])

        version = row["version"]
        done = row["completed_cuts"]
        total = row["total_cuts"]

        # 2) 基线校验：base_version 比服务端还新，对账区间无意义
        if req.base_version > version:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"base_version={req.base_version} 超过当前版本 "
                           f"{version}，无法对账，整批未推进",
                "base_version": req.base_version,
                "current_version": version,
            })

        plan = json.loads(row["plan_json"])
        cuts = flatten_cuts(plan)
        n = len(req.cuts)
        # 换刀确认也占版本号：base_version 对应的已完成刀数按切刀事件计数
        base_done = db.count_events_until_version(
            session_id, req.base_version, conn)
        skip = done - base_done

        # 3) 对账：base_version 之后已完成的区间必须与批次前缀逐项一致
        for i in range(skip):
            expected = cuts[base_done + i]
            received = req.cuts[i] if i < n else None
            if (received is None
                    or received.panel_no != expected["panel_no"]
                    or received.step != expected["step"]):
                raise SessionConflictError(_batch_mismatch_detail(
                    "reconcile", i, expected, received, skip, n))

        # 4) 其余后缀必须连续匹配当前方案的下一刀序列；且不得越过寿命检查点
        life = row["blade_life"]
        used = float(row["blade_used"])
        # base_version 之后服务端若已确认换刀，待应用切刀（设备离线时用旧
        # 刀片完成）不能记到新刀片上：整批改以换刀后版本为基线重发
        applied_cut_count = n - skip
        if (applied_cut_count > 0
                and db.has_blade_change_after_version(
                    session_id, req.base_version, conn)):
            raise SessionConflictError({
                "error": "blade_changed_since_base",
                "message": f"base_version={req.base_version} 之后服务端已确认"
                           "换刀，本批次待应用切刀不能记到新刀片上，整批未推进；"
                           "请以换刀后的版本为基线重发后续切刀",
                "base_version": req.base_version,
                "current_version": version,
            })
        applied_length = 0.0
        for k in range(skip, n):
            pos = done + (k - skip)
            expected = cuts[pos] if pos < total else None
            received = req.cuts[k]
            if (expected is None
                    or received.panel_no != expected["panel_no"]
                    or received.step != expected["step"]):
                raise SessionConflictError(_batch_mismatch_detail(
                    "apply", k, expected, received, skip, n))
            length = float(expected["length"])
            if life is not None and used + applied_length + length \
                    > float(life) + BLADE_EPS:
                raise SessionConflictError(_blade_checkpoint_detail(
                    k, expected, used + applied_length, length, life))
            applied_length += length

        # 5) 验证通过：一次性推进版本与完成数，记录批次，同事务提交
        applied = n - skip
        now = _now_iso()
        if applied > 0:
            new_done = done + applied
            finished = new_done >= total
            new_used = (_round6(used + applied_length)
                        if life is not None else None)
            updated = db.update_session_progress(
                session_id=session_id,
                expected_version=version,
                version=version + applied,
                completed_cuts=new_done,
                status="completed" if finished else "active",
                updated_at=now,
                completed_at=now if finished else None,
                conn=conn,
                blade_used=new_used,
            )
            if updated == 0:
                # 并发事件/换刀/批次抢先提交：整批回滚，客户端重读后重试对账
                raise SessionConflictError({
                    "error": "version_conflict",
                    "message": f"base_version={req.base_version} 对应的会话状态 "
                               "已被并发写者推进，整批未推进，请重读后重试",
                    "base_version": req.base_version,
                    "current_version": version,
                })
            for j in range(applied):
                applied_cut = cuts[done + j]
                db.insert_session_op(
                    session_id, version + j + 1, "batch", conn,
                    panel_no=applied_cut["panel_no"], step=applied_cut["step"])
            row = db.get_session(session_id, conn)
        result = {
            "skipped": skip,
            "applied": applied,
            "session": _session_state(row, plan),
        }
        db.insert_session_batch(
            session_id=session_id,
            batch_id=req.batch_id,
            base_version=req.base_version,
            cut_count=n,
            skipped=skip,
            applied=applied,
            request_json=request_json,
            response_json=json.dumps(result, ensure_ascii=False),
            created_at=now,
            conn=conn,
        )
        return result
