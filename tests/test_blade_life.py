"""刀具寿命检查点与换刀确认：寿命预判、单刀/批次拦截、换刀幂等与并发。

语义：启动会话时可设置 blade_life（刀片寿命，mm）与 initial_used（启动时
已用切长）。推进前按方案快照中该刀长度预判：下一刀会使累计切长越过寿命时
next_action=blade_change，单刀事件 409（blade_life_exceeded）且不推进；
离线批次只要任一待应用切刀会越过检查点就整批 409 且不推进。换刀确认在同一
SQLite 写事务中把已用切长归零、版本 +1；并发换刀与切刀只有一方成功。

服务层并发测试直接用多线程调用 services（每个事务各自开 SQLite 连接，
真实并发）；API 层用 TestClient 做端到端冒烟。需要 fastapi/pydantic 依赖。
"""
import json
import os
import tempfile
import threading
import unittest

from app import database as db
from app.models import (
    BatchCutIn,
    BladeChangeRequest,
    CutBatchRequest,
    CutEventRequest,
    StartSessionRequest,
)
from app.services import (
    SessionConflictError,
    confirm_blade_change,
    create_plan,
    flatten_cuts,
    get_session_state,
    start_session,
    submit_cut_batch,
    submit_cut_event,
)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# 4 块母板、20 刀的多板方案；切长序列前 5 刀为
# 122, 120, 59, 60, 121（累计 122,242,301,361,482）
REQ = {
    "board_width": 244.0,
    "board_height": 122.0,
    "kerf_width": 3.0,
    "max_boards": 4,
    "parts": [
        {"id": "door", "width": 120, "height": 60, "quantity": 4,
         "rotatable": True},
        {"id": "shelf", "width": 60, "height": 40, "quantity": 6,
         "rotatable": True},
    ],
}

ZERO_CUT_REQ = {
    "board_width": 100.0,
    "board_height": 100.0,
    "kerf_width": 3.0,
    "max_boards": 1,
    "parts": [
        {"id": "slab", "width": 100, "height": 100, "quantity": 1,
         "rotatable": False},
    ],
}


def _req(payload):
    """模拟 JSON 往返后的请求体（与线上序列化路径一致）。"""
    return json.loads(json.dumps(payload))


def _batch(batch_id, base_version, cuts):
    return CutBatchRequest(
        batch_id=batch_id,
        base_version=base_version,
        cuts=[BatchCutIn(panel_no=c["panel_no"], step=c["step"])
              for c in cuts],
    )


class _DbTestCase(unittest.TestCase):
    """每个用例一个独立临时库，并把模块级 DB_PATH 指过去。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "blade.db")
        db.init_db(self.path)
        self._orig_path = db.DB_PATH
        db.DB_PATH = self.path

    def tearDown(self):
        db.DB_PATH = self._orig_path
        self.tmp.cleanup()

    def _plan(self, req=REQ):
        return create_plan(_req(req))

    def _start(self, plan_id, blade_life=None, initial_used=0.0):
        body = None
        if blade_life is not None:
            body = StartSessionRequest(blade_life=blade_life,
                                       initial_used=initial_used)
        return start_session(plan_id, body)

    def _new_session(self, blade_life=None, initial_used=0.0, req=REQ):
        plan = self._plan(req)
        session = self._start(plan["plan_id"], blade_life, initial_used)
        return plan, session

    @staticmethod
    def _event(session, key, version, cut):
        return submit_cut_event(session["session_id"], CutEventRequest(
            expected_version=version,
            idempotency_key=key,
            panel_no=cut["panel_no"],
            step=cut["step"],
        ))

    @staticmethod
    def _change(session, key, version):
        return confirm_blade_change(session["session_id"], BladeChangeRequest(
            expected_version=version, idempotency_key=key))


class TestBladeLifeState(_DbTestCase):
    def test_default_session_has_no_blade_life(self):
        plan, sess = self._new_session()
        self.assertIsNone(sess["blade_life"])
        self.assertEqual(sess["blade_used"], 0.0)
        self.assertIsNone(sess["blade_remaining"])
        self.assertEqual(sess["next_action"], "cut")

    def test_start_with_life_and_initial_used(self):
        plan, sess = self._new_session(blade_life=300.0, initial_used=50.0)
        self.assertEqual(sess["blade_life"], 300.0)
        self.assertEqual(sess["blade_used"], 50.0)
        self.assertEqual(sess["blade_remaining"], 250.0)
        # 下一刀 122：50+122=172 ≤ 300 -> 仍可切
        self.assertEqual(sess["next_action"], "cut")

    def test_checkpoint_preview_on_start(self):
        # 初始已用 200：下一刀 122 -> 322 > 300，启动即停在检查点
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        self.assertEqual(sess["next_action"], "blade_change")
        self.assertEqual(sess["blade_remaining"], 100.0)
        self.assertEqual(sess["current_panel"], 1)
        self.assertEqual(sess["next_cut"]["step"], 1)

    def test_exact_life_is_not_exceeded(self):
        # 第一刀恰好 122：initial=178 -> 切完正好 300，不算越过
        plan, sess = self._new_session(blade_life=300.0, initial_used=178.0)
        self.assertEqual(sess["next_action"], "cut")
        cuts = flatten_cuts(plan)
        state = self._event(sess, "e0", 0, cuts[0])
        self.assertEqual(state["blade_used"], 300.0)
        self.assertEqual(state["blade_remaining"], 0.0)
        # 下一刀 120 必然越过 -> blade_change
        self.assertEqual(state["next_action"], "blade_change")

    def test_completed_session_next_action_done(self):
        plan, sess = self._new_session(blade_life=1_000_000.0, req=ZERO_CUT_REQ)
        self.assertTrue(sess["completed"])
        self.assertEqual(sess["next_action"], "done")
        self.assertEqual(sess["blade_remaining"], 1_000_000.0)

    def test_start_validation(self):
        plan = self._plan()
        with self.assertRaises(Exception):
            StartSessionRequest(blade_life=0)
        with self.assertRaises(Exception):
            StartSessionRequest(blade_life=10, initial_used=11)
        with self.assertRaises(Exception):
            # 有 initial_used 就必须给 blade_life
            StartSessionRequest(initial_used=1)
        with self.assertRaises(Exception):
            StartSessionRequest(blade_life=-1)
        with self.assertRaises(Exception):
            StartSessionRequest(blade_life=10, bogus=1)


class TestSingleCutCheckpoint(_DbTestCase):
    def test_event_blocked_at_checkpoint_and_resumes_after_change(self):
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]

        # 第 1 刀 122 -> used 122
        s = self._event(sess, "e0", 0, cuts[0])
        self.assertEqual(s["blade_used"], 122.0)
        self.assertEqual(s["next_action"], "cut")
        # 第 2 刀 120 -> used 242；第 3 刀 59 -> 301 > 300
        s = self._event(sess, "e1", 1, cuts[1])
        self.assertEqual(s["blade_used"], 242.0)
        self.assertEqual(s["next_action"], "blade_change")

        # 越过检查点的第 3 刀被拒：409 blade_life_exceeded，状态不推进
        with self.assertRaises(SessionConflictError) as ctx:
            self._event(sess, "e2", 2, cuts[2])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "blade_life_exceeded")
        self.assertEqual(detail["next_action"], "blade_change")
        self.assertEqual(detail["blade_used"], 242.0)
        self.assertEqual(detail["blade_remaining"], 58.0)
        self.assertEqual(detail["next_cut"]["length"], 59.0)
        state = get_session_state(sid)
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["completed_cuts"], 2)
        self.assertEqual(state["blade_used"], 242.0)
        self.assertEqual(state["next_action"], "blade_change")
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_events(sid, conn), 2)
            self.assertEqual(db.count_blade_changes(sid, conn), 0)

        # 换刀确认：used 归零，version +1（completed_cuts 不变）
        changed = self._change(sess, "blade-1", 2)
        self.assertEqual(changed["version"], 3)
        self.assertEqual(changed["completed_cuts"], 2)
        self.assertEqual(changed["blade_used"], 0.0)
        self.assertEqual(changed["blade_remaining"], 300.0)
        self.assertEqual(changed["next_action"], "cut")
        self.assertEqual(changed["next_cut"]["step"], cuts[2]["step"])

        # 换刀后第 3 刀以新版本提交成功
        s = self._event(sess, "e2", 3, cuts[2])
        self.assertEqual(s["version"], 4)
        self.assertEqual(s["blade_used"], 59.0)
        self.assertEqual(s["next_action"], "cut")

    def test_change_requires_enabled_life(self):
        plan, sess = self._new_session()  # 未设置 blade_life
        with self.assertRaises(SessionConflictError) as ctx:
            self._change(sess, "b", 0)
        self.assertEqual(ctx.exception.detail["error"],
                         "blade_life_not_enabled")
        self.assertEqual(get_session_state(sess["session_id"])["version"], 0)

    def test_change_blocked_when_completed(self):
        plan, sess = self._new_session(blade_life=1_000_000.0,
                                       req=ZERO_CUT_REQ)
        with self.assertRaises(SessionConflictError) as ctx:
            self._change(sess, "b", 0)
        self.assertEqual(ctx.exception.detail["error"], "session_completed")

    def test_change_version_conflict(self):
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        # 已停在检查点；旧版本号 -> 409
        with self.assertRaises(SessionConflictError) as ctx:
            self._change(sess, "b", 7)
        self.assertEqual(ctx.exception.detail["error"], "version_conflict")

    def test_change_idempotent_replay(self):
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        first = self._change(sess, "retry", 0)
        for _ in range(3):
            replay = self._change(sess, "retry", 0)
            self.assertEqual(replay, first)
        state = get_session_state(sess["session_id"])
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["blade_used"], 0.0)
        with db.get_conn() as conn:
            self.assertEqual(db.count_blade_changes(sess["session_id"], conn), 1)

    def test_change_same_key_different_payload_conflicts(self):
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        self._change(sess, "dup", 0)
        with self.assertRaises(SessionConflictError) as ctx:
            self._change(sess, "dup", 1)
        self.assertEqual(ctx.exception.detail["error"],
                         "idempotency_key_reuse")

    def test_change_allows_proactive_change(self):
        # 未到检查点也允许提前换刀（寿命管理由现场决定）
        plan, sess = self._new_session(blade_life=300.0)
        changed = self._change(sess, "early", 0)
        self.assertEqual(changed["version"], 1)
        self.assertEqual(changed["blade_used"], 0.0)
        cuts = flatten_cuts(plan)
        s = self._event(sess, "e0", 1, cuts[0])
        self.assertEqual(s["version"], 2)
        self.assertEqual(s["blade_used"], 122.0)

    def test_event_key_space_separate_from_change_key(self):
        # 事件与换刀的幂等键空间互不影响：同名键可各用一次
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        changed = self._change(sess, "same-key", 0)
        cuts = flatten_cuts(plan)
        s = self._event(sess, "same-key", 1, cuts[0])
        self.assertEqual(changed["version"], 1)
        self.assertEqual(s["version"], 2)

    def test_run_full_session_with_repeated_changes(self):
        # 寿命 300 跑完整张方案：每次到检查点换刀，最终全部完成
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        changes = 0
        for i, cut in enumerate(cuts):
            state = get_session_state(sid)
            if state["next_action"] == "blade_change":
                state = self._change(sess, f"blade-{changes}", state["version"])
                changes += 1
                self.assertEqual(state["next_action"], "cut")
            self.assertEqual(state["next_action"], "cut")
            state = self._event(sess, f"cut-{i}", state["version"], cut)
            self.assertLessEqual(state["blade_used"], 300.0 + 1e-6)
        self.assertTrue(state["completed"])
        self.assertEqual(state["next_action"], "done")
        self.assertEqual(state["completed_cuts"], len(cuts))
        # 版本 = 刀数 + 换刀次数
        self.assertEqual(state["version"], len(cuts) + changes)
        self.assertGreater(changes, 0)
        # 完成后不能再换刀
        with self.assertRaises(SessionConflictError) as ctx:
            self._change(sess, "late", state["version"])
        self.assertEqual(ctx.exception.detail["error"], "session_completed")


class TestBatchCheckpoint(_DbTestCase):
    def test_batch_blocked_when_any_cut_crosses_checkpoint(self):
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        # 先用事件切完前 2 刀：used 242；批次从 base_version=2 回传第 3、4 刀，
        # 第 3 刀 59 -> 301 越过 300
        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])

        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, _batch("offline-x", 2, cuts[2:4]))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "blade_life_exceeded")
        self.assertEqual(detail["phase"], "blade_checkpoint")
        self.assertEqual(detail["index"], 0)  # 批次内第 1 项即越过
        self.assertEqual(detail["cut"]["step"], cuts[2]["step"])
        self.assertEqual(detail["blade_used_before_batch"], 242.0)
        state = get_session_state(sid)
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["completed_cuts"], 2)
        self.assertEqual(state["blade_used"], 242.0)
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_batches(sid, conn), 0)

    def test_batch_ending_exactly_at_life_ok(self):
        plan, sess = self._new_session(blade_life=242.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        # 前 2 刀累计恰为 242
        result = submit_cut_batch(sid, _batch("ok", 0, cuts[:2]))
        self.assertEqual(result["applied"], 2)
        state = result["session"]
        self.assertEqual(state["blade_used"], 242.0)
        self.assertEqual(state["blade_remaining"], 0.0)
        self.assertEqual(state["next_action"], "blade_change")

        # 换刀后再批次推进
        changed = self._change(sess, "b1", 2)
        self.assertEqual((changed["version"], changed["blade_used"]), (3, 0.0))
        # base_version=3（换刀版本），第 3、4 刀 59+60=119 ≤ 242
        result = submit_cut_batch(sid, _batch("ok2", 3, cuts[2:4]))
        self.assertEqual((result["skipped"], result["applied"]), (0, 2))
        self.assertEqual(result["session"]["version"], 5)
        self.assertEqual(result["session"]["blade_used"], 119.0)

    def test_batch_after_change_reconcile_through_blade_version(self):
        """base_version 取换刀版本号时，对账刀数按切刀版本节点计算。"""
        plan, sess = self._new_session(blade_life=242.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        # 前 2 刀走事件 -> version 2，used 242，到检查点
        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])
        # 换刀 -> version 3，done 仍为 2
        changed = self._change(sess, "blade", 2)
        self.assertEqual(changed["version"], 3)

        # 设备在换刀后（看到 version 3）回传第 3、4 刀（59+60=119 ≤ 242）
        result = submit_cut_batch(sid, _batch("ok-3", 3, cuts[2:4]))
        self.assertEqual((result["skipped"], result["applied"]), (0, 2))
        self.assertEqual(result["session"]["version"], 5)
        self.assertEqual(result["session"]["completed_cuts"], 4)
        self.assertEqual(result["session"]["blade_used"], 119.0)

        # 以换刀版本 3 为基线重传第 3~5 刀：第 3、4 刀已推进 -> 跳过 2、
        # 应用 1（第 5 刀 121，累计 240 ≤ 242）
        result = submit_cut_batch(sid, _batch("recon3", 3, cuts[2:5]))
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["session"]["version"], 6)
        self.assertEqual(result["session"]["blade_used"], 240.0)

    def test_batch_replay_after_change_returns_original(self):
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        first = submit_cut_batch(sid, _batch("b", 0, cuts[:2]))
        self._change(sess, "blade", 2)
        # 换刀后重放旧批次：原样返回首次响应（version 2、used 242）
        replay = submit_cut_batch(sid, _batch("b", 0, cuts[:2]))
        self.assertEqual(replay, first)

    def test_batch_after_change_requires_rebase(self):
        """base 之后服务端已换刀：带待应用切刀的旧基线批次 409 整批不推进。"""
        plan, sess = self._new_session(blade_life=242.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        # 前 2 刀事件推进 -> version 2，到检查点；换刀 -> version 3
        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])
        self._change(sess, "blade", 2)

        # 设备旧基线 0 回传前 4 刀（前 2 刀对账跳过，后 2 刀待应用）：
        # base 之后存在换刀版本 -> 409 blade_changed_since_base，整批不推进
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, _batch("stale", 0, cuts[:4]))
        self.assertEqual(ctx.exception.detail["error"],
                         "blade_changed_since_base")
        self.assertEqual(get_session_state(sid)["completed_cuts"], 2)

        # 仅对账不推进的空批次仍允许
        result = submit_cut_batch(sid, _batch("sync", 3, []))
        self.assertEqual((result["skipped"], result["applied"]), (0, 0))

        # 以换刀后版本 3 为基线重发 -> 成功
        result = submit_cut_batch(sid, _batch("rebased", 3, cuts[2:4]))
        self.assertEqual(result["applied"], 2)
        self.assertEqual(result["session"]["blade_used"], 119.0)

    def test_batch_blocked_checkpoint_does_not_consume_id(self):
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])
        with self.assertRaises(SessionConflictError):
            submit_cut_batch(sid, _batch("reuse", 2, cuts[2:4]))
        # 换刀后用同一 batch_id 回传第 3、4 刀（59+60=119 ≤ 300）
        self._change(sess, "blade", 2)
        result = submit_cut_batch(sid, _batch("reuse", 3, cuts[2:4]))
        self.assertEqual(result["applied"], 2)
        self.assertEqual(result["session"]["blade_used"], 119.0)

    def test_event_then_batch_mixed_to_completion(self):
        # 事件与批次混合，寿命 250，分批/换刀直到完成
        plan, sess = self._new_session(blade_life=250.0)
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        pos = 0
        changes = 0
        while pos < len(cuts):
            state = get_session_state(sid)
            if state["next_action"] == "blade_change":
                self._change(sess, f"blade-{changes}", state["version"])
                changes += 1
                continue
            # 事件切 1 刀，批次切 2 刀（不越检查点的前提下）
            self._event(sess, f"e-{pos}", state["version"], cuts[pos])
            pos += 1
            if pos >= len(cuts):
                break
            state = get_session_state(sid)
            window, used = [], state["blade_used"]
            for c in cuts[pos:pos + 2]:
                if used + c["length"] > 250.0 + 1e-6:
                    break
                window.append(c)
                used += c["length"]
            if window:
                r = submit_cut_batch(
                    sid, _batch(f"b-{pos}", state["version"], window))
                self.assertEqual(r["applied"], len(window))
                pos += len(window)
        final = get_session_state(sid)
        self.assertTrue(final["completed"])
        self.assertEqual(final["completed_cuts"], len(cuts))
        self.assertEqual(final["next_action"], "done")


class TestBladeConcurrency(_DbTestCase):
    def test_concurrent_cut_and_change_single_winner(self):
        """8 线程：4 个切同一刀 + 4 个换刀确认，同版本竞争只有一方成功。"""
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        first = flatten_cuts(plan)[0]
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        results, lock = [], threading.Lock()

        def cut_worker(i):
            req = CutEventRequest(expected_version=0,
                                  idempotency_key=f"cut-{i}",
                                  panel_no=first["panel_no"],
                                  step=first["step"])
            barrier.wait()
            try:
                submit_cut_event(sess["session_id"], req)
                outcome = "cut_applied"
            except SessionConflictError as exc:
                outcome = exc.detail["error"]
            with lock:
                results.append(outcome)

        def change_worker(i):
            req = BladeChangeRequest(expected_version=0,
                                     idempotency_key=f"blade-{i}")
            barrier.wait()
            try:
                confirm_blade_change(sess["session_id"], req)
                outcome = "change_applied"
            except SessionConflictError as exc:
                outcome = exc.detail["error"]
            with lock:
                results.append(outcome)

        threads = ([threading.Thread(target=cut_worker, args=(i,))
                    for i in range(4)]
                   + [threading.Thread(target=change_worker, args=(i,))
                      for i in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r.endswith("_applied")]
        self.assertEqual(len(winners), 1, results)
        # 其余全部 409：换刀线程只会收到 version_conflict；切刀线程收到
        # version_conflict（换刀抢先）或 blade_life_exceeded（另一切刀抢先，
        # 此时下一刀 122 仍越过仅剩的 100）
        self.assertEqual(len([r for r in results if r in (
            "version_conflict", "blade_life_exceeded")]), n_threads - 1)
        state = get_session_state(sess["session_id"])
        self.assertEqual(state["version"], 1)
        if winners[0] == "change_applied":
            self.assertEqual(state["completed_cuts"], 0)
            self.assertEqual(state["blade_used"], 0.0)
        else:
            self.assertEqual(state["completed_cuts"], 1)
            self.assertEqual(state["blade_used"], 122.0)

    def test_concurrent_batch_and_change_single_winner(self):
        """批次（带待应用切刀）与换刀同版本竞争 12 轮：每轮恰一方成功。"""
        plan, sess = self._new_session(blade_life=1000.0)
        first = flatten_cuts(plan)[0]
        rounds = 12
        summary = {}

        def one_round(i):
            # 每轮独立的带寿命会话
            s = start_session(
                plan["plan_id"], StartSessionRequest(blade_life=1000.0))
            sid = s["session_id"]
            barrier = threading.Barrier(2)
            out = {}

            def batch_worker():
                req = _batch(f"b-{i}", 0, [first])
                barrier.wait()
                try:
                    submit_cut_batch(sid, req)
                    out["b"] = "batch_ok"
                except SessionConflictError as exc:
                    out["b"] = "batch_" + exc.detail["error"]

            def change_worker():
                req = BladeChangeRequest(expected_version=0,
                                         idempotency_key=f"k-{i}")
                barrier.wait()
                try:
                    confirm_blade_change(sid, req)
                    out["c"] = "change_ok"
                except SessionConflictError as exc:
                    out["c"] = "change_" + exc.detail["error"]

            t1 = threading.Thread(target=batch_worker)
            t2 = threading.Thread(target=change_worker)
            t1.start(); t2.start(); t1.join(); t2.join()
            return out, get_session_state(sid)["version"]

        for i in range(rounds):
            out, ver = one_round(i)
            summary[(out["b"], out["c"])] = \
                summary.get((out["b"], out["c"]), 0) + 1
            winners = [v for v in out.values() if v.endswith("_ok")]
            self.assertEqual(len(winners), 1, out)
            self.assertEqual(ver, 1)
            # 换刀胜：批次必为 blade_changed_since_base 或 version_conflict；
            # 批次胜：换刀必为 version_conflict
            self.assertIn(out["b"], (
                "batch_ok", "batch_blade_changed_since_base",
                "batch_version_conflict"))
            self.assertIn(out["c"], ("change_ok", "change_version_conflict"))

    def test_concurrent_change_same_key_all_get_original(self):
        plan, sess = self._new_session(blade_life=300.0, initial_used=200.0)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        responses, lock = [], threading.Lock()

        def worker():
            req = BladeChangeRequest(expected_version=0,
                                     idempotency_key="shared-blade")
            barrier.wait()
            with lock:
                responses.append(
                    confirm_blade_change(sess["session_id"], req))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(responses), n_threads)
        for resp in responses:
            self.assertEqual(resp, responses[0])
        state = get_session_state(sess["session_id"])
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["blade_used"], 0.0)
        with db.get_conn() as conn:
            self.assertEqual(
                db.count_blade_changes(sess["session_id"], conn), 1)


    def test_concurrent_soak_events_batches_changes(self):
        """事件/批次/换刀看门狗混合并发跑到完成：终态一致、版本节点守恒。"""
        import uuid
        plan, sess = self._new_session(blade_life=250.0)
        cuts = flatten_cuts(plan)
        total = len(cuts)
        sid = sess["session_id"]
        errors, lock = [], threading.Lock()
        done_ev, done_ba, changes = [0], [0], [0]
        stop = threading.Event()

        def ev_worker():
            while not stop.is_set():
                st = get_session_state(sid)
                if st["completed"]:
                    return
                if st["next_action"] == "blade_change":
                    continue  # 交给看门狗
                nxt = st["next_cut"]
                try:
                    submit_cut_event(sid, CutEventRequest(
                        expected_version=st["version"],
                        idempotency_key=uuid.uuid4().hex,
                        panel_no=nxt["panel_no"], step=nxt["step"]))
                    with lock:
                        done_ev[0] += 1
                except SessionConflictError as exc:
                    if exc.detail["error"] not in (
                            "version_conflict", "blade_life_exceeded",
                            "session_completed"):
                        with lock:
                            errors.append(("ev", exc.detail))
                except Exception as exc:
                    with lock:
                        errors.append(("ev", str(exc)))

        def ba_worker():
            while not stop.is_set():
                st = get_session_state(sid)
                if st["completed"]:
                    return
                used = st["blade_used"]
                chunk = []
                for c in cuts[st["completed_cuts"]:]:
                    if used + c["length"] > 250.0 + 1e-6:
                        break
                    chunk.append(c)
                    used += c["length"]
                    if len(chunk) == 3:
                        break
                if not chunk:
                    continue
                try:
                    r = submit_cut_batch(sid, _batch(
                        uuid.uuid4().hex, st["version"], chunk))
                    with lock:
                        done_ba[0] += r["applied"]
                except SessionConflictError as exc:
                    if exc.detail["error"] not in (
                            "batch_mismatch", "version_conflict",
                            "blade_life_exceeded",
                            "blade_changed_since_base",
                            "session_completed"):
                        with lock:
                            errors.append(("ba", exc.detail))
                except Exception as exc:
                    with lock:
                        errors.append(("ba", str(exc)))

        def watcher():
            while not stop.is_set():
                st = get_session_state(sid)
                if st["completed"]:
                    return
                if st["next_action"] == "blade_change":
                    try:
                        confirm_blade_change(sid, BladeChangeRequest(
                            expected_version=st["version"],
                            idempotency_key=uuid.uuid4().hex))
                        with lock:
                            changes[0] += 1
                    except SessionConflictError as exc:
                        if exc.detail["error"] not in (
                                "version_conflict", "session_completed"):
                            with lock:
                                errors.append(("w", exc.detail))
                    except Exception as exc:
                        with lock:
                            errors.append(("w", str(exc)))

        threads = ([threading.Thread(target=ev_worker) for _ in range(3)]
                   + [threading.Thread(target=ba_worker) for _ in range(2)]
                   + [threading.Thread(target=watcher)])
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        stop.set()
        for t in threads:
            t.join()

        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(errors, [])
        final = get_session_state(sid)
        self.assertTrue(final["completed"])
        self.assertEqual(final["next_action"], "done")
        self.assertEqual(final["completed_cuts"], total)
        # 每个版本节点非刀即换刀：version == 总刀数 + 换刀次数
        self.assertEqual(final["version"], total + changes[0])
        self.assertEqual(done_ev[0] + done_ba[0], total)
        self.assertLessEqual(final["blade_used"], 250.0 + 1e-6)


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi，跳过 API 层测试")
class TestBladeApi(_DbTestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app)

    def _create_plan(self, req=REQ):
        r = self.client.post("/api/v1/plans", json=req)
        self.assertEqual(r.status_code, 200)
        return r.json()

    def _start(self, plan_id, body=None):
        r = self.client.post(f"/api/v1/plans/{plan_id}/sessions", json=body)
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def test_start_without_body_still_works(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"])  # 无请求体
        self.assertIsNone(sess["blade_life"])
        self.assertIsNone(sess["blade_remaining"])
        self.assertEqual(sess["next_action"], "cut")

    def test_start_with_blade_life(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"],
                           {"blade_life": 300, "initial_used": 200})
        self.assertEqual(sess["blade_life"], 300.0)
        self.assertEqual(sess["blade_used"], 200.0)
        self.assertEqual(sess["blade_remaining"], 100.0)
        self.assertEqual(sess["next_action"], "blade_change")

        # 查询接口同构
        r = self.client.get(f"/api/v1/sessions/{sess['session_id']}")
        self.assertEqual(r.json(), sess)

    def test_start_validation_422(self):
        plan = self._create_plan()
        url = f"/api/v1/plans/{plan['plan_id']}/sessions"
        # initial_used 超过 blade_life
        r = self.client.post(url, json={"blade_life": 10, "initial_used": 11})
        self.assertEqual(r.status_code, 422)
        # 只有 initial_used
        r = self.client.post(url, json={"initial_used": 1})
        self.assertEqual(r.status_code, 422)
        # blade_life 非正
        r = self.client.post(url, json={"blade_life": 0})
        self.assertEqual(r.status_code, 422)
        # 多余字段
        r = self.client.post(url, json={"blade_life": 10, "bogus": 1})
        self.assertEqual(r.status_code, 422)

    def test_event_blocked_and_change_flow(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"], {"blade_life": 300})
        sid = sess["session_id"]
        cuts = flatten_cuts(plan)

        def event(i, version, key=None):
            return self.client.post(
                f"/api/v1/sessions/{sid}/events",
                json={"expected_version": version,
                      "idempotency_key": key or f"e{i}",
                      "panel_no": cuts[i]["panel_no"],
                      "step": cuts[i]["step"]})

        r = event(0, 0)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["blade_used"], 122.0)
        r = event(1, 1)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["blade_used"], 242.0)
        self.assertEqual(body["next_action"], "blade_change")

        # 第 3 刀越过检查点 -> 409 blade_life_exceeded
        r = event(2, 2)
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "blade_life_exceeded")
        self.assertEqual(detail["next_action"], "blade_change")

        # 错误版本换刀 -> 409
        r = self.client.post(f"/api/v1/sessions/{sid}/blade-changes",
                             json={"expected_version": 9,
                                   "idempotency_key": "b1"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "version_conflict")

        # 正确换刀 -> used 归零、version 3
        r = self.client.post(f"/api/v1/sessions/{sid}/blade-changes",
                             json={"expected_version": 2,
                                   "idempotency_key": "b1"})
        self.assertEqual(r.status_code, 200)
        changed = r.json()
        self.assertEqual(changed["version"], 3)
        self.assertEqual(changed["blade_used"], 0.0)
        self.assertEqual(changed["blade_remaining"], 300.0)
        self.assertEqual(changed["next_action"], "cut")

        # 同键重放逐字段一致
        r2 = self.client.post(f"/api/v1/sessions/{sid}/blade-changes",
                              json={"expected_version": 2,
                                    "idempotency_key": "b1"})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json(), changed)

        # 换刀后第 3 刀以版本 3 提交
        r = event(2, 3)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], 4)
        self.assertEqual(r.json()["blade_used"], 59.0)

    def test_change_on_session_without_life_409(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"])
        r = self.client.post(
            f"/api/v1/sessions/{sess['session_id']}/blade-changes",
            json={"expected_version": 0, "idempotency_key": "b"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"],
                         "blade_life_not_enabled")

    def test_batch_crossing_checkpoint_409_whole_batch(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"], {"blade_life": 300})
        sid = sess["session_id"]
        cuts = flatten_cuts(plan)
        payload = {"batch_id": "offline", "base_version": 0,
                   "cuts": [{"panel_no": c["panel_no"], "step": c["step"]}
                            for c in cuts[:4]]}
        r = self.client.post(f"/api/v1/sessions/{sid}/event-batches",
                             json=payload)
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "blade_life_exceeded")
        self.assertEqual(detail["phase"], "blade_checkpoint")
        self.assertEqual(detail["index"], 2)
        # 整批未推进
        state = self.client.get(f"/api/v1/sessions/{sid}").json()
        self.assertEqual(state["version"], 0)
        self.assertEqual(state["blade_used"], 0.0)

    def test_change_validation_422_and_404(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"], {"blade_life": 300})
        url = f"/api/v1/sessions/{sess['session_id']}/blade-changes"
        r = self.client.post(url, json={"idempotency_key": "b"})
        self.assertEqual(r.status_code, 422)
        r = self.client.post(url, json={"expected_version": -1,
                                        "idempotency_key": "b"})
        self.assertEqual(r.status_code, 422)
        r = self.client.post(url, json={"expected_version": 0,
                                        "idempotency_key": "b", "x": 1})
        self.assertEqual(r.status_code, 422)
        missing = "00000000-0000-0000-0000-000000000000"
        r = self.client.post(
            f"/api/v1/sessions/{missing}/blade-changes",
            json={"expected_version": 0, "idempotency_key": "b"})
        self.assertEqual(r.status_code, 404)

    def test_openapi_lists_blade_endpoints(self):
        spec = self.client.get("/openapi.json").json()
        paths = spec["paths"]
        self.assertIn("/api/v1/sessions/{session_id}/blade-changes", paths)
        schemas = spec["components"]["schemas"]
        self.assertIn("BladeChangeRequest", schemas)
        self.assertIn("StartSessionRequest", schemas)
        session_props = schemas["SessionResponse"]["properties"]
        for name in ("blade_life", "blade_used", "blade_remaining",
                     "next_action"):
            self.assertIn(name, session_props)


if __name__ == "__main__":
    unittest.main()
