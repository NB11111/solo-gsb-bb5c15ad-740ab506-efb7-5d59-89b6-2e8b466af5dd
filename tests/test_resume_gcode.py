"""按切割会话进度导出续切 G-code：版本快照校验、续切点、换板暂停与只读语义。

服务层在同一 SQLite 只读快照事务中校验 expected_version，并仅从
completed_cuts 对应的下一刀起生成程序：首个待切母板直接进入，之后跨板各
输出一次 M0；注释与 cut_map 保留原始 panel_no/step，行号与总切长按新程序
重算。会话已完成或 next_action=blade_change 时 409 且不输出部分程序；
版本不符时 409 带当前版本；任何导出都不得改写会话状态。
API 层用 TestClient 做端到端冒烟。需要 fastapi/pydantic 依赖。
"""
import json
import os
import re
import tempfile
import threading
import unittest

from app import database as db
from app.models import (
    BladeChangeRequest,
    CutEventRequest,
    ResumeGcodeRequest,
    StartSessionRequest,
)
from app.services import (
    PlanGeometryError,
    SessionConflictError,
    confirm_blade_change,
    create_plan,
    export_gcode,
    export_resume_gcode,
    flatten_cuts,
    get_session_state,
    start_session,
    submit_cut_event,
)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# 多板方案（与 test_blade_life 同款）：4 块母板、20 刀；前 5 刀切长
# 122, 120, 59, 60, 121
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


def _greq(version, origin="top_left", unit="mm", safe_z=5.0):
    return ResumeGcodeRequest(
        expected_version=version, origin=origin, unit=unit, safe_z=safe_z)


class _DbTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "resume.db")
        db.init_db(self.path)
        self._orig_path = db.DB_PATH
        db.DB_PATH = self.path

    def tearDown(self):
        db.DB_PATH = self._orig_path
        self.tmp.cleanup()

    def _new_session(self, req=REQ, blade_life=None, initial_used=0.0):
        plan = create_plan(_req(req))
        body = None
        if blade_life is not None:
            body = StartSessionRequest(blade_life=blade_life,
                                       initial_used=initial_used)
        session = start_session(plan["plan_id"], body)
        return plan, session

    @staticmethod
    def _event(session, key, version, cut):
        return submit_cut_event(session["session_id"], CutEventRequest(
            expected_version=version,
            idempotency_key=key,
            panel_no=cut["panel_no"],
            step=cut["step"],
        ))


class TestResumeExportContent(_DbTestCase):
    def test_export_at_zero_equals_full_program_with_resume_banner(self):
        plan, sess = self._new_session()
        out = export_resume_gcode(sess["session_id"], _greq(0))
        full = export_gcode(plan["plan_id"], _greq(0))
        body = out["gcode"].splitlines()
        # 头部：续切横幅 + PLAN + SESSION，其余与整板导出完全一致
        self.assertEqual(body[0], "(PANEL-NESTING-API RESUME GCODE EXPORT)")
        self.assertEqual(body[1], f"(PLAN {plan['plan_id']})")
        self.assertEqual(body[2], f"(SESSION {sess['session_id']})")
        self.assertEqual(
            out["gcode"].split("(SESSION", 1)[1].split(")\n", 1)[1],
            full["gcode"].split("(PLAN", 1)[1].split(")\n", 1)[1],
        )
        # cut_map 的刀身份与整板一致；行号因多一行 SESSION 头整体 +1
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in out["cut_map"]],
            [(e["panel_no"], e["step"]) for e in full["cut_map"]])
        self.assertEqual(
            [e["line"] for e in out["cut_map"]],
            [e["line"] + 1 for e in full["cut_map"]])
        self.assertEqual(out["line_count"], full["line_count"] + 1)
        self.assertEqual(out["total_cut_length"], full["total_cut_length"])
        self.assertEqual(out["version"], 0)
        self.assertEqual(out["completed_cuts"], 0)

    def test_starts_at_next_cut_and_keeps_original_ids(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        # 完成前 3 刀（仍在板 1 内）
        for i in range(3):
            sess = self._event(sess, f"e{i}", i, cuts[i])
        out = export_resume_gcode(sess["session_id"], _greq(3))
        expected = [(c["panel_no"], c["step"]) for c in cuts[3:]]
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in out["cut_map"]], expected)
        # 第一个待切母板（板 1）直接进入：程序头之后第一条刀注释即板 1 余刀
        body = out["gcode"].splitlines()
        first_cut = next(ln for ln in body
                         if re.fullmatch(r"\(P\d+ S\d+ [VH]\)", ln))
        self.assertEqual(
            first_cut, f"(P{cuts[3]['panel_no']} S{cuts[3]['step']} "
                       f"{cuts[3]['axis']})")
        # 行号指向新程序中的 G1 切割移动，严格递增
        prev = 0
        for entry in out["cut_map"]:
            self.assertTrue(body[entry["line"] - 1].startswith("G1 X"))
            self.assertGreater(entry["line"], prev)
            prev = entry["line"]
        # 总切长按剩余刀重算
        expect_len = round(sum(float(c["length"]) for c in cuts[3:]), 6)
        self.assertEqual(out["total_cut_length"], expect_len)

    def test_panel_change_pauses_only_between_remaining_panels(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        panel_bounds = {}
        for c in cuts:
            panel_bounds.setdefault(c["panel_no"], 0)
            panel_bounds[c["panel_no"]] += 1
        n_panels = len(panel_bounds)

        # 场景 1：板 1 还剩刀 -> 待切母板仍是全部 4 块，M0 = 3
        sess = self._event(sess, "e0", 0, cuts[0])
        out = export_resume_gcode(sess["session_id"], _greq(1))
        remaining_panels = {c["panel_no"] for c in cuts[1:]}
        self.assertEqual(out["gcode"].count("M0"),
                         len(remaining_panels) - 1)

        # 场景 2：板 1 全部完成 -> 首个待切母板为板 2，M0 = 2（无前置暂停）
        done = panel_bounds[1]
        for i in range(1, done):
            sess = self._event(sess, f"e{i}", i, cuts[i])
        out = export_resume_gcode(sess["session_id"], _greq(done))
        body = out["gcode"].splitlines()
        self.assertEqual(out["gcode"].count("M0"), n_panels - 2)
        self.assertFalse(any("(P1 " in ln for ln in body))
        # 第一个 M0 是板 2 -> 板 3
        first_m0 = next(ln for ln in body if ln.startswith("M0"))
        self.assertIn("CHANGE TO PANEL 3", first_m0)
        # 板头保留原始编号与总板数
        self.assertIn(f"(PANEL 2 OF {n_panels}:", "\n".join(body))

    def test_resume_after_blade_version_uses_done_not_version(self):
        # 换刀确认也占版本号：完成 2 刀 + 1 次换刀后 version=3，但
        # completed_cuts 仍为 2，续切点必须按刀数而不是版本号定位
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sess = self._event(sess, "e0", 0, cuts[0])
        sess = self._event(sess, "e1", 1, cuts[1])
        sess = confirm_blade_change(sess["session_id"], BladeChangeRequest(
            expected_version=2, idempotency_key="b1"))
        self.assertEqual((sess["version"], sess["completed_cuts"]), (3, 2))
        out = export_resume_gcode(sess["session_id"], _greq(3))
        self.assertEqual(out["version"], 3)
        self.assertEqual(out["completed_cuts"], 2)
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in out["cut_map"]][:2],
            [(cuts[2]["panel_no"], cuts[2]["step"]),
             (cuts[3]["panel_no"], cuts[3]["step"])])


class TestResumeExportConflicts(_DbTestCase):
    def test_session_not_found(self):
        self.assertIsNone(
            export_resume_gcode("does-not-exist", _greq(0)))

    def test_stale_version_conflict_returns_current_version(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        self._event(sess, "e0", 0, cuts[0])
        with self.assertRaises(SessionConflictError) as cm:
            export_resume_gcode(sess["session_id"], _greq(0))
        detail = cm.exception.detail
        self.assertEqual(detail["error"], "version_conflict")
        self.assertEqual(detail["expected_version"], 0)
        self.assertEqual(detail["current_version"], 1)

    def test_concurrent_advance_makes_export_return_current_version(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        errors = []

        def worker():
            try:
                export_resume_gcode(sess["session_id"], _greq(0))
            except SessionConflictError as exc:
                errors.append(exc.detail)

        # 导出读快照与并发推进同时进行：读要么发生在提交前（成功），
        # 要么发生在提交后（409 带当前版本 1），绝不能读到半个状态
        t_advance = threading.Thread(
            target=lambda: self._event(sess, "e0", 0, cuts[0]))
        t_export = threading.Thread(target=worker)
        t_advance.start()
        t_export.start()
        t_advance.join()
        t_export.join()
        if errors:
            self.assertEqual(errors[0]["error"], "version_conflict")
            self.assertEqual(errors[0]["current_version"], 1)

    def test_completed_session_409_no_partial_program(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        for i, cut in enumerate(cuts):
            sess = self._event(sess, f"e{i}", i, cut)
        self.assertTrue(sess["completed"])
        with self.assertRaises(SessionConflictError) as cm:
            export_resume_gcode(
                sess["session_id"], _greq(sess["version"]))
        self.assertEqual(cm.exception.detail["error"], "session_completed")

    def test_zero_cut_session_completed_at_start_409(self):
        plan, sess = self._new_session(req=ZERO_CUT_REQ)
        self.assertTrue(sess["completed"])
        with self.assertRaises(SessionConflictError) as cm:
            export_resume_gcode(sess["session_id"], _greq(0))
        self.assertEqual(cm.exception.detail["error"], "session_completed")

    def test_blade_change_required_409_no_partial_program(self):
        # 寿命 300：前两刀累计 242，第三刀 59 会越过 -> 必须先换刀
        plan, sess = self._new_session(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sess = self._event(sess, "e0", 0, cuts[0])
        sess = self._event(sess, "e1", 1, cuts[1])
        self.assertEqual(sess["next_action"], "blade_change")
        with self.assertRaises(SessionConflictError) as cm:
            export_resume_gcode(sess["session_id"], _greq(2))
        detail = cm.exception.detail
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["next_action"], "blade_change")
        self.assertEqual(detail["next_cut"]["panel_no"],
                         cuts[2]["panel_no"])
        self.assertEqual(detail["next_cut"]["step"], cuts[2]["step"])
        self.assertNotIn("gcode", detail)
        # 换刀确认后同一版本（version=3）即可导出
        confirm_blade_change(sess["session_id"], BladeChangeRequest(
            expected_version=2, idempotency_key="b1"))
        out = export_resume_gcode(sess["session_id"], _greq(3))
        self.assertTrue(out["gcode"].rstrip("\n").endswith("M2"))

    def test_invalid_plan_geometry_409(self):
        # 直接落库一份几何越界方案，再为其启动会话
        bad = {
            "status": "complete",
            "summary": {
                "panels_used": 1, "max_boards": 1, "placed_parts": 1,
                "unplaced_parts": 0, "utilization": 0.5,
                "board_width": 100.0, "board_height": 100.0,
                "kerf_width": 3.0,
            },
            "panels": [{
                "panel_no": 1, "width": 100.0, "height": 100.0,
                "placements": [],
                "cuts": [{
                    "step": 1, "axis": "V", "x": 150.0, "y": 0.0,
                    "length": 100.0,
                    "region": {"x": 0.0, "y": 0.0,
                               "width": 100.0, "height": 100.0},
                }],
                "remnants": [],
            }],
            "unplaced": [],
            "plan_id": "bad-plan",
            "created_at": "2026-09-20T00:00:00Z",
        }
        with db.get_conn() as conn:
            db.save_plan("bad-plan", "{}", json.dumps(bad),
                         bad["created_at"], conn)
        sess = start_session("bad-plan")
        with self.assertRaises(PlanGeometryError) as cm:
            export_resume_gcode(sess["session_id"], _greq(0))
        self.assertIn("cut_outside_board",
                      {v["code"] for v in cm.exception.violations})


class TestResumeExportReadOnly(_DbTestCase):
    def _snapshot(self, session_id):
        with db.get_conn() as conn:
            return db.get_session(session_id, conn)

    def test_export_never_mutates_session(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        for i in range(5):
            sess = self._event(sess, f"e{i}", i, cuts[i])
        before = self._snapshot(sess["session_id"])
        for unit, origin, z in [("mm", "top_left", 5.0),
                                ("in", "bottom_right", 0.2),
                                ("mm", "top_right", 1.0)]:
            export_resume_gcode(
                sess["session_id"], _greq(sess["version"], origin, unit, z))
        after = self._snapshot(sess["session_id"])
        for col in ("status", "version", "completed_cuts", "total_cuts",
                    "blade_life", "blade_used", "updated_at",
                    "completed_at"):
            self.assertEqual(before[col], after[col], col)
        # 不写任何事件/换刀/批次记录
        with db.get_conn() as conn:
            sid = sess["session_id"]
            self.assertEqual(db.count_session_events(sid, conn), 5)
            self.assertEqual(db.count_blade_changes(sid, conn), 0)
            self.assertEqual(db.count_session_batches(sid, conn), 0)

    def test_failed_export_never_mutates_session(self):
        plan, sess = self._new_session()
        cuts = flatten_cuts(plan)
        self._event(sess, "e0", 0, cuts[0])
        before = self._snapshot(sess["session_id"])
        for bad_version in (0, 99):
            with self.assertRaises(SessionConflictError):
                export_resume_gcode(
                    sess["session_id"], _greq(bad_version))
        after = self._snapshot(sess["session_id"])
        self.assertEqual(before["version"], after["version"])
        self.assertEqual(before["completed_cuts"], after["completed_cuts"])

    def test_get_session_after_export_unchanged(self):
        plan, sess = self._new_session()
        export_resume_gcode(sess["session_id"], _greq(0))
        state = get_session_state(sess["session_id"])
        self.assertEqual(state["version"], 0)
        self.assertEqual(state["completed_cuts"], 0)


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi，跳过 API 端到端")
class TestResumeGcodeApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmp.name, "resume_e2e.db")
        os.environ["NESTING_DB_PATH"] = cls.db_path
        db.DB_PATH = cls.db_path
        db.init_db(cls.db_path)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _setup(self, blade_life=None):
        r = self.client.post("/api/v1/plans", json=_req(REQ))
        self.assertEqual(r.status_code, 200)
        plan = r.json()
        body = {} if blade_life is None else {"blade_life": blade_life}
        r = self.client.post(
            f"/api/v1/plans/{plan['plan_id']}/sessions", json=body)
        self.assertEqual(r.status_code, 201)
        return plan, r.json()

    def _advance(self, session, cuts, count):
        sid = session["session_id"]
        for i in range(count):
            r = self.client.post(
                f"/api/v1/sessions/{sid}/events",
                json={"expected_version": i,
                      "idempotency_key": f"e{i}",
                      "panel_no": cuts[i]["panel_no"],
                      "step": cuts[i]["step"]})
            self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_resume_endpoint_ok(self):
        plan, session = self._setup()
        cuts = flatten_cuts(plan)
        session = self._advance(session, cuts, 3)
        r = self.client.post(
            f"/api/v1/sessions/{session['session_id']}/resume-gcode",
            json={"expected_version": 3, "origin": "bottom_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["session_id"], session["session_id"])
        self.assertEqual(body["plan_id"], plan["plan_id"])
        self.assertEqual(body["version"], 3)
        self.assertEqual(body["completed_cuts"], 3)
        self.assertEqual(body["origin"], "bottom_left")
        text = body["gcode"]
        self.assertTrue(text.startswith(
            "(PANEL-NESTING-API RESUME GCODE EXPORT)\n"))
        self.assertIn(f"(SESSION {session['session_id']})", text)
        self.assertTrue(text.rstrip("\n").endswith("M2"))
        # 剩余刀全部在 cut_map 中，行号有效
        lines = text.splitlines()
        expected = [(c["panel_no"], c["step"]) for c in cuts[3:]]
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in body["cut_map"]], expected)
        for e in body["cut_map"]:
            self.assertTrue(lines[e["line"] - 1].startswith("G1 X"))
        expect_len = sum(c["length"] for c in cuts[3:])
        self.assertAlmostEqual(body["total_cut_length"], expect_len)
        self.assertEqual(body["line_count"], len(lines))

    def test_resume_endpoint_404_422(self):
        r = self.client.post(
            "/api/v1/sessions/nope/resume-gcode",
            json={"expected_version": 0, "origin": "top_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 404)
        plan, session = self._setup()
        sid = session["session_id"]
        # 缺 expected_version / 非法 origin / safe_z 非正 -> 422
        self.assertEqual(self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"origin": "top_left", "unit": "mm", "safe_z": 5},
        ).status_code, 422)
        self.assertEqual(self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": 0, "origin": "middle",
                  "unit": "mm", "safe_z": 5},
        ).status_code, 422)
        self.assertEqual(self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": -1, "origin": "top_left",
                  "unit": "mm", "safe_z": 5},
        ).status_code, 422)

    def test_resume_endpoint_conflicts(self):
        plan, session = self._setup(blade_life=300.0)
        cuts = flatten_cuts(plan)
        sid = session["session_id"]

        # 旧版本：推进一刀后再以 expected_version=0 导出 -> 409 带当前版本
        self._advance(session, cuts, 1)
        r = self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": 0, "origin": "top_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "version_conflict")
        self.assertEqual(detail["current_version"], 1)
        self.assertNotIn("gcode", r.json())

        # 刀具寿命检查点：再切一刀（累计 242），下一刀 59 越界 -> 409
        r = self.client.post(
            f"/api/v1/sessions/{sid}/events",
            json={"expected_version": 1, "idempotency_key": "e1",
                  "panel_no": cuts[1]["panel_no"], "step": cuts[1]["step"]})
        self.assertEqual(r.status_code, 200)
        r = self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": 2, "origin": "top_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"],
                         "blade_change_required")

        # 换刀确认（version -> 3）后导出成功
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 2, "idempotency_key": "b1"})
        self.assertEqual(r.status_code, 200)
        r = self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": 3, "origin": "top_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["completed_cuts"], 2)

        # 推进到全部完成（途中可能多次撞到寿命检查点，按需先换刀再切）
        version = 3
        blade_no = 2  # b1 已用于前面的换刀确认，续切循环从 b2 起编号
        pos = 2
        while pos < len(cuts):
            cut = cuts[pos]
            rr = self.client.post(
                f"/api/v1/sessions/{sid}/events",
                json={"expected_version": version,
                      "idempotency_key": f"e{pos}",
                      "panel_no": cut["panel_no"], "step": cut["step"]})
            if rr.status_code == 409 and \
                    rr.json()["detail"]["error"] == "blade_life_exceeded":
                rc = self.client.post(
                    f"/api/v1/sessions/{sid}/blade-changes",
                    json={"expected_version": version,
                          "idempotency_key": f"b{blade_no}"})
                self.assertEqual(rc.status_code, 200, rc.text)
                blade_no += 1
                version += 1
                continue
            self.assertEqual(rr.status_code, 200, rr.text)
            version += 1
            pos += 1
        r = self.client.post(
            f"/api/v1/sessions/{sid}/resume-gcode",
            json={"expected_version": version, "origin": "top_left",
                  "unit": "mm", "safe_z": 5})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "session_completed")


if __name__ == "__main__":
    unittest.main()
