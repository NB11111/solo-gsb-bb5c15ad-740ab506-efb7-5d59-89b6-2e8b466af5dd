"""按切割会话进度导出续切 G-code 的测试。

覆盖：
- generate_resume_gcode 纯渲染：续切起点、跨板暂停一次、首板直接进入、
  原始 panel_no/step 注释与 cut_map、行号/总长重算、确定性；
- 服务层（只读快照事务）：404、正常续切、409 已完成/版本不符（回传当前
  版本）/需先换刀、几何违规 409 不输出部分程序、导出不推进会话状态、
  并发推进后旧版本 409；
- HTTP 端到端（fastapi 可用时）。
"""
import json
import os
import tempfile
import threading
import unittest

from app.gcode import MM_PER_INCH, generate_gcode, generate_resume_gcode

from app import database as db
from app.models import CutEventRequest, ResumeGcodeRequest, StartSessionRequest
from app.services import (
    PlanGeometryError,
    SessionConflictError,
    create_plan,
    export_gcode,
    export_session_gcode,
    flatten_cuts,
    start_session,
    submit_cut_event,
)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# 多板方案（与 test_sessions.REQ 同构：4 块母板上限，实际使用多板）
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


def make_synthetic_plan():
    """两块母板：板 1 两刀（V 后 H），板 2 一刀（H）。与 test_gcode 同构。"""
    return {
        "plan_id": "synthetic",
        "status": "complete",
        "summary": {
            "panels_used": 2, "max_boards": 2, "placed_parts": 3,
            "unplaced_parts": 0, "utilization": 0.5,
            "board_width": 244.0, "board_height": 122.0, "kerf_width": 3.0,
        },
        "panels": [
            {
                "panel_no": 1, "width": 244.0, "height": 122.0,
                "placements": [],
                "cuts": [
                    {"step": 1, "axis": "V", "x": 120.0, "y": 0.0,
                     "length": 122.0,
                     "region": {"x": 0.0, "y": 0.0,
                                "width": 244.0, "height": 122.0}},
                    {"step": 2, "axis": "H", "x": 0.0, "y": 60.0,
                     "length": 120.0,
                     "region": {"x": 0.0, "y": 0.0,
                                "width": 120.0, "height": 122.0}},
                ],
                "remnants": [],
            },
            {
                "panel_no": 2, "width": 244.0, "height": 122.0,
                "placements": [],
                "cuts": [
                    {"step": 1, "axis": "H", "x": 0.0, "y": 40.0,
                     "length": 244.0,
                     "region": {"x": 0.0, "y": 0.0,
                                "width": 244.0, "height": 122.0}},
                ],
                "remnants": [],
            },
        ],
        "unplaced": [],
    }


class TestResumeRender(unittest.TestCase):
    def setUp(self):
        self.plan = make_synthetic_plan()

    def test_resume_from_start_equals_full_bodies_but_new_header(self):
        full = generate_gcode(self.plan, "synthetic", "bottom_left", "mm", 5.0)
        r0 = generate_resume_gcode(self.plan, "synthetic", "s1", 0,
                                   "bottom_left", "mm", 5.0)
        # completed_cuts=0 时，从第一块母板的板头起与全量导出逐行相同
        marker = "(PANEL 1 OF 2: 244.000 X 122.000 mm)"
        fb = full["gcode"].splitlines()
        rb = r0["gcode"].splitlines()
        self.assertEqual(rb[rb.index(marker):], fb[fb.index(marker):])
        # 续切头部标识
        self.assertEqual(rb[0], "(PANEL-NESTING-API GCODE RESUME EXPORT)")
        self.assertIn("(SESSION s1)", rb)
        self.assertIn("(RESUME AFTER COMPLETED_CUTS 0)", rb)

    def test_resume_mid_panel_no_leading_pause(self):
        # 板 1 已完成 1 刀：续切首块即板 1（直接进入，无前置 M0），
        # 跨到板 2 前恰有一次 M0。
        out = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                    "bottom_left", "mm", 5.0)
        body = out["gcode"].splitlines()
        # 程序头之后直接是板 1 的板头，前面没有任何 M0
        i_init = body.index("G0 Z5.000")
        self.assertEqual(body[i_init + 1],
                         "(PANEL 1 OF 2: 244.000 X 122.000 mm)")
        self.assertTrue(all(not ln.startswith("M0")
                            for ln in body[:i_init + 1]))
        # 恰有一次跨板暂停，指向板 2
        m0 = [ln for ln in body if ln.startswith("M0")]
        self.assertEqual(m0, ["M0 (CHANGE TO PANEL 2)"])
        # 板 1 仅剩第 2 刀，注释保留原始 step
        self.assertIn("(P1 S2 H)", body)
        self.assertNotIn("(P1 S1", body)
        self.assertIn("(P2 S1 H)", body)
        self.assertEqual(body[-1], "M2")

    def test_resume_at_panel_boundary_first_panel_direct(self):
        # 板 1 两刀全部完成：首块即板 2，直接进入且整个程序无任何 M0
        out = generate_resume_gcode(self.plan, "synthetic", "s1", 2,
                                    "top_left", "mm", 5.0)
        body = out["gcode"].splitlines()
        self.assertNotIn("M0", body)
        self.assertIn("(P2 S1 H)", body)
        self.assertNotIn("(P1 ", body)
        i_init = body.index("G0 Z5.000")
        self.assertEqual(body[i_init + 1],
                         "(PANEL 2 OF 2: 244.000 X 122.000 mm)")

    def test_cut_map_keeps_original_ids_but_lines_are_recomputed(self):
        out = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                    "bottom_left", "mm", 5.0)
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in out["cut_map"]],
            [(1, 2), (2, 1)],
        )
        body = out["gcode"].splitlines()
        prev = 0
        for entry in out["cut_map"]:
            # 行号从 1 起、严格递增且指向新程序内的 G1 切割移动行
            self.assertTrue(body[entry["line"] - 1].startswith("G1 X"))
            self.assertGreater(entry["line"], prev)
            prev = entry["line"]
        # 行号与全量导出不同（按新程序重算）
        full = generate_gcode(self.plan, "synthetic", "bottom_left", "mm", 5.0)
        self.assertNotEqual(
            [e["line"] for e in out["cut_map"]],
            [e["line"] for e in full["cut_map"]],
        )

    def test_total_length_recomputed_over_remaining_cuts(self):
        # 续切只统计剩余刀：板1S2=120 + 板2S1=244
        out = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                    "top_left", "mm", 5.0)
        self.assertEqual(out["total_cut_length"], 120.0 + 244.0)
        self.assertEqual(out["line_count"], len(out["gcode"].splitlines()))
        # in：按 ÷25.4 换算
        out_in = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                       "top_left", "in", 0.2)
        self.assertAlmostEqual(
            out_in["total_cut_length"],
            round((120.0 + 244.0) / MM_PER_INCH, 6))
        self.assertIn("G20", out_in["gcode"])

    def test_deterministic(self):
        a = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                  "bottom_left", "mm", 5.0)
        b = generate_resume_gcode(self.plan, "synthetic", "s1", 1,
                                  "bottom_left", "mm", 5.0)
        self.assertEqual(a, b)

    def test_no_remaining_cuts_raises(self):
        with self.assertRaises(ValueError):
            generate_resume_gcode(self.plan, "synthetic", "s1", 3,
                                  "bottom_left", "mm", 5.0)


class _DbTestCase(unittest.TestCase):
    """每个用例一个独立临时库，并把模块级 DB_PATH 指过去。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "resume_gcode.db")
        db.init_db(self.path)
        self._orig_path = db.DB_PATH
        db.DB_PATH = self.path

    def tearDown(self):
        db.DB_PATH = self._orig_path
        self.tmp.cleanup()

    @staticmethod
    def _event(session, key, version, cut):
        return submit_cut_event(session["session_id"], CutEventRequest(
            expected_version=version,
            idempotency_key=key,
            panel_no=cut["panel_no"],
            step=cut["step"],
        ))

    def _plan_session(self, req=REQ):
        plan = create_plan(json.loads(json.dumps(req)))
        session = start_session(plan["plan_id"])
        return plan, session


class TestResumeService(_DbTestCase):
    def _req(self, version, **kw):
        kw.setdefault("origin", "bottom_left")
        kw.setdefault("unit", "mm")
        kw.setdefault("safe_z", 5.0)
        return ResumeGcodeRequest(expected_version=version, **kw)

    def test_unknown_session_returns_none(self):
        self.assertIsNone(export_session_gcode(
            "00000000-0000-0000-0000-000000000000", self._req(0)))

    def test_export_remaining_from_progress(self):
        plan, sess = self._plan_session()
        cuts = flatten_cuts(plan)
        sid = sess["session_id"]
        # 完成前 2 刀
        s1 = self._event(sess, "e1", 0, cuts[0])
        s2 = self._event(s1, "e2", s1["version"], cuts[1])
        done = s2["completed_cuts"]
        self.assertEqual(done, 2)

        out = export_session_gcode(sid, self._req(s2["version"]))
        remaining = cuts[done:]
        # cut_map 覆盖且仅覆盖剩余刀，保留原始 panel_no/step
        self.assertEqual(
            [(e["panel_no"], e["step"]) for e in out["cut_map"]],
            [(c["panel_no"], c["step"]) for c in remaining],
        )
        # 总长按剩余刀重算
        self.assertAlmostEqual(
            out["total_cut_length"], sum(c["length"] for c in remaining))
        # 行号自洽
        body = out["gcode"].splitlines()
        self.assertEqual(out["line_count"], len(body))
        for e in out["cut_map"]:
            self.assertTrue(body[e["line"] - 1].startswith("G1 X"))
        # 跨板暂停次数 = 剩余刀涉及的不同母板数 - 1
        panels = []
        for c in remaining:
            if not panels or panels[-1] != c["panel_no"]:
                panels.append(c["panel_no"])
        self.assertEqual(sum(ln.startswith("M0") for ln in body),
                         len(panels) - 1)
        # 回显字段
        self.assertEqual(out["session_id"], sid)
        self.assertEqual(out["plan_id"], plan["plan_id"])
        self.assertEqual(out["completed_cuts"], done)
        self.assertEqual(out["version"], s2["version"])

    def test_export_does_not_advance_session(self):
        plan, sess = self._plan_session()
        before = dict(sess)
        export_session_gcode(sess["session_id"], self._req(sess["version"]))
        # 再次导出相同版本仍成功；会话状态逐字段不变
        export_session_gcode(sess["session_id"], self._req(sess["version"]))
        from app.services import get_session_state
        after = get_session_state(sess["session_id"])
        self.assertEqual(after, before)

    def test_stale_version_conflict_returns_current_version(self):
        plan, sess = self._plan_session()
        s1 = self._event(sess, "e1", 0, flatten_cuts(plan)[0])
        # 客户端拿着旧版本 0 导出
        with self.assertRaises(SessionConflictError) as cm:
            export_session_gcode(sess["session_id"], self._req(0))
        self.assertEqual(cm.exception.detail["error"], "version_conflict")
        self.assertEqual(cm.exception.detail["expected_version"], 0)
        self.assertEqual(cm.exception.detail["current_version"], s1["version"])

    def test_completed_session_409_no_partial_program(self):
        plan, sess = self._plan_session()
        cuts = flatten_cuts(plan)
        state = sess
        for i, cut in enumerate(cuts):
            state = self._event(state, f"e{i}", state["version"], cut)
        self.assertTrue(state["completed"])
        with self.assertRaises(SessionConflictError) as cm:
            export_session_gcode(
                sess["session_id"], self._req(state["version"]))
        self.assertEqual(cm.exception.detail["error"], "session_completed")
        self.assertNotIn("gcode", cm.exception.detail)

    def test_blade_change_required_409(self):
        plan = create_plan(json.loads(json.dumps(REQ)))
        cuts = flatten_cuts(plan)
        # 寿命设为：第一刀长度恰好等于寿命 -> 切完第一刀后 next_action
        # 必为 blade_change（第二刀必越过）
        life = float(cuts[0]["length"])
        sess = start_session(plan["plan_id"],
                             StartSessionRequest(blade_life=life))
        s1 = self._event(sess, "e1", 0, cuts[0])
        self.assertEqual(s1["next_action"], "blade_change")
        with self.assertRaises(SessionConflictError) as cm:
            export_session_gcode(sess["session_id"], self._req(s1["version"]))
        detail = cm.exception.detail
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["next_action"], "blade_change")
        self.assertEqual(detail["current_version"], s1["version"])
        self.assertEqual(detail["next_cut"]["panel_no"], cuts[1]["panel_no"])
        self.assertEqual(detail["next_cut"]["step"], cuts[1]["step"])
        self.assertNotIn("gcode", detail)

    def test_after_blade_change_export_allowed(self):
        plan = create_plan(json.loads(json.dumps(REQ)))
        cuts = flatten_cuts(plan)
        life = float(cuts[0]["length"])
        sess = start_session(plan["plan_id"],
                             StartSessionRequest(blade_life=life))
        s1 = self._event(sess, "e1", 0, cuts[0])
        # 换刀确认（服务层直接走 confirm_blade_change）
        from app.models import BladeChangeRequest
        from app.services import confirm_blade_change
        s2 = confirm_blade_change(sess["session_id"], BladeChangeRequest(
            expected_version=s1["version"], idempotency_key="b1"))
        self.assertEqual(s2["next_action"], "cut")
        out = export_session_gcode(sess["session_id"], self._req(s2["version"]))
        self.assertEqual(
            (out["cut_map"][0]["panel_no"], out["cut_map"][0]["step"]),
            (cuts[1]["panel_no"], cuts[1]["step"]))

    def test_geometry_violation_409_no_partial_output(self):
        # 落库一份几何违规的方案并构造指向它的会话行
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
            export_session_gcode(sess["session_id"], self._req(0))
        codes = {v["code"] for v in cm.exception.violations}
        self.assertIn("cut_outside_board", codes)

    def test_concurrent_progress_makes_stale_export_409(self):
        plan, sess = self._plan_session()
        cut = flatten_cuts(plan)[0]
        committed = threading.Event()
        err = []

        def export_stale():
            committed.wait(5)
            try:
                export_session_gcode(
                    sess["session_id"], self._req(sess["version"]))
            except SessionConflictError as exc:
                err.append(exc.detail["error"])
            except Exception as exc:  # noqa: BLE001
                err.append(repr(exc))

        t = threading.Thread(target=export_stale)
        t.start()
        self._event(sess, "e1", 0, cut)  # 先把版本推进到 1 并提交
        committed.set()
        t.join()
        self.assertEqual(err, ["version_conflict"])


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi，跳过 API 端到端")
class TestResumeApi(_DbTestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_api_full_flow(self):
        plan, sess = self._plan_session()
        sid = sess["session_id"]
        cuts = flatten_cuts(plan)
        # 推进 1 刀
        s1 = self._event(sess, "e1", 0, cuts[0])

        payload = {"expected_version": s1["version"],
                   "origin": "bottom_left", "unit": "mm", "safe_z": 5}

        # 200：续切成功
        r = self.client.post(f"/api/v1/sessions/{sid}/gcode", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["session_id"], sid)
        self.assertEqual(body["version"], s1["version"])
        self.assertEqual(body["completed_cuts"], 1)
        self.assertTrue(body["gcode"].rstrip("\n").endswith("M2"))
        self.assertIn("RESUME EXPORT", body["gcode"])
        # 剩余刀全部在 cut_map 中
        self.assertEqual(
            len(body["cut_map"]), len(cuts) - 1)

        # 409：旧版本，回传当前版本
        r_stale = self.client.post(
            f"/api/v1/sessions/{sid}/gcode",
            json={**payload, "expected_version": 0})
        self.assertEqual(r_stale.status_code, 409)
        self.assertEqual(r_stale.json()["detail"]["error"],
                         "version_conflict")
        self.assertEqual(
            r_stale.json()["detail"]["current_version"], s1["version"])

        # 404：不存在的会话
        r404 = self.client.post(
            "/api/v1/sessions/00000000-0000-0000-0000-000000000000/gcode",
            json=payload)
        self.assertEqual(r404.status_code, 404)

        # 422：缺 expected_version / 非法 origin / safe_z
        self.assertEqual(self.client.post(
            f"/api/v1/sessions/{sid}/gcode",
            json={"origin": "bottom_left", "unit": "mm", "safe_z": 5},
        ).status_code, 422)
        self.assertEqual(self.client.post(
            f"/api/v1/sessions/{sid}/gcode",
            json={**payload, "origin": "middle"},
        ).status_code, 422)

        # 把会话切完 -> 409 session_completed，且无 gcode 字段
        state = s1
        for i, cut in enumerate(cuts[1:], start=1):
            state = self._event(state, f"e{i}", state["version"], cut)
        r_done = self.client.post(
            f"/api/v1/sessions/{sid}/gcode",
            json={**payload, "expected_version": state["version"]})
        self.assertEqual(r_done.status_code, 409)
        self.assertEqual(r_done.json()["detail"]["error"],
                         "session_completed")
        self.assertNotIn("gcode", r_done.json())

        # 导出是只读的：会话仍可正常查询
        r_get = self.client.get(f"/api/v1/sessions/{sid}")
        self.assertEqual(r_get.status_code, 200)
        self.assertTrue(r_get.json()["completed"])


if __name__ == "__main__":
    unittest.main()
