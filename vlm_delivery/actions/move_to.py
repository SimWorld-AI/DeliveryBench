# actions/move_to.py
# -*- coding: utf-8 -*-

import math
from typing import Any

from ..base.defs import DMAction
from ..utils.util import get_tol, is_at_xy, fmt_xy_m, fmt_xy_m_opt


def _estimate_distance_cm(
    dm: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    use_route: bool,
    snap_cm: float,
) -> float:
    """
    Estimates travel distance (in cm) between two points.
    If routing is enabled and supported by the map, uses routed path length.
    Otherwise uses straight-line distance.
    """
    if use_route and hasattr(dm.city_map, "route_xy_to_xy_mode"):
        pts = dm.city_map.route_xy_to_xy_mode(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            snap_cm=float(snap_cm),
            mode=dm.mode.value,
        ) or []
        if len(pts) >= 2:
            dist = 0.0
            for i in range(len(pts) - 1):
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
                dist += math.hypot(dx, dy)
            return float(dist)

    return float(math.hypot(x1 - x0, y1 - y0))


def handle_move_to(dm: Any, act: DMAction, allow_interrupt: bool) -> None:
    """
    Handles movement from the agent’s current position to a target coordinate.
    """
    dm.vlm_clear_ephemeral()
    dm._interrupt_move_flag = False

    sx, sy = float(dm.x), float(dm.y)
    tx, ty = float(act.data.get("tx", dm.x)), float(act.data.get("ty", dm.y))
    tol = float(act.data.get("arrive_tolerance_cm", get_tol(dm.cfg, "nearby")))

    # Complete immediately if already at destination
    if is_at_xy(dm, tx, ty, tol_cm=tol):
        dm._log(f"already at target location {fmt_xy_m(tx, ty)}")
        dm.vlm_add_ephemeral(
            "location_status",
            f"already at target location {fmt_xy_m_opt((tx, ty))}, choose a new action",
        )
        dm._finish_action(success=True)
        return

    # Process pace settings
    pace = str(act.data.get("pace", "normal")).strip().lower()
    if pace not in ("accel", "decel", "normal"):
        pace = "normal"
    dm.pace_state = pace

    if pace == "accel" and dm.insulated_bag:
        dm.insulated_bag.bump_motion_damage(inc=1)

    dm._recalc_towing()

    # Estimate distance if not provided
    if "expected_dist_cm" not in act.data:
        act.data["expected_dist_cm"] = _estimate_distance_cm(
            dm,
            sx,
            sy,
            tx,
            ty,
            bool(act.data.get("use_route", True)),
            float(act.data.get("snap_cm", 120.0)),
        )

    mode_str = "towing" if dm.towing_scooter else dm.mode.value
    speed_to_use = float(dm.speed_cm_s) * dm._pace_scale()

    dm._log(
        f"move from {fmt_xy_m(sx, sy)} to {fmt_xy_m(tx, ty)} "
        f"[mode={mode_str}, speed={speed_to_use:.1f} cm/s, pace={dm.pace_state}]"
    )

    # Route generation
    if hasattr(dm.city_map, "route_xy_to_xy_mode"):
        route = dm.city_map.route_xy_to_xy_mode(
            float(sx),
            float(sy),
            float(tx),
            float(ty),
            snap_cm=float(120),
            mode=dm.mode.value,
        ) or []
    else:
        route = [(sx, sy), (tx, ty)]

    # Movement context for polling-based progress tracking
    now = dm.clock.now_sim()
    dm._move_ctx = {
        "tx": float(tx),
        "ty": float(ty),
        "tol": float(tol),
        "blocked": 0.0,
        "last_position": (float(sx), float(sy)),
        "last_position_time": now,
        "stagnant_time": 0.0,
        "stagnant_threshold": 60.0,
    }

    # Viewer visualization
    if dm._viewer and dm._viewer_agent_id and hasattr(dm._viewer, "go_to_xy"):
        dm._viewer.go_to_xy(
            dm._viewer_agent_id,
            route,
            allow_interrupt=allow_interrupt,
            show_path_ms=2000,
        )

    # UE backend async movement
    if dm._ue and hasattr(dm._ue, "go_to_xy_async"):
        dm._ue.go_to_xy_async(
            dm._viewer_agent_id,
            route,
            speed_cm_s=dm.get_current_speed_for_viewer(),
            accel_cm_s2=None,
            decel_cm_s2=None,
            arrive_tolerance_cm=tol,
        )