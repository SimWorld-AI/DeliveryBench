# actions/switch_transport.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, TransportMode
from ..entities.escooter import ScooterState
from ..utils.util import get_tol, is_at_xy


def handle_switch_transport(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Switches the agent's transport mode.

    Supported transitions:
    - to="walk"
    - to="e-scooter" / "scooter": if an assisting scooter exists, automatically downgrade to DRAG_SCOOTER
    - to="car"
    - to="drag_scooter" / "drag": explicitly switch to dragging mode (prefers assisting scooter; otherwise drag own scooter)

    Rules:
    - Switching to dragging mode does NOT park the scooter beforehand.
    - Switching to other modes:
        * Car: if currently driving, the car must be parked before switching.
        * Own e-scooter: automatically parked unless switching into dragging mode.
    """

    to = str(act.data.get("to", "")).strip().lower()
    tol = get_tol(dm.cfg, "nearby")
    want_drag = to in ("drag_scooter", "drag")

    # --- Clean up current vehicle state ---
    # For cars: always park before switching to any other mode
    if dm.mode == TransportMode.CAR and dm.car:
        dm.car.park_here(dm.x, dm.y)

    # For own scooter: only auto-park when NOT switching into dragging mode
    if not want_drag and dm.e_scooter and getattr(dm.e_scooter, "with_owner", True):
        if dm.mode == TransportMode.SCOOTER or (
            dm.mode == TransportMode.DRAG_SCOOTER and dm.assist_scooter is None
        ):
            dm.e_scooter.park_here(dm.x, dm.y)

    # --- Explicit switch to dragging mode (prefer assisting scooter) ---
    if want_drag:
        # (A) Drag assisting scooter
        if dm.assist_scooter is not None:
            sc = dm.assist_scooter
            if sc.park_xy:
                px, py = sc.park_xy
                if not is_at_xy(dm, px, py, tol_cm=tol):
                    dm.vlm_add_error("switch failed: not near the assisting scooter")
                    dm._finish_action(success=False)
                    return
                try:
                    sc.unpark()
                    dm._charge_ctx = None
                except Exception:
                    sc.park_xy = None
                    dm._charge_ctx = None

            dm.set_mode(TransportMode.DRAG_SCOOTER)
            dm._log("switch -> DRAG_SCOOTER (assisting)")
            dm._finish_action(success=True)
            return

        # (B) Drag own scooter
        if not dm.e_scooter:
            dm.vlm_add_error("switch failed: no scooter to drag")
            dm._finish_action(success=False)
            return

        if not getattr(dm.e_scooter, "with_owner", True):
            dm.vlm_add_error(
                "switch failed: your e-scooter is currently handed off (TempBox). "
                "Retrieve it first."
            )
            dm._finish_action(success=False)
            return

        if dm.e_scooter.park_xy:
            px, py = dm.e_scooter.park_xy
            if not is_at_xy(dm, px, py, tol_cm=tol):
                dm.vlm_add_error("switch failed: not near your scooter")
                dm._finish_action(success=False)
                return
            try:
                dm.e_scooter.unpark()
            except Exception:
                dm.e_scooter.park_xy = None

        dm.set_mode(TransportMode.DRAG_SCOOTER)
        dm._log("switch -> DRAG_SCOOTER")
        dm._finish_action(success=True)
        return

    # --- Walking mode ---
    if to in ("walk", TransportMode.WALK.value):
        dm.set_mode(TransportMode.WALK)
        dm._log("switch -> WALK")
        dm._finish_action(success=True)
        return

    # --- Riding e-scooter: downgrade to dragging if assisting scooter exists ---
    if to in ("e-scooter", "scooter", "escooter", TransportMode.SCOOTER.value):
        if dm.assist_scooter is not None:
            sc = dm.assist_scooter
            if sc.park_xy:
                px, py = sc.park_xy
                if not is_at_xy(dm, px, py, tol_cm=tol):
                    dm.vlm_add_error("switch failed: not near the assisting scooter")
                    dm._finish_action(success=False)
                    return
                try:
                    sc.unpark()
                    dm._charge_ctx = None
                except Exception:
                    sc.park_xy = None
                    dm._charge_ctx = None

            dm.set_mode(TransportMode.DRAG_SCOOTER)
            dm._log("switch -> DRAG_SCOOTER (assist scooter cannot be ridden)")
            dm._finish_action(success=True)
            return

        if not dm.e_scooter:
            dm.vlm_add_error("switch failed: no scooter")
            dm._finish_action(success=False)
            return

        if dm.e_scooter.park_xy:
            px, py = dm.e_scooter.park_xy
            if not is_at_xy(dm, px, py, tol_cm=tol):
                dm.vlm_add_error("switch failed: not near your scooter")
                dm._finish_action(success=False)
                return
            try:
                dm.e_scooter.unpark()
                dm._charge_ctx = None
            except Exception:
                dm.e_scooter.park_xy = None
                dm._charge_ctx = None

        if dm.e_scooter.state == ScooterState.DEPLETED:
            dm.set_mode(TransportMode.DRAG_SCOOTER)
            dm._log("switch -> DRAG_SCOOTER (battery depleted)")
        else:
            dm.set_mode(TransportMode.SCOOTER)
            dm._log("switch -> SCOOTER")

        dm._finish_action(success=True)
        return

    # --- Car mode ---
    if to in ("car", TransportMode.CAR.value):
        if not dm.car:
            dm.vlm_add_error("switch failed: no rented car")
            dm._finish_action(success=False)
            return

        if dm.car.park_xy:
            px, py = dm.car.park_xy
            if not is_at_xy(dm, px, py, tol_cm=tol):
                dm.vlm_add_error("switch failed: not near your car")
                dm._finish_action(success=False)
                return
            dm.car.unpark()

        dm.set_mode(TransportMode.CAR)
        dm._log("switch -> CAR")
        dm._finish_action(success=True)
        return

    # --- Invalid target ---
    dm.vlm_add_error("switch failed: invalid target")
    dm._finish_action(success=False)