# actions/charge_escooter.py
# -*- coding: utf-8 -*-

import os
from typing import Any

from ..base.defs import DMAction, TransportMode
from ..utils.util import nearest_poi_xy, is_at_xy, get_tol
from ..gameplay.comms import get_comms

IS_MULTI_AGENT = (os.getenv("DELIVERYBENCH_MULTI_AGENT", "1") == "1")


def advance_charge_to_now(dm: Any) -> None:
    """
    Advance charging progress based on the stored charge context.
    """
    if dm._charge_ctx and dm._charge_ctx.get("scooter_ref"):
        sc = dm._charge_ctx["scooter_ref"]
        cur = float(dm._charge_ctx.get("paid_pct", getattr(sc, "battery_pct", 0.0)))
        sc.charge_to(cur)


def handle_charge_escooter(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handle initiating e-scooter charging at a nearby charging station.

    In multi-agent setups (IS_MULTI_AGENT = True), this also reserves and
    releases a charging spot via the global comms channel (if available).
    """
    # Already charging: do not start another session.
    if dm._charge_ctx is not None:
        dm.vlm_add_error("charge failed: already charged; don't charge again")
        dm._finish_action(success=False)
        return

    # Must be near a charging station.
    station_xy = nearest_poi_xy(dm, "charging_station", tol_cm=get_tol(dm.cfg, "nearby"))
    if station_xy is None:
        dm.vlm_add_error("charge failed: not near a charging station")
        dm._finish_action(success=False)
        return

    tol = float(act.data.get("tol_cm", get_tol(dm.cfg, "nearby")))

    def _with_me(s):
        """Check whether the scooter is physically with the agent (ridden or dragged)."""
        if not s:
            return False
        if dm.mode == TransportMode.SCOOTER:
            return s is dm.e_scooter
        if dm.mode == TransportMode.DRAG_SCOOTER:
            return (
                (dm.assist_scooter is not None and s is dm.assist_scooter)
                or (dm.assist_scooter is None and s is dm.e_scooter)
            )
        return False

    def _parked_nearby(s):
        """Check whether the scooter is parked near the agent within tolerance."""
        return bool(s and s.park_xy and is_at_xy(dm, s.park_xy[0], s.park_xy[1], tol_cm=tol))

    # Select scooters that are either with the agent or parked close by.
    # Assisted scooter has priority.
    candidates = []
    if dm.assist_scooter:
        candidates.append(("assist", dm.assist_scooter))
    if dm.e_scooter and getattr(dm.e_scooter, "with_owner", True):
        candidates.append(("own", dm.e_scooter))

    sc, which, with_me = None, None, False
    for kind, s in candidates:
        if _with_me(s) or _parked_nearby(s):
            sc, which, with_me = s, kind, _with_me(s)
            break

    if not sc:
        dm.vlm_add_error("charge failed: scooter not with you and not parked nearby")
        dm._finish_action(success=False)
        return

    # If the scooter is physically with the agent, park it at the current position.
    # If already parked nearby, ensure the agent is actually at the scooter's parked location.
    if with_me:
        sc.park_here(dm.x, dm.y)
        if dm.mode in (TransportMode.SCOOTER, TransportMode.DRAG_SCOOTER):
            dm.set_mode(TransportMode.WALK)
    else:
        px, py = sc.park_xy
        if not is_at_xy(dm, px, py, tol_cm=tol):
            dm.vlm_add_error("charge failed: not at parked scooter location")
            dm._finish_action(success=False)
            return

    # === Multi-agent: reserve a charging spot via comms (if enabled) ===
    comms = None
    station_key = None
    print("IS_MULTI_AGENT =", IS_MULTI_AGENT)
    if IS_MULTI_AGENT:
        comms = get_comms()
        if comms and hasattr(comms, "reserve_charging_spot"):
            ok, msg, station_key = comms.reserve_charging_spot(station_xy, str(dm.agent_id))
            print(f"Reserving charging spot at {station_xy} for agent {dm.agent_id}: {ok}, {msg}")
            if not ok:
                dm.vlm_add_error(f"charge failed: {msg}")
                dm._finish_action(success=False)
                return

    # Target battery percentage (clamped to [0, 100]).
    target_pct = float(
        act.data.get(
            "target_pct",
            dm.cfg.get("escooter_defaults", {}).get(
                "charge_target_pct",
                dm.cfg.get("defaults", {}).get("charge_target_pct", 100.0),
            ),
        )
    )
    target_pct = max(0.0, min(100.0, target_pct))

    before = float(sc.battery_pct)
    if target_pct <= before + 1e-6:
        # Nothing to charge; release spot if we reserved one.
        if IS_MULTI_AGENT and station_key is not None and comms and hasattr(
            comms, "release_charging_spot"
        ):
            comms.release_charging_spot(station_key, agent_id=str(dm.agent_id))
        dm._finish_action(success=True)
        return

    rate_m = float(sc.charge_rate_pct_per_min)
    if rate_m <= 0.0:
        # Invalid charge rate; release spot if we reserved one.
        if IS_MULTI_AGENT and station_key is not None and comms and hasattr(
            comms, "release_charging_spot"
        ):
            comms.release_charging_spot(station_key, agent_id=str(dm.agent_id))
        dm.vlm_add_error("charge failed: invalid rate")
        dm._finish_action(success=False)
        return

    duration_sim_s = (target_pct - before) / rate_m * 60.0
    now_sim = dm.clock.now_sim()

    # Record charging context for future updates.
    ctx = dict(
        start_sim=now_sim,
        end_sim=now_sim + duration_sim_s,
        start_pct=before,
        target_pct=target_pct,
        paid_pct=before,
        price_per_pct=float(dm.charge_price_per_pct),
        scooter_ref=sc,
        which=("assist" if which == "assist" else "own"),
        park_xy_start=tuple(sc.park_xy) if sc.park_xy else None,
        station_xy=tuple(station_xy),
    )
    if IS_MULTI_AGENT:
        ctx["station_key"] = station_key

    dm._charge_ctx = ctx

    dm._log(
        f"start charging scooter ({'assist' if which=='assist' else 'own'}): "
        f"{before:.0f}% -> {target_pct:.0f}% (~{duration_sim_s/60.0:.1f} min @virtual)"
    )
    dm._recorder.inc_preventive("early_charge")
    dm._recorder.tick_inactive("wait", duration_sim_s)
    dm._finish_action(success=True)