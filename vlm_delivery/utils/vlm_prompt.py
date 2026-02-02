# utils/vlm_prompt.py
# -*- coding: utf-8 -*-

"""
Prompt-building utilities for VLM-based DeliveryMan agents.

This module assembles textual state descriptions, POI hints, map summaries,
and help-related context into a structured prompt for a vision–language model.
It also maintains lightweight, ephemeral hint fields on the agent for
contextual guidance (e.g., pickup, charging, rest, TempBox, drop-off).
"""

import math
import os
from typing import Any, List, Tuple, Optional

from .util import (
    xy_of_node,
    is_at_xy,
    nearest_poi_xy,
    fmt_xy_m,
    fmt_xy_m_opt,
    remaining_range_m,
    fmt_time,
    get_tol,
)
from .trajectory_recorder import save_text

from ..gameplay.comms import get_comms, HelpType
from ..entities.escooter import ScooterState


# ===== Agent state & basic text blocks =====

def vlm_agent_state_text(agent: Any) -> str:
    """
    Build a compact, VLM-friendly description of the agent's current state.

    This includes:
        - Agent ID, total number of agents
        - Transport mode, position, speed, pace, energy, earnings
        - Active orders, help orders, carried orders
        - Rest recovery rate
        - Inventory summary
        - Personal e-scooter status and ongoing charging sessions
        - Assisted scooter and rental car information
    """
    active_ids = [getattr(o, "id", None) for o in (agent.active_orders or []) if getattr(o, "id", None) is not None]
    help_ids   = list(getattr(agent, "help_orders", {}).keys())
    carrying_ids = list(getattr(agent, "carrying", []))

    mode_str = "towing a scooter" if getattr(agent, "towing_scooter", False) else agent.mode.value
    speed_ms = agent.speed_cm_s / 100.0

    # Current simulation time (formatted but not currently shown in text)
    current_time = agent._recorder.active_elapsed_s
    time_str = fmt_time(current_time)
    
    lines: List[str] = []
    lines.append(f"You are Agent {agent.agent_id}. There are {agent.cfg.get('agent_count', 0)} delivery agents in total in this city.")
    lines.append(f"Current time is {time_str}.")
    lines.append(f"Your current transport mode is {mode_str}, at {fmt_xy_m(agent.x, agent.y)}.")
    lines.append(f"Your speed is ~{speed_ms:.1f} m/s, energy is {agent.energy_pct:.0f}%.")
    pace_map = {"accel": "accelerating", "normal": "normal", "decelerating": "decelerating"}
    lines.append(f"Your current pace is {pace_map.get(agent.pace_state,'normal')} (×{agent._pace_scale():.2f}).")
    lines.append(f"Earnings is ${agent.earnings_total:.2f}.")
    if active_ids:
        lines.append(f"Active orders: {', '.join(map(str, active_ids))}.")
    if help_ids:
        lines.append(f"Helping orders: {', '.join(map(str, help_ids))}.")
    if carrying_ids:
        lines.append(f"Carrying: {', '.join(map(str, carrying_ids))}.")
    lines.append(f"Rest energy recovery rate is +{agent.rest_rate_pct_per_min:.1f}%/min.")

    # Inventory
    inv = getattr(agent, "inventory", None)
    if inv:
        inv_str = ", ".join([f"{k} x{int(v)}" for k, v in inv.items() if int(v) > 0]) or "empty"
    else:
        inv_str = "empty"
    lines.append(f"Inventory: {inv_str}.")

    # Own e-scooter
    if agent.e_scooter:
        rng_m = remaining_range_m(agent)
        park_str = (
            f"parked at {fmt_xy_m_opt(agent.e_scooter.park_xy)}"
            if agent.e_scooter.park_xy else "not parked"
        )
        if rng_m is not None:
            lines.append(
                f"Scooter: {agent.e_scooter.state.value}, batt {agent.e_scooter.battery_pct:.0f}%, "
                f"range {rng_m:.1f} m,"
            )
        else:
            lines.append(
                f"Scooter: {agent.e_scooter.state.value}, batt {agent.e_scooter.battery_pct:.0f}%, range N/A,"
            )
        lines.append(f"{park_str}.")

    # Ongoing scooter charging session
    if getattr(agent, "_charge_ctx", None):
        ctx = agent._charge_ctx
        sc = ctx.get("scooter_ref") or agent.e_scooter
        if sc:
            cur  = float(ctx.get("paid_pct", getattr(sc, "battery_pct", 0.0)))
            pt   = float(ctx.get("target_pct", 100.0))
            spot = fmt_xy_m_opt(getattr(sc, "park_xy", None))
            which = ctx.get("which", "own")
            lines.append(
                f"Charging in progress ({'assist' if which=='assist' else 'own'}): "
                f"{cur:.0f}% → {pt:.0f}% at {spot}."
            )
            lines.append(f"Charge rate is {agent.e_scooter.charge_rate_pct_per_min:.1f}%/min.")

    # Assisted e-scooter (charging for another agent)
    if agent.assist_scooter:
        asc = agent.assist_scooter
        owner = getattr(asc, "owner_id", "?")
        rng_m = float(asc.battery_pct) / max(1e-9, agent.scooter_batt_decay_pct_per_m)
        park_str = (
            f"parked at {fmt_xy_m_opt(asc.park_xy)}"
            if asc.park_xy else "not parked"
        )
        lines.append(
            f"Assisting scooter (owner agent {owner}), batt {asc.battery_pct:.0f}%, "
            f"range {rng_m:.1f} m, {park_str}."
        )

    # TempBox information for the agent's scooter
    try:
        comms = get_comms()
        if comms:
            my_active = comms.list_my_posts_active(str(agent.agent_id))
            for r in my_active:
                if r.kind == HelpType.HELP_CHARGE:
                    info = comms.get_temp_box_info(int(r.req_id)) or {}
                    pub_box = info.get("publisher_box", {})
                    if pub_box.get("xy"):
                        lines.append(
                            f"Your scooter is placed in TempBox at {fmt_xy_m_opt(pub_box['xy'])}. "
                            f"You can TAKE_FROM_TEMP_BOX(req_id={int(r.req_id)}) to reclaim it when ready."
                        )
    except Exception:
        pass

    # Rental car information
    if agent.car:
        park_str = fmt_xy_m_opt(agent.car.park_xy) if agent.car.park_xy else "N/A"
        lines.append(
            f"Car {agent.car.state.value}, rate ${agent.car.rate_per_min:.2f}/min, "
            f"park_xy={park_str}, rental={'on' if agent._rental_ctx else 'off'}."
        )

    return " ".join(lines)


def vlm_build_bag_place_hint(agent: Any) -> str:
    """
    Build an instruction block for arranging unplaced food items into the bag.

    The text includes:
        - A short description of the task.
        - Unplaced items per order.
        - Current bag layout (if available).
        - Example commands for single and multiple orders.
    """
    lines = [
        "You have UNPLACED FOOD items that must be arranged into the insulated bag.",
        "Output ONE combined bag_cmd per order, e.g.:",
        "  order <id>: 1,2 -> A; 3 -> B",
        ""
    ]
    # Unplaced items per order
    for oid, items in (getattr(agent, "_pending_food_by_order", {}) or {}).items():
        lines.append(f"- Order #{int(oid)} items:")
        if items:
            for i, it in enumerate(items, start=1):
                name = (
                    getattr(it, "name", None)
                    or getattr(it, "title", None)
                    or getattr(it, "label", None)
                    or str(it)
                )
                lines.append(f"    {i}. {name}")
        else:
            lines.append("    (none)")

    # Current bag layout
    lines += ["", "Current bag layout:"]
    if agent.insulated_bag:
        lines.append(agent.insulated_bag.list_items())
    else:
        lines.append("(no bag)")

    lines += [
        "",
        "Example (single order):",
        "  order 12: 1,2 -> A; 3 -> B",
        "",
        "Example (multiple orders):",
        "  order 2: 1,2 -> A",
        "  order 3: 1,2,3,4 -> B",
    ]
    return "\n".join(lines)


def vlm_build_pickup_arrival_hint(
    agent: Any,
    ready_orders: List[Any],
    waiting_pairs: List[Tuple[Any, int]],
) -> str:
    """
    Build a human-readable hint when the agent is at a pickup location.

    Args:
        ready_orders: Orders that are prepared and ready to pick up.
        waiting_pairs: List of (order_obj, mins_remaining) for orders still cooking.

    Returns:
        A multi-line string hint summarizing ready and pending orders,
        plus an example PICKUP command if applicable.
    """
    lines: List[str] = []

    # Try to use the first order's pickup road name as the place label
    place = None
    if ready_orders:
        place = getattr(ready_orders[0], "pickup_road_name", None) or place
    if place is None and waiting_pairs:
        place = getattr(waiting_pairs[0][0], "pickup_road_name", None) or place

    if place:
        lines.append(f"You have arrived at the pickup location: {place}.")
    else:
        lines.append("You are near a pickup location.")

    if ready_orders:
        ready_ids = [getattr(o, "id", None) for o in ready_orders]
        ready_ids = [f"#{oid}" for oid in ready_ids if oid is not None]
        if ready_ids:
            lines.append(f"Orders ready for pickup: {', '.join(ready_ids)}")

    if waiting_pairs:
        waiting_texts = []
        for (o, mins) in waiting_pairs:
            oid = getattr(o, "id", None)
            if oid is not None:
                waiting_texts.append(f"#{oid} ~{mins} min")
        if waiting_texts:
            lines.append(f"Still being prepared (ETA): {', '.join(waiting_texts)}")

    if ready_orders:
        ids_list = [
            str(getattr(o, "id"))
            for o in ready_orders
            if getattr(o, "id", None) is not None
        ]
        if ids_list:
            lines.append("")
            lines.append(
                f"You can execute: PICKUP(orders=[{','.join(ids_list)}]) to collect the ready orders."
            )

    return "\n".join(lines)


# ===== POI / pickup hints =====

def vlm_refresh_pickup_hint_nearby(agent: Any) -> None:
    """
    Refresh pickup hints when the agent is near a pickup location.

    Scans both:
        - The agent's own active_orders
        - Orders where the agent serves as helper (help_orders)

    If any orders are ready or soon-to-be-ready at the current position,
    writes a pickup hint into `agent.vlm_ephemeral["pickup_hint"]`.
    """
    tol = get_tol(agent.cfg, "nearby")

    here_orders_ready: List[Any] = []
    here_orders_waiting: List[Tuple[Any, int]] = []

    all_considered: List[Any] = list(agent.active_orders or []) + list(
        getattr(agent, "help_orders", {}).values()
    )

    for o in all_considered:
        if getattr(o, "has_picked_up", False):
            continue

        pu = xy_of_node(getattr(o, "pickup_node", None))
        if not pu or not is_at_xy(agent, pu[0], pu[1], tol_cm=tol):
            continue

        local_now = o.active_now()

        if not hasattr(o, "is_ready_for_pickup"):
            is_ready = True
        else:
            is_ready = o.is_ready_for_pickup(local_now)

        if is_ready:
            here_orders_ready.append(o)
        else:
            remain_s = 0.0
            if hasattr(o, "remaining_prep_s"):
                remain_s = float(o.remaining_prep_s(local_now))

            mins = max(0, int(math.ceil(remain_s / 60.0)))
            here_orders_waiting.append((o, mins))

    if here_orders_ready or here_orders_waiting:
        agent.vlm_ephemeral["pickup_hint"] = vlm_build_pickup_arrival_hint(
            agent, here_orders_ready, here_orders_waiting
        )
    else:
        agent.vlm_ephemeral.pop("pickup_hint", None)


def vlm_refresh_poi_hints_nearby(agent: Any) -> None:
    """
    Refresh all "arrived at POI" hints at the agent's current location.

    This function:
        - Clears the current POI hints in `vlm_ephemeral`.
        - Invokes pickup hint generation.
        - Adds contextual hints for charging stations, rest areas,
          stores, bus stops, car rentals, hospitals, TempBoxes, and
          drop-off locations based on proximity and agent state.

    It only writes lightweight textual hints into `agent.vlm_ephemeral`
    and does not perform any automatic actions (no auto-buy, auto-charge, etc.).
    """
    tol = get_tol(agent.cfg, "nearby")
    now_sim = agent.clock.now_sim()

    # Clear per-step POI hints
    for k in (
        "charging_hint",
        "rest_hint",
        "store_hint",
        "bus_hint",
        "rental_hint",
        "hospital_hint",
        "tempbox_hint",
        "escooter_depleted",
        "dropoff_hint",
    ):
        agent.vlm_ephemeral.pop(k, None)

    # Pickup hints
    vlm_refresh_pickup_hint_nearby(agent)

    # (A) e-scooter depleted while being towed
    if agent.e_scooter and agent.e_scooter.state == ScooterState.DEPLETED and agent.towing_scooter:
        agent.vlm_ephemeral["escooter_depleted"] = (
            "Your e-scooter battery is depleted. You may SWITCH_TRANSPORT(to='walk') to leave the scooter, "
            "or keep dragging it to a charging station and then CHARGE_ESCOOTER(target_pct=80)."
        )

    # (B) Charging station hints
    near_chg = nearest_poi_xy(agent, "charging_station", tol_cm=tol)
    if near_chg:
        sc_charge = (
            agent.assist_scooter
            if agent.assist_scooter is not None
            else (
                agent.e_scooter
                if (agent.e_scooter and getattr(agent.e_scooter, "with_owner", True))
                else None
            )
        )

        parked_here_cmd = None
        for label, s in (("assist", agent.assist_scooter), ("own", agent.e_scooter)):
            if s is not None and getattr(s, "park_xy", None) and is_at_xy(
                agent, s.park_xy[0], s.park_xy[1], tol_cm=tol
            ):
                parked_here_cmd = (
                    'Switch(to="assist_scooter")' if label == "assist" else 'Switch(to="e-scooter")'
                )
                break

        hints: List[str] = []

        if sc_charge:
            scooter_here = (
                agent.towing_scooter
                or (
                    getattr(sc_charge, "park_xy", None)
                    and is_at_xy(agent, sc_charge.park_xy[0], sc_charge.park_xy[1], tol_cm=tol)
                )
            )
            if scooter_here and float(getattr(sc_charge, "battery_pct", 0.0)) < 100.0:
                hints.append(
                    "You are at a charging station. You can CHARGE_ESCOOTER(target_pct=60..100). "
                    "(If you are riding, it will park the scooter automatically before charging.)"
                )

        if parked_here_cmd:
            hints.append(f"There is a parked scooter here. You can use {parked_here_cmd} to get it.")

        if hints:
            agent.vlm_ephemeral["charging_hint"] = " ".join(hints)

    # (C) Rest area
    if nearest_poi_xy(agent, "rest_area", tol_cm=tol) is not None and agent.energy_pct < 100.0:
        agent.vlm_ephemeral["rest_hint"] = "You are at a rest area. You can REST(target_pct=100)."

    # (D) Store
    if nearest_poi_xy(agent, "store", tol_cm=tol) is not None:
        agent.vlm_ephemeral["store_hint"] = (
            "You are inside a store. You can BUY(item_id='energy_drink', qty=XX) or "
            "BUY(items=[{'item_id':'energy_drink','qty':XX}, "
            "{'item_id':'escooter_battery_pack','qty':XX}, "
            "{'item_id':'ice_pack','qty':XX}, {'item_id':'heat_pack','qty':XX}])."
        )

    # (E) Bus station
    if nearest_poi_xy(agent, "bus_station", tol_cm=tol) is not None:
        agent.vlm_ephemeral["bus_hint"] = (
            "You are at a bus stop. When a bus arrives, you can board it."
        )

    # (F) Car rental
    if nearest_poi_xy(agent, "car_rental", tol_cm=tol) is not None:
        if agent.car is None:
            agent.vlm_ephemeral["rental_hint"] = (
                "You are at a car rental. You can RENT_CAR()."
            )
        else:
            agent.vlm_ephemeral["rental_hint"] = (
                "You are at a car rental. You can RETURN_CAR() to stop billing."
            )

    # (G) Hospital treatment
    if getattr(agent, "_hospital_ctx", None):
        remain = max(0.0, float(agent._hospital_ctx["end_sim"] - now_sim))
        mins = int(math.ceil(remain / 60.0))
        agent.vlm_ephemeral["hospital_hint"] = (
            f"You are being treated in the hospital. About {mins} min to full energy."
        )

    # (H) TempBox hints (for collaborative help tasks)
    try:
        comms = get_comms()
        if comms:
            msgs: List[str] = []
            me = str(agent.agent_id)

            def _as_kind(val):
                if isinstance(val, HelpType):
                    return val
                s = str(val)
                for h in HelpType:
                    if s == h.name or s == h.value:
                        return h
                return None

            def _tokens(kind_val):
                k = _as_kind(kind_val)
                if k == HelpType.HELP_CHARGE:
                    return ("an e-scooter", "the e-scooter", "{'escooter': ''}")
                if k in (HelpType.HELP_DELIVERY, HelpType.HELP_PICKUP):
                    return ("food", "the food", "{'food': ''}")
                if k == HelpType.HELP_BUY:
                    return ("inventory items", "the purchased items", "{'inventory': {...}}")
                return ("items", "the items", "{'food': ''} or {'inventory': {...}}")

            # Helper view: TempBox near helper
            for r in getattr(comms, "_active", {}).values():
                if r.completed or str(getattr(r, "accepted_by", "")) != me:
                    continue
                rid = int(getattr(r, "req_id"))
                info = comms.get_temp_box_info(rid) or {}
                pub_box = info.get("publisher_box", {})
                if pub_box.get("xy"):
                    bx, by = pub_box["xy"]
                    if is_at_xy(agent, float(bx), float(by), tol_cm=tol):
                        place_noun, take_noun, _ = _tokens(getattr(r, "kind", None))
                        if pub_box.get("has_content"):
                            msgs.append(
                                f"[Help #{rid}] TempBox is here. "
                                f"Use TAKE_FROM_TEMP_BOX(req_id={rid}) to collect {take_noun}."
                            )
                        else:
                            msgs.append(
                                f"[Help #{rid}] TempBox is here but empty. "
                                f"Wait for the publisher to place {place_noun}."
                            )

            # Publisher A: at the handover point
            for r in getattr(comms, "_active", {}).values():
                if r.completed or str(getattr(r, "publisher_id", "")) != me or not getattr(r, "accepted_by", None):
                    continue
                rid = int(getattr(r, "req_id"))
                det = comms.get_request_detail(rid) or {}
                pxy = det.get("provide_xy")
                if pxy and is_at_xy(agent, float(pxy[0]), float(pxy[1]), tol_cm=tol):
                    place_noun, _take_noun, content_example = _tokens(det.get("kind"))
                    msgs.append(
                        f"[Help #{rid}] You are at the handover point. "
                        f"Use PLACE_TEMP_BOX(req_id={rid}, content={content_example}) to place {place_noun}."
                    )

            # Publisher B: at the helper's TempBox
            for r in getattr(comms, "_active", {}).values():
                if r.completed or str(getattr(r, "publisher_id", "")) != me:
                    continue
                rid = int(getattr(r, "req_id"))
                info = comms.get_temp_box_info(rid) or {}
                helper_box = info.get("helper_box", {})
                if helper_box.get("xy"):
                    bx, by = helper_box["xy"]
                    if is_at_xy(agent, float(bx), float(by), tol_cm=tol):
                        _place_noun, take_noun, _ = _tokens(getattr(r, "kind", None))
                        if helper_box.get("has_content"):
                            msgs.append(
                                f"[Help #{rid}] Helper's TempBox is here. "
                                f"Use TAKE_FROM_TEMP_BOX(req_id={rid}) to retrieve {take_noun}."
                            )
                        else:
                            msgs.append(
                                f"[Help #{rid}] Helper's TempBox is here but empty."
                            )

            if msgs:
                agent.vlm_ephemeral["tempbox_hint"] = "\n".join(msgs)
    except Exception:
        pass

    # (I) Drop-off hints (both own orders and helper orders)
    try:
        msgs: List[str] = []

        # Own orders
        for o in (agent.active_orders or []):
            if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                dxy = xy_of_node(getattr(o, "dropoff_node", None))
                if dxy and is_at_xy(agent, dxy[0], dxy[1], tol_cm=tol):
                    oid = int(getattr(o, "id", -1))
                    msgs.append(
                        f"You are at the drop-off for order #{oid}. "
                        "Choose a delivery method and call "
                        f"DROP_OFF(oid={oid}, method='leave_at_door|knock|call|hand_to_customer')."
                    )

        # Orders where the agent is a helper
        for oid, o in (getattr(agent, "help_orders", {}) or {}).items():
            if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                dxy = xy_of_node(getattr(o, "dropoff_node", None))
                if dxy and is_at_xy(agent, dxy[0], dxy[1], tol_cm=tol):
                    msgs.append(
                        f"You are at the helper drop-off for order #{int(oid)}. "
                        f"Call DROP_OFF(oid={int(oid)}, method='leave_at_door|knock|call|hand_to_customer')."
                    )

        if msgs:
            agent.vlm_ephemeral["dropoff_hint"] = "\n".join(msgs)
        else:
            agent.vlm_ephemeral.pop("dropoff_hint", None)
    except Exception:
        pass


# ===== Map snapshot & full prompt =====

def vlm_map_brief(agent: Any) -> str:
    """
    Request a textual map snapshot around the agent.

    Uses city_map.agent_info_package_xy (if available) to obtain:
        - Nearby roads and POIs
        - Limited-size context around the agent

    Returns:
        A textual summary from the package, or a fallback string if
        no snapshot is available.
    """
    limits = agent.cfg.get("map_snapshot_limits", {})
    if hasattr(agent.city_map, "agent_info_package_xy"):
        pkg = agent.city_map.agent_info_package_xy(
            float(agent.x),
            float(agent.y),
            include_docks=False,
            limit_next=int(limits.get("next", 20)),
            limit_s=int(limits.get("s", 40)),
            limit_poi=int(limits.get("poi", 80)),
            active_orders=agent.active_orders,
            help_orders=agent.help_orders,
        )
        if isinstance(pkg, dict) and pkg.get("text"):
            return str(pkg["text"])
    return "map_brief: N/A"


def vlm_build_input(agent: Any) -> str:
    """
    Assemble the full VLM input prompt for the agent.

    Sections include:
        - past_memory: persisted reflections or notes
        - agent_state: current status summary
        - store_catalog: available items and effects (if any)
        - active_orders: own + helper orders that are accepted and undelivered
        - accepted_help: help requests the agent is currently fulfilling
        - posted_help: help requests the agent has published
        - pickables: environment items that can be picked up via Comms
        - map_snapshot: localized map context
        - recent_actions: last executed actions
        - post_action_plan: previous high-level plan
        - recent_error: recent failure reason, if any
        - ephemeral_context: short-lived hints (e.g., POI hints, bag layout)

    The compiled prompt is also stored in `agent.vlm_last_compiled_input`.
    """
    parts: List[str] = []

    # past memory
    if getattr(agent, "vlm_past_memory", None):
        parts.append("### past_memory")
        parts += [f"- {m}" for m in agent.vlm_past_memory]

    # agent state
    parts.append("### agent_state")
    parts.append(vlm_agent_state_text(agent))

    # store catalog
    if getattr(agent, "_store_manager", None) and hasattr(agent._store_manager, "to_text"):
        try:
            parts.append("### store_catalog")
            parts.append(agent._store_manager.to_text(title="Available items & effects"))
        except Exception:
            pass

    # active orders (own + helper)
    active_blocks: List[str] = []
    for o in (agent.active_orders or []):
        if getattr(o, "is_accepted", False) and not getattr(o, "has_delivered", False):
            active_blocks.append(o.to_text())
    for o in getattr(agent, "help_orders", {}).values():
        if not getattr(o, "has_delivered", False):
            active_blocks.append(o.to_text())

    parts.append("### active_orders")
    if active_blocks:
        parts.append("You have accepted the following active orders:")
        parts.append("\n" + ("\n" + "-" * 48 + "\n").join(active_blocks))
    else:
        parts.append("You currently have no accepted orders.")

    # accepted_help
    try:
        sync_help_lists(agent)
        comms = get_comms()
        parts.append("### accepted_help")
        if comms and getattr(agent, "accepted_help", None):
            now = agent.clock.now_sim()
            for r in agent.accepted_help.values():
                parts.append(
                    r.to_text_for(str(agent.agent_id), comms, now=now, view_as="helper")
                )
        else:
            parts.append("(none)")
    except Exception:
        pass

    # posted_help
    try:
        comms = get_comms()
        parts.append("### posted_help")
        if comms:
            now = agent.clock.now_sim()
            my_open = comms.list_my_posts_open(str(agent.agent_id))
            my_active = comms.list_my_posts_active(str(agent.agent_id))
            if not (my_open or my_active):
                parts.append("(none)")
            else:
                for r in (my_active + my_open):
                    parts.append(
                        r.to_text_for(str(agent.agent_id), comms, now=now, view_as="publisher")
                    )
        else:
            parts.append("(none)")
    except Exception:
        pass

    # pickables
    comms = get_comms()
    parts.append("### pickables")
    if comms:
        parts.append(comms.pickables_text_for(agent.agent_id))
    else:
        parts.append("(none)")

    # map snapshot
    parts.append("### map_snapshot")
    parts.append(vlm_map_brief(agent))

    # recent actions
    if getattr(agent, "vlm_last_actions", None):
        parts.append("### recent_actions")
        actions = list(agent.vlm_last_actions)
        for i, a in enumerate(actions):
            if i == len(actions) - 1:
                parts.append(f"- [Your last successfully executed action] {a}")
            else:
                parts.append(f"- {a}")

    # previous language plan
    if getattr(agent, "_previous_language_plan", None):
        parts.append("### post_action_plan")
        parts.append("After your last action, you planned to:")
        parts.append(agent._previous_language_plan)

    # recent error
    if getattr(agent, "vlm_errors", None):
        parts.append("### recent_error")
        parts.append(agent.vlm_errors)

    # ephemeral context
    if getattr(agent, "_force_place_food_now", False):
        hint = vlm_build_bag_place_hint(agent)
        parts.append("### ephemeral_context")
        parts.append(f"[bag_hint]\n{hint}")
    elif getattr(agent, "vlm_ephemeral", None):
        parts.append("### ephemeral_context")
        for k, v in agent.vlm_ephemeral.items():
            parts.append(f"[{k}]\n{v}")

    txt = "\n".join(parts)
    agent.vlm_last_compiled_input = txt
    return txt


def sync_help_lists(agent: Any) -> None:
    """
    Synchronize helper-related request lists with the Comms system.

    Populates:
        - agent.accepted_help: active help requests being fulfilled.
        - agent.completed_help: help requests that have been completed.

    Errors are silently ignored to match the original behavior.
    """
    try:
        comms = get_comms()
        if not comms:
            return
        mine_active = {
            int(r.req_id): r
            for r in comms.list_my_helps(str(agent.agent_id))
        }
        mine_done = {
            int(r.req_id): r
            for r in comms.list_my_helps_completed(str(agent.agent_id))
        }
        agent.accepted_help = mine_active
        agent.completed_help = mine_done
    except Exception:
        # Fail silently to keep behavior consistent with original code.
        pass