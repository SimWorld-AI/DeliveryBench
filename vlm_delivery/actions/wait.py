# actions/wait.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_wait(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handle wait action for either fixed duration or until charging is completed.
    """

    # Case 1: Wait until the charging process is completed.
    # Only create a wait context; the charging logic will determine when the wait ends.
    if str(act.data.get("until") or "").lower() == "charge_done":
        now_sim = dm.clock.now_sim()
        dm._wait_ctx = {
            "until": "charge_done",
            # Unified fields for pause-safe update logic
            "last_update_sim": now_sim,
            "elapsed_active_s": 0.0,
        }
        dm._log("start waiting until charge done @virtual")
        return

    # Case 2: Wait for a fixed duration (pause-safe accumulation).
    duration_s = float(act.data.get("duration_s", 0.0))
    if duration_s <= 0.0:
        dm._log("wait skipped: duration <= 0s")
        dm._finish_action(success=True)
        return

    now_sim = dm.clock.now_sim()
    dm._wait_ctx = {
        "duration_s": duration_s,         # Target total active wait time
        "elapsed_active_s": 0.0,          # Accumulated active wait time
        "last_update_sim": now_sim,       # Last update timestamp
        # Legacy fields kept only for backward compatibility (not used to drive completion):
        # "start_sim": now_sim,
        # "end_sim":   now_sim + duration_s,
    }

    dm._log(f"start waiting: {duration_s:.1f}s (~{duration_s/60.0:.1f} min) @virtual")
    rec = getattr(dm, "_recorder", None)
    if rec:
        rec.tick_inactive("wait", duration_s)
    # Actual wait accounting is handled when the wait completes (in the update logic).