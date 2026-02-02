# actions/turn_around.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_turn_around(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Perform a turn-around action for the delivery agent.
    The action requires:
        - angle: the rotation angle in degrees
        - direction: rotation direction (e.g., left/right depending on UE logic)
    """
    try:
        angle = act.data.get("angle")
        direction = act.data.get("direction")
        dm._ue.delivery_man_turn_around(dm.agent_id, angle, direction)
        dm._finish_action(success=True)
    except Exception as e:
        dm.vlm_add_error(f"turn_around failed: {e}")
        dm._finish_action(success=False)