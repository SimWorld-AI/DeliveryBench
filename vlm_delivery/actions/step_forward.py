# actions/step_forward.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_step_forward(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Executes a single forward step for the agent.

    The current implementation performs:
    - A fixed forward movement of 100 cm.
    - The parameter `1` indicates one forward step action.

    All logic matches the original DeliveryMan._handle_step_forward,
    except `self` has been replaced with `dm`.
    """
    try:
        # Perform a forward step: 100 cm, 1 step.
        dm._ue.delivery_man_step_forward(dm.agent_id, 100, 1)
        dm._finish_action(success=True)

    except Exception as e:
        dm.vlm_add_error(f"step_forward failed: {e}")
        dm._finish_action(success=False)