# actions/view_orders.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_view_orders(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Display current order-pool information if available.
    """
    om = act.data.get("order_manager") or dm._order_manager

    # Retrieve order pool text if the manager provides it
    if om and hasattr(om, "orders_text"):
        pool_text = om.orders_text()
        if pool_text:
            dm.vlm_add_ephemeral("order_pool", pool_text)
            dm._log("view orders")

    dm._finish_action(success=True)