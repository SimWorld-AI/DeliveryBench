"""SimWorld package for simulation of urban environments and traffic.

This package provides tools for city generation, traffic simulation,
and visualization in Unreal Engine.
"""

from __future__ import annotations

# NOTE:
# Keep `simworld` import lightweight.
# Some submodules (e.g. assets retrieval / embedding models) have heavy optional dependencies.
# Those should not be imported at package import time.

from simworld.config import Config  # always available
from simworld.utils.logger import Logger  # always available

try:
    from simworld.citygen.city.city_generator import CityGenerator
    from simworld.citygen.function_call.city_function_call import CityFunctionCall
except Exception:  # pragma: no cover
    CityGenerator = None  # type: ignore[assignment]
    CityFunctionCall = None  # type: ignore[assignment]

try:
    from simworld.agent.base_agent import BaseAgent
except Exception:  # pragma: no cover
    BaseAgent = None  # type: ignore[assignment]

try:
    from simworld.llm.base_llm import BaseLLM
except Exception:  # pragma: no cover
    BaseLLM = None  # type: ignore[assignment]

try:
    from simworld.map.map import Edge, Map, Node
except Exception:  # pragma: no cover
    Edge = None  # type: ignore[assignment]
    Map = None  # type: ignore[assignment]
    Node = None  # type: ignore[assignment]

try:
    from simworld.communicator.communicator import Communicator
    from simworld.communicator.unrealcv import UnrealCV
except Exception:  # pragma: no cover
    Communicator = None  # type: ignore[assignment]
    UnrealCV = None  # type: ignore[assignment]

try:
    from simworld.traffic.controller.traffic_controller import TrafficController
    from simworld.traffic.manager.intersection_manager import IntersectionManager
    from simworld.traffic.manager.pedestrian_manager import PedestrianManager
    from simworld.traffic.manager.vehicle_manager import VehicleManager
except Exception:  # pragma: no cover
    TrafficController = None  # type: ignore[assignment]
    PedestrianManager = None  # type: ignore[assignment]
    VehicleManager = None  # type: ignore[assignment]
    IntersectionManager = None  # type: ignore[assignment]

# Heavy optional module (sentence_transformers / CLIP / etc). Import lazily.
try:
    from simworld.assets_rp.AssetsRP import AssetsRetrieverPlacer
except Exception:  # pragma: no cover
    AssetsRetrieverPlacer = None  # type: ignore[assignment]

__all__ = [
    n
    for n in [
        "Config",
        "Logger",
        "CityGenerator",
        "CityFunctionCall",
        "BaseLLM",
        "AssetsRetrieverPlacer",
        "TrafficController",
        "PedestrianManager",
        "VehicleManager",
        "IntersectionManager",
        "Map",
        "Node",
        "Edge",
        "Communicator",
        "UnrealCV",
        "BaseAgent",
    ]
    if globals().get(n) is not None
]
