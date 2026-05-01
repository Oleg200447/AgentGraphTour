from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.services.hotels_repository import HotelsRepository
from app.services.tour_search_client import TourSearchClient


@dataclass
class GraphDependencies:
    settings: Settings
    tour_search_client: TourSearchClient
    hotels_repository: HotelsRepository
    router_llm: Any | None
    agent_llm: Any | None
    offer_graph: Any | None = None
    tourism_graph: Any | None = None
