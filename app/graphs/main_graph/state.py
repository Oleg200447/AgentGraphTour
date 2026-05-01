from __future__ import annotations

from app.graphs.subgraphs.offer_agent.state import OfferGraphState


class MainGraphState(OfferGraphState, total=False):
    messages: list[dict[str, str]]
    main_route: str
    response_agent: str
