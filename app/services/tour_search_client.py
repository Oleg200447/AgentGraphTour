from __future__ import annotations

import httpx

from app.schemas.tour_search import TourSearchRequest, TourSearchResponse


class TourSearchClient:
    def __init__(self, base_url: str, timeout_seconds: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def search(self, payload: TourSearchRequest) -> TourSearchResponse:
        url = f"{self.base_url}/tours/search"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload.model_dump(mode="json", by_alias=True))
            response.raise_for_status()
        return TourSearchResponse.model_validate(response.json())
