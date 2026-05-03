from __future__ import annotations

import logging

import httpx

from app.schemas.tour_search import TourSearchRequest, TourSearchResponse

logger = logging.getLogger(__name__)


class TourSearchClient:
    def __init__(self, base_url: str, timeout_seconds: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def search(self, payload: TourSearchRequest) -> TourSearchResponse:
        url = f"{self.base_url}/tours/search"
        logger.debug(
            "Calling tour search service",
            extra={
                "url": url,
                "countries_count": len(payload.countries),
                "nights_count": len(payload.num_nights),
            },
        )

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload.model_dump(mode="json", by_alias=True))
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Tour search service returned non-success status",
                    extra={"url": url, "status_code": exc.response.status_code},
                    exc_info=True,
                )
                raise
            except Exception:
                logger.exception("Tour search service call failed", extra={"url": url})
                raise

        logger.info("Tour search service call completed", extra={"url": url})
        return TourSearchResponse.model_validate(response.json())
