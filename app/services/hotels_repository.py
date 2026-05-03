from __future__ import annotations

import logging

import asyncpg

from app.schemas.hotel import HotelBrief, HotelDetails

logger = logging.getLogger(__name__)


class HotelsRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def get_hotels_by_teztour_ids(self, teztour_ids: list[int]) -> list[HotelBrief]:
        if not teztour_ids:
            logger.debug("No teztour ids provided for hotel lookup")
            return []

        query = """
        SELECT hotel_id, teztour_id, name, short_description
        FROM hotels
        WHERE teztour_id = ANY($1::int[])
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, teztour_ids)

        logger.debug(
            "Fetched hotels by teztour ids",
            extra={"requested_count": len(teztour_ids), "fetched_count": len(rows)},
        )

        by_id = {
            int(row["teztour_id"]): HotelBrief(
                hotel_id=int(row["hotel_id"]),
                teztour_id=int(row["teztour_id"]),
                name=str(row["name"] or ""),
                short_description=(str(row["short_description"]) if row["short_description"] is not None else None),
            )
            for row in rows
        }
        return [by_id[tid] for tid in teztour_ids if tid in by_id]

    async def get_details_by_hotel_id(self, hotel_id: int) -> HotelDetails | None:
        query = """
        SELECT hotel_id, teztour_id, name, description
        FROM hotels
        WHERE hotel_id = $1
        LIMIT 1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, hotel_id)

        if row is None:
            logger.info("Hotel details not found by hotel_id", extra={"hotel_id": hotel_id})
            return None

        logger.debug("Hotel details found by hotel_id", extra={"hotel_id": hotel_id})

        return HotelDetails(
            hotel_id=int(row["hotel_id"]),
            teztour_id=int(row["teztour_id"]),
            name=str(row["name"] or ""),
            description=(str(row["description"]) if row["description"] is not None else None),
        )

    async def get_details_by_teztour_id(self, teztour_id: int) -> HotelDetails | None:
        query = """
        SELECT hotel_id, teztour_id, name, description
        FROM hotels
        WHERE teztour_id = $1
        LIMIT 1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, teztour_id)

        if row is None:
            logger.info("Hotel details not found by teztour_id", extra={"teztour_id": teztour_id})
            return None

        logger.debug("Hotel details found by teztour_id", extra={"teztour_id": teztour_id})

        return HotelDetails(
            hotel_id=int(row["hotel_id"]),
            teztour_id=int(row["teztour_id"]),
            name=str(row["name"] or ""),
            description=(str(row["description"]) if row["description"] is not None else None),
        )
