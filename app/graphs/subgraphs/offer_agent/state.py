from __future__ import annotations

from typing import Any, TypedDict


class OfferGraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    reply_url: str

    latest_user_text: str
    messages_from_main: list[dict[str, str]]

    data_min: str
    data_max: str
    num_adults: int
    num_childs: int
    birthdays: list[str]
    budget: float
    countries: list[str]
    num_nights: list[int]
    query: str

    missing_fields: list[str]
    requirements_complete: bool
    confirmation_needed: bool
    query_requested: bool

    offers: list[dict[str, Any]]
    teztour_ids: list[int]
    cursor: int
    shown_hotel_ids: list[int]

    selected_hotel_id: int
    selected_teztour_id: int

    route_target: str
    assistant_response_text: str
