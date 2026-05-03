from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graphs.deps import GraphDependencies
from app.graphs.subgraphs.offer_agent.state import OfferGraphState
from app.schemas.tour_search import TourSearchRequest

logger = logging.getLogger(__name__)

CORE_REQUIRED_FIELDS: list[str] = [
    "data_min",
    "data_max",
    "num_adults",
    "num_childs",
    "birthdays",
    "budget",
    "countries",
    "num_nights",
]

ALL_REQUIRED_FIELDS: list[str] = [
    *CORE_REQUIRED_FIELDS,
    "query",
]


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_state(state: OfferGraphState) -> OfferGraphState:
    s = dict(state)
    s.setdefault("messages_from_main", [])
    s.setdefault("birthdays", [])
    s.setdefault("countries", [])
    s.setdefault("num_nights", [])
    s.setdefault("offers", [])
    s.setdefault("teztour_ids", [])
    s.setdefault("shown_hotel_ids", [])
    s.setdefault("cursor", 0)
    s.setdefault("missing_fields", [])
    s.setdefault("requirements_complete", False)
    s.setdefault("confirmation_needed", False)
    s.setdefault("query_requested", False)
    return s


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = text[start : end + 1]
    try:
        loaded = json.loads(fragment)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


async def _extract_updates_with_llm(
    deps: GraphDependencies,
    text: str,
    fields_to_fill: list[str],
) -> dict[str, Any]:
    if deps.agent_llm is None or not fields_to_fill:
        if deps.agent_llm is None:
            logger.debug("Offer extraction LLM is unavailable; skipping LLM updates")
        return {}

    today = date.today()
    field_descriptions = {
        "data_min": "data_min (YYYY-MM-DD или null)",
        "data_max": "data_max (YYYY-MM-DD или null)",
        "num_adults": "num_adults (int или null)",
        "num_childs": "num_childs (int или null)",
        "birthdays": "birthdays (массив YYYY-MM-DD)",
        "budget": "budget (float или null)",
        "countries": "countries (массив строк)",
        "num_nights": "num_nights (массив int)",
    }
    ordered_fields = [field for field in CORE_REQUIRED_FIELDS if field in fields_to_fill]
    if not ordered_fields:
        return {}

    fields_block = "\n".join(f"- {field_descriptions[field]}" for field in ordered_fields)
    prompt = f"""
        Ты — AI-ассистент туристического агентства. Твоя задача — точно извлекать параметры подбора тура из сообщения пользователя.

        ПРАВИЛА ИЗВЛЕЧЕНИЯ:
        1. Извлекай данные только для тех ключей, которые перечислены в блоке "ОЖИДАЕМЫЕ КЛЮЧИ".
        2. Если в сообщении нет явной информации для ключа — обязательно ставь `null` (или `[]` для списков). Запрещено выдумывать параметры или брать их "из головы".
        3. Значения стран должны строго соответствовать списку: ["Турция", "Египет", "Китай", "Таиланд", "Грузия"]. Если пользователь называет другую страну, игнорируй ее (ставь null или не добавляй в список).
        4. Все даты должны быть в формате YYYY-MM-DD. 
        5. ТЕКУЩАЯ ДАТА: {today.isoformat()}. Если пользователь использует относительное время ("завтра", "через неделю", "в конце месяца"), вычисли точную дату на основе текущей.

        ОЖИДАЕМЫЕ КЛЮЧИ (сейчас нужно заполнить только их):
        {fields_block}

        Сообщение пользователя: "{text}"

        ВЫВОД:
        Верни строго валидный JSON-объект. Не добавляй никаких пояснений до или после JSON. Не используй markdown-разметку (например, блоки ```json).
    """.strip()
    try:
        ai_msg = await deps.agent_llm.ainvoke(prompt)
    except Exception:
        logger.exception("Failed to extract offer updates with LLM")
        return {}

    content = getattr(ai_msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    extracted = _extract_json_object(content)
    return extracted or {}


def _extract_updates_heuristic(text: str, state: OfferGraphState) -> dict[str, Any]:
    lower = text.lower()
    updates: dict[str, Any] = {}

    adults_match = re.search(r"(\d+)\s*(взросл|adult)", lower)
    if adults_match:
        updates["num_adults"] = int(adults_match.group(1))

    childs_match = re.search(r"(\d+)\s*(реб|дет|child)", lower)
    if childs_match:
        updates["num_childs"] = int(childs_match.group(1))

    budget_match = re.search(r"(?:бюджет|до)\s*(\d+(?:[\.,]\d+)?)", lower)
    if budget_match:
        updates["budget"] = float(budget_match.group(1).replace(",", "."))

    nights_match = re.findall(r"(\d+)\s*ноч", lower)
    if nights_match:
        updates["num_nights"] = [int(x) for x in nights_match if int(x) > 0]

    date_tokens = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if len(date_tokens) >= 2:
        updates["data_min"] = date_tokens[0]
        updates["data_max"] = date_tokens[1]
    elif len(date_tokens) == 1:
        if "data_min" not in state:
            updates["data_min"] = date_tokens[0]
        else:
            updates["data_max"] = date_tokens[0]

    countries_match = re.search(r"(?:в|страна|страны)\s+([A-Za-zА-Яа-я,\s-]{3,})", text)
    if countries_match:
        raw = countries_match.group(1)
        countries = [part.strip() for part in raw.split(",") if part.strip()]
        if countries:
            updates["countries"] = countries[:5]

    return updates


def _merge_updates(state: OfferGraphState, updates: dict[str, Any]) -> OfferGraphState:
    s = dict(state)
    for key, value in updates.items():
        if value is None:
            continue
        if key in {"countries", "birthdays", "num_nights"} and isinstance(value, list):
            s[key] = value
            continue
        s[key] = value
    return s


def _compute_missing_fields(state: OfferGraphState, *, include_query: bool = True) -> list[str]:
    missing: list[str] = []
    required_fields = ALL_REQUIRED_FIELDS if include_query else CORE_REQUIRED_FIELDS

    for field in required_fields:
        value = state.get(field)
        if field in {"countries", "num_nights"}:
            if not isinstance(value, list) or not value:
                missing.append(field)
            continue
        if field == "birthdays":
            num_childs = int(state.get("num_childs") or 0)
            if num_childs > 0:
                if not isinstance(value, list) or len(value) != num_childs:
                    missing.append(field)
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)

    try:
        if "data_min" in state and "data_max" in state:
            dmin = date.fromisoformat(str(state["data_min"]))
            dmax = date.fromisoformat(str(state["data_max"]))
            if dmin > dmax and "data_max" not in missing:
                missing.append("data_max")
    except Exception:
        if "data_min" not in missing:
            missing.append("data_min")
        if "data_max" not in missing:
            missing.append("data_max")

    birthdays = state.get("birthdays", [])
    data_min_raw = state.get("data_min")
    if birthdays and data_min_raw:
        try:
            dmin = date.fromisoformat(str(data_min_raw))
            for b in birthdays:
                if date.fromisoformat(str(b)) > dmin:
                    if "birthdays" not in missing:
                        missing.append("birthdays")
                    break
        except Exception:
            if "birthdays" not in missing:
                missing.append("birthdays")

    return sorted(set(missing))


def _missing_fields_to_ru(missing: list[str]) -> str:
    mapping = {
        "data_min": "дата вылета от (data_min, YYYY-MM-DD)",
        "data_max": "дата вылета до (data_max, YYYY-MM-DD)",
        "num_adults": "число взрослых (1-2)",
        "num_childs": "число детей (0-2)",
        "birthdays": "дни рождения детей (список дат)",
        "budget": "бюджет (число > 0)",
        "countries": "страны (список)",
        "num_nights": "количество ночей (список)",
        "query": "текстовый запрос",
    }
    human = [mapping.get(field, field) for field in missing]
    return "\n".join(f"- {item}" for item in human)


def _render_requirements_summary(state: OfferGraphState) -> str:
    return (
        "Собрал параметры для подбора тура:\n"
        f"- data_min: {state.get('data_min')}\n"
        f"- data_max: {state.get('data_max')}\n"
        f"- num_adults: {state.get('num_adults')}\n"
        f"- num_childs: {state.get('num_childs')}\n"
        f"- birthdays: {state.get('birthdays', [])}\n"
        f"- budget: {state.get('budget')}\n"
        f"- countries: {state.get('countries', [])}\n"
        f"- num_nights: {state.get('num_nights', [])}\n"
        f"- query: {state.get('query')}\n\n"
        "Если всё верно, напишите: «покажи варианты». Если нужно — пришлите правки."
    )


def _render_query_request_message() -> str:
    return (
        "Основные параметры собраны.\n"
        "Теперь отдельно опишите, что хотелось бы видеть в отеле "
        "(например: первая линия, семейный отель, аквапарк, спокойный район и т.д.)."
    )


def _wants_search(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ["покажи варианты", "подбери", "найди", "искать", "варианты"])


def _wants_next(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ["след", "еще", "ещё", "дальше", "next"]) and (
        "3" in lower or "три" in lower or "вариант" in lower
    )


def _wants_details(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ["подроб", "описан", "расскажи", "детал"])


async def _route_with_llm(deps: GraphDependencies, state: OfferGraphState) -> str | None:
    if deps.router_llm is None:
        logger.debug("Offer router LLM is unavailable, heuristic routing will be used")
        return None

    prompt = f"""
        Ты — бездушный конечный автомат (маршрутизатор). Твоя единственная задача — применять правила IF/THEN. ЗАПРЕЩЕНО использовать "здравый смысл" или додумывать логику.

        ПРАВИЛА:
        1. Верни "collect", ЕСЛИ requirements_complete == False.
        2. Верни "search_first_three", ЕСЛИ requirements_complete == True И пользователь пишет фразы вроде "покажи варианты", "ищи", "давай". (ВНИМАНИЕ: Если teztour_ids == 0, это НОРМАЛЬНО для этого шага, так как поиск еще не запускался!).
        3. Верни "show_next_three", ЕСЛИ пользователь просит "еще/дальше" И teztour_ids > 0.
        4. Верни "show_hotel_details", ЕСЛИ пользователь просит детали конкретного отеля.
        5. Верни "collect" во всех остальных непонятных случаях.

        ТЕКУЩЕЕ СОСТОЯНИЕ:
        - requirements_complete == {state.get('requirements_complete')}
        - cursor == {state.get('cursor')}
        - teztour_ids == {len(state.get('teztour_ids', []))}

        СООБЩЕНИЕ: "{state.get('latest_user_text', '')}"

        Выведи строго одно слово без кавычек, скобок и JSON:
    """.strip()
    try:
        ai_msg = await deps.router_llm.ainvoke(prompt)
    except Exception:
        logger.exception("Offer router LLM invocation failed")
        return None
    content = getattr(ai_msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    content = content.strip().lower()
    for token in ["collect", "search_first_three", "show_next_three", "show_hotel_details"]:
        if token in content:
            return token
    return None


async def _resolve_route(deps: GraphDependencies, state: OfferGraphState) -> str:
    llm_route = await _route_with_llm(deps, state)
    if llm_route:
        logger.debug("Offer route selected by LLM", extra={"route": llm_route})
        return llm_route
    
    text = str(state.get("latest_user_text", ""))
    if _wants_next(text) and state.get("teztour_ids"):
        logger.debug("Offer route selected by heuristic", extra={"route": "show_next_three"})
        return "show_next_three"
    if _wants_details(text):
        logger.debug("Offer route selected by heuristic", extra={"route": "show_hotel_details"})
        return "show_hotel_details"
    if _wants_search(text) and state.get("requirements_complete"):
        logger.debug("Offer route selected by heuristic", extra={"route": "search_first_three"})
        return "search_first_three"

    logger.debug("Offer route selected by heuristic", extra={"route": "collect"})
    return "collect"


def _parse_hotel_id_from_text(text: str) -> int | None:
    matches = re.findall(r"\b\d+\b", text)
    if not matches:
        return None
    try:
        return int(matches[0])
    except ValueError:
        return None


async def _render_hotels_page(
    deps: GraphDependencies,
    teztour_ids: list[int],
    cursor: int,
) -> tuple[str, int, list[int]]:
    chunk = teztour_ids[cursor : cursor + 3]
    if not chunk:
        return (
            "Больше вариантов пока нет. Могу изменить параметры и выполнить новый подбор.",
            cursor,
            [],
        )

    hotels = await deps.hotels_repository.get_hotels_by_teztour_ids(chunk)
    if not hotels:
        return (
            "По найденным teztour_id не удалось получить карточки отелей из БД. Попробуйте новый запрос.",
            cursor,
            [],
        )

    lines: list[str] = ["Нашёл варианты (показываю до 3):"]
    shown_hotel_ids: list[int] = []
    for idx, hotel in enumerate(hotels, start=1):
        shown_hotel_ids.append(hotel.hotel_id)
        lines.append(
            f"{idx}. hotel_id={hotel.hotel_id} | {hotel.name}\n"
            f"Кратко: {hotel.short_description or 'Описание отсутствует.'}"
        )
    lines.append("\nЧтобы увидеть ещё три — напишите: «покажи следующие 3». ")
    lines.append("Чтобы получить детали по отелю — напишите, например: «подробно про 12345». ")
    return "\n\n".join(lines), cursor + len(chunk), shown_hotel_ids


async def build_offer_agent_graph(deps: GraphDependencies):
    logger.info("Building offer graph")

    async def router_node(state: OfferGraphState) -> OfferGraphState:
        s = _normalize_state(state)
        route_target = await _resolve_route(deps, s)
        logger.info("Offer graph route selected", extra={"route": route_target})
        return {"route_target": route_target}

    async def collect_or_edit_requirements_node(state: OfferGraphState) -> OfferGraphState:
        s = _normalize_state(state)
        user_text = str(s.get("latest_user_text", "")).strip()

        if s.get("query_requested") and not _as_non_empty_str(s.get("query")):
            if not user_text or _wants_search(user_text):
                return {
                    **s,
                    "missing_fields": ["query"],
                    "requirements_complete": False,
                    "confirmation_needed": False,
                    "query_requested": True,
                    "assistant_response_text": _render_query_request_message(),
                }

            next_state = _merge_updates(
                s,
                {
                    "query": user_text,
                    "query_requested": False,
                },
            )
            summary = _render_requirements_summary(next_state)
            return {
                **next_state,
                "missing_fields": [],
                "requirements_complete": True,
                "confirmation_needed": True,
                "query_requested": False,
                "assistant_response_text": summary,
            }

        fields_to_fill = _compute_missing_fields(s, include_query=False)
        llm_updates = await _extract_updates_with_llm(deps, user_text, fields_to_fill)
        heuristic_updates = _extract_updates_heuristic(user_text, s)
        
        merged_updates = dict(llm_updates)

        for key, value in heuristic_updates.items():
            if key == "countries":
                if not merged_updates.get("countries"):
                    merged_updates[key] = value
            else:
                merged_updates[key] = value

        if "data_min" not in merged_updates and "data_min" not in s:
            merged_updates["data_min"] = date.today().isoformat()
        if "data_max" not in merged_updates and "data_max" not in s:
            merged_updates["data_max"] = (date.today() + timedelta(days=90)).isoformat()

        next_state = _merge_updates(s, merged_updates)
        missing = _compute_missing_fields(next_state, include_query=False)

        if missing:
            logger.debug("Offer requirements are incomplete", extra={"missing_fields": missing})
            response = (
                "Нужны дополнительные данные для подбора тура.\n"
                f"Не хватает:\n{_missing_fields_to_ru(missing)}\n\n"
                "Пришлите недостающие параметры одним сообщением."
            )
            return {
                **next_state,
                "missing_fields": missing,
                "requirements_complete": False,
                "confirmation_needed": False,
                "query_requested": False,
                "assistant_response_text": response,
            }

        if not _as_non_empty_str(next_state.get("query")):
            logger.debug("Offer requirements complete but query is missing")
            return {
                **next_state,
                "missing_fields": ["query"],
                "requirements_complete": False,
                "confirmation_needed": False,
                "query_requested": True,
                "assistant_response_text": _render_query_request_message(),
            }

        summary = _render_requirements_summary(next_state)
        logger.info("Offer requirements collected successfully")
        return {
            **next_state,
            "missing_fields": [],
            "requirements_complete": True,
            "confirmation_needed": True,
            "query_requested": False,
            "assistant_response_text": summary,
        }

    async def search_first_three_node(state: OfferGraphState) -> OfferGraphState:
        s = _normalize_state(state)
        missing = _compute_missing_fields(s, include_query=True)
        if missing:
            logger.debug("Search requested with missing fields", extra={"missing_fields": missing})
            if missing == ["query"]:
                return {
                    "missing_fields": missing,
                    "requirements_complete": False,
                    "query_requested": True,
                    "assistant_response_text": _render_query_request_message(),
                }
            return {
                "missing_fields": missing,
                "requirements_complete": False,
                "query_requested": "query" in missing,
                "assistant_response_text": (
                    "Пока не могу выполнить подбор, не хватает параметров:\n"
                    f"{_missing_fields_to_ru(missing)}"
                ),
            }

        payload = {
            "data_min": s.get("data_min"),
            "data_max": s.get("data_max"),
            "num_adults": s.get("num_adults"),
            "num_childs": s.get("num_childs"),
            "birthdays": s.get("birthdays", []),
            "budget": s.get("budget"),
            "countries": s.get("countries", []),
            "num_nights": s.get("num_nights", []),
            "query": _as_non_empty_str(s.get("query")),
        }

        try:
            request_payload = TourSearchRequest.model_validate(payload)
        except Exception as exc:
            logger.warning("Tour search payload validation failed", exc_info=True)
            return {
                "assistant_response_text": (
                    "Параметры подбора не прошли валидацию сервиса. "
                    f"Исправьте данные и повторите запрос. Детали: {exc}"
                ),
                "requirements_complete": False,
            }

        try:
            result = await deps.tour_search_client.search(request_payload)
        except Exception as exc:
            logger.error("Tour search service call failed", exc_info=True)
            return {
                "assistant_response_text": f"Сервис подбора туров временно недоступен: {exc}",
            }

        offers = result.offers
        if not offers:
            logger.info("Tour search returned zero offers")
            return {
                "offers": [],
                "teztour_ids": [],
                "cursor": 0,
                "assistant_response_text": (
                    "По заданным параметрам офферов не найдено. "
                    "Попробуйте расширить бюджет, диапазон дат или список стран."
                ),
                "confirmation_needed": False,
            }

        teztour_ids = [int(offer.teztour_id) for offer in offers]
        logger.info("Tour search returned offers", extra={"offers_count": len(teztour_ids)})
        response_text, next_cursor, shown = await _render_hotels_page(deps, teztour_ids, cursor=0)

        return {
            "offers": [offer.model_dump(mode="json") for offer in offers],
            "teztour_ids": teztour_ids,
            "cursor": next_cursor,
            "shown_hotel_ids": shown,
            "assistant_response_text": response_text,
            "confirmation_needed": False,
        }

    async def show_next_three_node(state: OfferGraphState) -> OfferGraphState:
        s = _normalize_state(state)
        teztour_ids = [int(x) for x in s.get("teztour_ids", [])]
        if not teztour_ids:
            logger.info("Requested next hotels page without active selection")
            return {
                "assistant_response_text": (
                    "Пока нет активной подборки. Сначала скажите «покажи варианты», "
                    "чтобы я выполнил поиск."
                )
            }

        cursor = int(s.get("cursor", 0))
        response_text, next_cursor, shown = await _render_hotels_page(deps, teztour_ids, cursor=cursor)
        shown_total = list(s.get("shown_hotel_ids", [])) + shown
        return {
            "cursor": next_cursor,
            "shown_hotel_ids": shown_total,
            "assistant_response_text": response_text,
        }

    async def show_hotel_details_node(state: OfferGraphState) -> OfferGraphState:
        s = _normalize_state(state)
        text = str(s.get("latest_user_text", ""))
        hotel_id = _parse_hotel_id_from_text(text) or s.get("selected_hotel_id")

        if not hotel_id:
            logger.debug("Hotel details requested without hotel_id")
            shown = s.get("shown_hotel_ids", [])
            hint = f" Например, один из показанных: {shown[:3]}" if shown else ""
            return {
                "assistant_response_text": (
                    "Уточните hotel_id отеля, по которому нужно полное описание." + hint
                )
            }

        details = await deps.hotels_repository.get_details_by_hotel_id(int(hotel_id))
        if details is None:
            logger.info("Hotel details not found", extra={"hotel_id": int(hotel_id)})
            return {
                "assistant_response_text": (
                    f"Не нашёл отель с hotel_id={hotel_id} в таблице hotels. "
                    "Проверьте номер и попробуйте ещё раз."
                )
            }

        description = details.description or "Полное описание отсутствует в БД."
        response = (
            f"Подробно про hotel_id={details.hotel_id} ({details.name}):\n"
            f"{description}"
        )
        logger.debug("Hotel details rendered", extra={"hotel_id": details.hotel_id})
        return {
            "selected_hotel_id": details.hotel_id,
            "selected_teztour_id": details.teztour_id,
            "assistant_response_text": response,
        }

    async def finalize_node(state: OfferGraphState) -> OfferGraphState:
        return {
            "messages_from_main": [],
        }

    def route_selector(state: OfferGraphState) -> str:
        route = str(state.get("route_target", "collect"))
        if route in {"collect", "search_first_three", "show_next_three", "show_hotel_details"}:
            return route
        return "collect"

    graph = StateGraph(OfferGraphState)
    graph.add_node("router", router_node)
    graph.add_node("collect_or_edit_requirements", collect_or_edit_requirements_node)
    graph.add_node("search_first_three", search_first_three_node)
    graph.add_node("show_next_three", show_next_three_node)
    graph.add_node("show_hotel_details", show_hotel_details_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_selector,
        {
            "collect": "collect_or_edit_requirements",
            "search_first_three": "search_first_three",
            "show_next_three": "show_next_three",
            "show_hotel_details": "show_hotel_details",
        },
    )

    graph.add_edge("collect_or_edit_requirements", "finalize")
    graph.add_edge("search_first_three", "finalize")
    graph.add_edge("show_next_three", "finalize")
    graph.add_edge("show_hotel_details", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
