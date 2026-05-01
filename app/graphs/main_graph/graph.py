from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.graphs.deps import GraphDependencies
from app.graphs.main_graph.state import MainGraphState


def _looks_like_offer_request(text: str) -> bool:
    lower = text.lower()
    tokens = [
        "тур",
        "отел",
        "отель",
        "hotel",
        "подбери",
        "путев",
        "путёв",
        "resort",
        "перелет",
        "перелёт",
        "вылет",
        "ноч",
    ]
    return any(token in lower for token in tokens)


def _looks_like_company_request(text: str) -> bool:
    lower = text.lower()
    tokens = [
        "турфирм",
        "тур агентств",
        "тур-агентств",
        "офис",
        "документ",
        "регламент",
        "график работы",
        "как добраться",
        "что взять",
        "куда приехать",
    ]
    return any(token in lower for token in tokens)


def _looks_like_tourism_general(text: str) -> bool:
    lower = text.lower()
    tokens = [
        "погод",
        "куда поехать",
        "когда лучше",
        "сезон",
        "популярн",
        "что посмотреть",
        "виза",
        "достопримеч",
        "туризм",
    ]
    return any(token in lower for token in tokens)


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_text_from_agent_result(result: dict[str, Any]) -> str | None:
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in {"ai", "assistant"}:
            content = getattr(msg, "content", "")
            text = _extract_text_from_message_content(content)
            if text:
                return text

        if isinstance(msg, dict):
            role = msg.get("role")
            if role in {"assistant", "ai"}:
                text = _extract_text_from_message_content(msg.get("content", ""))
                if text:
                    return text

    return None


async def _route_with_llm(router_llm: Any | None, message: str) -> str | None:
    if router_llm is None:
        return None

    prompt = (
        "Ты роутер основного графа. Классифицируй сообщение. "
        "Если это подбор тура, просмотр отелей, уточнение по уже показанным отелям — верни token OFFER. "
        "Если это общий вопрос о туризме (погода, сезон, куда ехать, популярные направления) — верни token TOURISM. "
        "Если вопрос о турфирме/офисе/процессах обслуживания — верни token COMPANY. "
        "Если тема не связана с туризмом — верни token OTHER. Верни строго один token.\n"
        f"Сообщение: {message}"
    )
    try:
        ai_msg = await router_llm.ainvoke(prompt)
    except Exception:
        return None

    content = getattr(ai_msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    content = content.upper()
    if "OFFER" in content:
        return "offer"
    if "TOURISM" in content:
        return "tourism"
    if "COMPANY" in content:
        return "company"
    if "OTHER" in content:
        return "other"
    return None


async def build_main_graph(deps: GraphDependencies):
    async def main_router_node(state: MainGraphState) -> MainGraphState:
        text = str(state.get("latest_user_text", ""))
        routed = await _route_with_llm(deps.router_llm, text)
        if routed is None:
            if _looks_like_company_request(text):
                routed = "company"
            elif _looks_like_tourism_general(text):
                routed = "tourism"
            elif _looks_like_offer_request(text):
                routed = "offer"
            else:
                routed = "other"

        messages = state.get("messages", [])
        return {
            "main_route": routed,
            "messages_from_main": messages,
        }

    async def offer_subgraph_node(state: MainGraphState) -> MainGraphState:
        offer_graph = deps.offer_graph
        if offer_graph is None:
            return {
                "assistant_response_text": "Offer subgraph не инициализирован.",
            }

        result = await offer_graph.ainvoke(dict(state))
        if not isinstance(result, dict):
            return {"assistant_response_text": "Ошибка выполнения offer subgraph."}
        result.setdefault("response_agent", "offer_agent")
        return result

    async def tourism_subgraph_node(state: MainGraphState) -> MainGraphState:
        tourism_graph = deps.tourism_graph
        if tourism_graph is None:
            return {
                "assistant_response_text": "Tourism subgraph не инициализирован.",
                "response_agent": "tourism_agent",
            }

        raw_messages = state.get("messages", [])
        graph_messages: list[Any] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            if role == "user":
                graph_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                graph_messages.append(AIMessage(content=content))
            elif role == "system":
                graph_messages.append(SystemMessage(content=content))

        if not graph_messages:
            graph_messages = [HumanMessage(content=str(state.get("latest_user_text", "")))]

        result = await tourism_graph.ainvoke({"messages": graph_messages})
        if not isinstance(result, dict):
            return {
                "assistant_response_text": "Ошибка выполнения tourism subgraph.",
                "response_agent": "tourism_agent",
            }

        response_text = _extract_text_from_agent_result(result) or "Не удалось сформировать ответ по туризму."
        return {
            "assistant_response_text": response_text,
            "response_agent": "tourism_agent",
        }

    async def company_info_node(state: MainGraphState) -> MainGraphState:
        user_text = str(state.get("latest_user_text", "")).strip()
        if deps.agent_llm is None:
            return {
                "assistant_response_text": "Сервис ответов по турфирме сейчас недоступен.",
                "response_agent": "company_info",
            }

        prompt = deps.settings.tour_firm_system_prompt
        try:
            ai_msg = await deps.agent_llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_text),
                ]
            )
            content = _extract_text_from_message_content(getattr(ai_msg, "content", ""))
        except Exception:
            content = "Не удалось подготовить ответ по запросу о турфирме."

        return {
            "assistant_response_text": content or "Не удалось подготовить ответ по запросу о турфирме.",
            "response_agent": "company_info",
        }

    async def other_node(state: MainGraphState) -> MainGraphState:
        return {
            "assistant_response_text": (
                "Я помогаю только с вопросами о туризме и турфирме. "
                "Пожалуйста, вернитесь к теме путешествий: например, спросите про подбор тура, "
                "погоду в стране или популярные направления."
            ),
            "messages_from_main": [],
            "response_agent": "other",
        }

    def route_selector(state: MainGraphState) -> str:
        route = str(state.get("main_route", "offer"))
        return route if route in {"offer", "tourism", "company", "other"} else "offer"

    graph = StateGraph(MainGraphState)
    graph.add_node("main_router", main_router_node)
    graph.add_node("offer_subgraph", offer_subgraph_node)
    graph.add_node("tourism_subgraph", tourism_subgraph_node)
    graph.add_node("company_info", company_info_node)
    graph.add_node("other", other_node)

    graph.add_edge(START, "main_router")
    graph.add_conditional_edges(
        "main_router",
        route_selector,
        {
            "offer": "offer_subgraph",
            "tourism": "tourism_subgraph",
            "company": "company_info",
            "other": "other",
        },
    )
    graph.add_edge("offer_subgraph", END)
    graph.add_edge("tourism_subgraph", END)
    graph.add_edge("company_info", END)
    graph.add_edge("other", END)
    return graph.compile()
