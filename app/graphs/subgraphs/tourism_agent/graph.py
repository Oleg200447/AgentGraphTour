from __future__ import annotations

import logging
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

from app.graphs.deps import GraphDependencies

logger = logging.getLogger(__name__)


TOURISM_AGENT_PROMPT = (
    "Ты полезный ассистент по туризму. Отвечай на общие туристические вопросы: "
    "погода, сезонность, популярные направления, визовые базовые рекомендации, "
    "что посмотреть и когда лучше ехать. При необходимости используй поиск. "
    "Отвечай кратко и по делу на русском языке."
)


async def build_tourism_agent_graph(deps: GraphDependencies) -> Any | None:
    if deps.agent_llm is None:
        logger.warning("Tourism graph is disabled because agent_llm is None")
        return None

    tools = [DuckDuckGoSearchRun()]
    logger.info("Building tourism graph", extra={"tools_count": len(tools)})
    return create_react_agent(
        model=deps.agent_llm,
        tools=tools,
        prompt=TOURISM_AGENT_PROMPT,
    )
