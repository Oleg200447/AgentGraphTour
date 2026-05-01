from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings


class LLMFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build(self, model: str, temperature: float) -> Any | None:
        if not self.settings.openrouter_api_key:
            return None
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )

    def build_router_llm(self) -> Any | None:
        return self._build(model=self.settings.openrouter_router_model, temperature=0.0)

    def build_agent_llm(self) -> Any | None:
        return self._build(model=self.settings.openrouter_agent_model, temperature=0.2)
