from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8080

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_router_model: str = "openai/gpt-4o-mini"
    openrouter_agent_model: str = "openai/gpt-4o-mini"

    database_url: str = "postgresql+asyncpg://admin:YOUR_POSTGRES_PASSWORD@tours_db:5432/tours"
    state_table_name: str = "agent_thread_state"

    tour_search_base_url: str = "http://tour_search:8000"

    request_timeout_seconds: float = 25.0
    callback_timeout_seconds: float = 10.0
    debug_skip_callback: bool = False

    tour_firm_system_prompt: str = (
        "Ты ассистент турфирмы. Отвечай только на вопросы о работе турфирмы, "
        "офисе, документах, процессах и обслуживании клиентов. "
        "Если данных недостаточно — честно скажи, что нужно уточнение."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
