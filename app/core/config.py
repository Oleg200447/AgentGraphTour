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
    log_level: str = "INFO"
    log_json: bool = False

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_router_model: str = "google/gemini-2.5-flash-lite"
    openrouter_agent_model: str = "google/gemini-2.5-flash-lite"

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
        "Турфирма работает ежедневно с 10:00 до 19:00. При желании забронировать тур возмите с собой паспорт и страховку. Если страховки нет - поможем сделать. "
        "Для регистрации на визу необходимо воспользоваться @zxc. Для упрощенного процесса поможет ПРИГЛАШЕНИЕ."
        "Только отвечай на вопрос ничего предлагать не нужно"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
