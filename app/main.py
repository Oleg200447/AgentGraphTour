from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.postgres import create_postgres_pool
from app.graphs.deps import GraphDependencies
from app.graphs.main_graph import build_main_graph
from app.graphs.subgraphs.offer_agent import build_offer_agent_graph
from app.graphs.subgraphs.tourism_agent import build_tourism_agent_graph
from app.services.agent_runtime import AgentRuntime
from app.services.callback_sender import CallbackSender
from app.services.hotels_repository import HotelsRepository
from app.services.llm_factory import LLMFactory
from app.services.thread_state_repository import ThreadStateRepository
from app.services.tour_search_client import TourSearchClient

settings = get_settings()
setup_logging(log_level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan initialization")

    postgres_pool = await create_postgres_pool(settings.database_url)
    logger.info("Postgres pool initialized")

    hotels_repository = HotelsRepository(postgres_pool)
    tour_search_client = TourSearchClient(
        base_url=settings.tour_search_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    callback_sender = CallbackSender(timeout_seconds=settings.callback_timeout_seconds)
    llm_factory = LLMFactory(settings)

    deps = GraphDependencies(
        settings=settings,
        tour_search_client=tour_search_client,
        hotels_repository=hotels_repository,
        router_llm=llm_factory.build_router_llm(),
        agent_llm=llm_factory.build_agent_llm(),
    )

    deps.offer_graph = await build_offer_agent_graph(deps)
    deps.tourism_graph = await build_tourism_agent_graph(deps)
    main_graph = await build_main_graph(deps)
    logger.info("Graphs initialized")

    thread_repo = ThreadStateRepository(postgres_pool, table_name=settings.state_table_name)
    await thread_repo.init_table()
    logger.info("Thread state table is ready", extra={"table": settings.state_table_name})

    app.state.settings = settings
    app.state.postgres_pool = postgres_pool
    app.state.agent_runtime = AgentRuntime(
        main_graph=main_graph,
        thread_repo=thread_repo,
        callback_sender=callback_sender,
        debug_skip_callback=settings.debug_skip_callback,
    )
    logger.info("Agent runtime initialized")

    try:
        yield
    finally:
        logger.info("Shutting down application")
        await postgres_pool.close()
        logger.info("Postgres pool closed")


app = FastAPI(title="TravelAgent Graph API", lifespan=lifespan)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
