from __future__ import annotations

from uuid import uuid4

from .config import Settings
from .graph import build_graph
from .schemas import TravelResponse
from .vector_store import CityVectorStore


class TravelAssistant:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.vector_store = CityVectorStore(self.settings.vector_db_path)
        self.graph = build_graph(self.settings, self.vector_store)

    async def ask(self, query: str, thread_id: str | None = None) -> TravelResponse:
        if not query.strip():
            raise ValueError("The request cannot be empty.")
        config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
        state = await self.graph.ainvoke({"user_query": query.strip()}, config=config)
        return TravelResponse.model_validate(state["final_response"])
