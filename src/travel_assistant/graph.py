from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import resolve_city, summarize
from .schemas import TravelResponse
from .state import TravelState
from .tools import get_images, get_weather, search_city
from .vector_store import CityVectorStore


def build_graph(settings: Settings, vector_store: CityVectorStore):
    async def analyze_request(state: TravelState) -> dict:
        city, intent = resolve_city(state["user_query"], state.get("city"), settings)
        return {"city": city, "intent": intent, "warnings": []}

    def route_intent(state: TravelState) -> Literal["check_knowledge", "refresh_weather"]:
        return "refresh_weather" if state["intent"] == "weather_only" else "check_knowledge"

    def check_knowledge(state: TravelState) -> dict:
        hits = vector_store.search(f"Travel information about {state['city']}")
        return {
            "vector_hits": hits,
            "knowledge_available": vector_store.has_knowledge(state["city"], hits),
        }

    def route_source(state: TravelState) -> Literal["retrieve_vector", "search_web"]:
        return "retrieve_vector" if state["knowledge_available"] else "search_web"

    def retrieve_vector(state: TravelState) -> dict:
        best = state["vector_hits"][0]
        return {"source": "vector_store", "retrieved_context": best["document"]}

    async def search_web(state: TravelState) -> dict:
        context = await search_city(state["city"], settings)
        return {"source": "web_search", "retrieved_context": context}

    def prepare_summary(state: TravelState) -> dict:
        return {"city_summary": summarize(state["city"], state["retrieved_context"], settings)}

    async def fetch_weather(state: TravelState) -> dict:
        try:
            current, forecast = await get_weather(state["city"], settings)
            return {
                "current_weather": current.model_dump(mode="json"),
                "weather_forecast": [point.model_dump(mode="json") for point in forecast],
            }
        except Exception as exc:  # noqa: BLE001 - provider failures must degrade gracefully
            _, fallback = await get_weather(
                state["city"], Settings(mock_latency_seconds=0, vector_db_path=settings.vector_db_path)
            )
            return {
                "weather_forecast": [point.model_dump(mode="json") for point in fallback],
                "warnings": [f"Weather provider failed; showing mock data ({exc})."],
            }

    async def fetch_images(state: TravelState) -> dict:
        try:
            return {"image_urls": await get_images(state["city"], settings)}
        except Exception as exc:  # noqa: BLE001 - provider failures must degrade gracefully
            return {
                "image_urls": await get_images(
                    state["city"], Settings(mock_latency_seconds=0, vector_db_path=settings.vector_db_path)
                ),
                "warnings": [f"Image provider failed; showing fallback images ({exc})."],
            }

    def finalize(state: TravelState) -> dict:
        response = TravelResponse(
            city=state["city"],
            source=state["source"],
            city_summary=state["city_summary"],
            current_weather=state.get("current_weather"),
            weather_forecast=state["weather_forecast"],
            image_urls=state.get("image_urls", []),
            warnings=state.get("warnings", []),
        )
        # Checkpoint a plain JSON-compatible object; the service boundary restores the model.
        return {"final_response": response.model_dump(mode="json")}

    builder = StateGraph(TravelState)
    builder.add_node("analyze_request", analyze_request)
    builder.add_node("check_knowledge", check_knowledge)
    builder.add_node("retrieve_vector", retrieve_vector)
    builder.add_node("search_web", search_web)
    builder.add_node("prepare_summary", prepare_summary)
    builder.add_node("fetch_weather", fetch_weather)
    builder.add_node("fetch_images", fetch_images)
    builder.add_node("refresh_weather", fetch_weather)
    builder.add_node("finalize", finalize)
    builder.add_node("finalize_refresh", finalize)

    builder.add_edge(START, "analyze_request")
    builder.add_conditional_edges("analyze_request", route_intent)
    builder.add_conditional_edges("check_knowledge", route_source)
    builder.add_edge("retrieve_vector", "prepare_summary")
    builder.add_edge("search_web", "prepare_summary")
    # LangGraph fans these independent nodes out in parallel and joins at finalize.
    builder.add_edge("prepare_summary", "fetch_weather")
    builder.add_edge("prepare_summary", "fetch_images")
    builder.add_edge(["fetch_weather", "fetch_images"], "finalize")
    builder.add_edge("refresh_weather", "finalize_refresh")
    builder.add_edge("finalize", END)
    builder.add_edge("finalize_refresh", END)
    return builder.compile(checkpointer=MemorySaver())
