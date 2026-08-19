import asyncio

from travel_assistant.config import Settings
from travel_assistant.service import TravelAssistant


def test_graph_routes_both_paths_and_returns_schema(tmp_path) -> None:
    settings = Settings(mock_latency_seconds=0, vector_db_path=tmp_path / "chroma")
    assistant = TravelAssistant(settings)

    known = asyncio.run(assistant.ask("Tell me about Tokyo", thread_id="known"))
    unknown = asyncio.run(assistant.ask("Tell me about Kyoto", thread_id="unknown"))

    assert known.source == "vector_store"
    assert unknown.source == "web_search"
    assert len(known.weather_forecast) == 7
    assert len(unknown.image_urls) == 3


def test_follow_up_preserves_city_and_summary(tmp_path) -> None:
    settings = Settings(mock_latency_seconds=0, vector_db_path=tmp_path / "chroma")
    assistant = TravelAssistant(settings)

    first = asyncio.run(assistant.ask("Tokyo", thread_id="trip"))
    follow_up = asyncio.run(assistant.ask("What about the weather next week?", thread_id="trip"))

    assert follow_up.city == "Tokyo"
    assert follow_up.city_summary == first.city_summary
    assert follow_up.source == first.source

