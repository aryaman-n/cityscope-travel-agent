import asyncio

from travel_assistant.config import Settings
from travel_assistant.tools import get_images, get_weather, search_city


def test_mock_tools_return_structured_data(tmp_path) -> None:
    settings = Settings(mock_latency_seconds=0, vector_db_path=tmp_path / "chroma")

    summary = asyncio.run(search_city("Kyoto", settings))
    current, forecast = asyncio.run(get_weather("Kyoto", settings))
    images = asyncio.run(get_images("Kyoto", settings))

    assert "Kyoto" in summary
    assert 5 <= len(forecast) <= 7
    assert current.humidity_percent >= 0
    assert len(images) == 3

