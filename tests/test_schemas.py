from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from travel_assistant.schemas import TravelResponse, WeatherPoint


def test_response_requires_at_least_five_forecast_points() -> None:
    points = [
        WeatherPoint(
            date=datetime.now(UTC).date() + timedelta(days=offset),
            temperature_min_c=10,
            temperature_max_c=18,
            condition="Clear",
        )
        for offset in range(4)
    ]
    with pytest.raises(ValidationError):
        TravelResponse(
            city="Paris",
            source="vector_store",
            city_summary="Summary",
            weather_forecast=points,
            image_urls=[],
        )
