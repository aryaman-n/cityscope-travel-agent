from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class WeatherPoint(BaseModel):
    date: date
    temperature_min_c: float
    temperature_max_c: float
    condition: str


class CurrentWeather(BaseModel):
    temperature_c: float
    condition: str
    humidity_percent: int = Field(ge=0, le=100)


class TravelResponse(BaseModel):
    city_summary: str
    weather_forecast: list[WeatherPoint] = Field(min_length=5, max_length=7)
    image_urls: list[HttpUrl]
    city: str
    source: Literal["vector_store", "web_search"]
    current_weather: CurrentWeather | None = None
    warnings: list[str] = Field(default_factory=list)

