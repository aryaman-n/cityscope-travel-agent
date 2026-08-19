from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class VectorHit(TypedDict):
    document: str
    city: str
    distance: float


class TravelState(TypedDict, total=False):
    user_query: str
    city: str
    intent: Literal["full", "weather_only"]
    source: Literal["vector_store", "web_search"]
    vector_hits: list[VectorHit]
    knowledge_available: bool
    retrieved_context: str
    city_summary: str
    current_weather: dict
    weather_forecast: list[dict]
    image_urls: list[str]
    warnings: Annotated[list[str], operator.add]
    final_response: dict
