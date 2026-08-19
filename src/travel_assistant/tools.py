from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import requests

from .config import Settings
from .schemas import CurrentWeather, WeatherPoint

MOCK_SUMMARIES = {
    "kyoto": "Kyoto, Japan's former imperial capital, is known for wooden machiya houses, Buddhist temples, Shinto shrines, and carefully maintained gardens. Fushimi Inari's torii gates, Kiyomizu-dera, Arashiyama, and the Gion district are common highlights. The compact subway and bus network is useful, though walking is often the best way to explore individual districts. Spring blossoms and autumn foliage are popular and busy; quieter winter travel rewards visitors with smaller crowds.",
    "snohomish": "Snohomish is a small city in Washington State, northeast of Seattle, with a historic downtown beside the Snohomish River. Visitors come for antique shops, local cafes, riverfront walks, nearby farms, and access to the Cascade foothills. The area is easiest to explore by car. Rain is possible throughout much of the year, so layered clothing and a waterproof shell are practical.",
    "lisbon": "Lisbon is Portugal's hilly coastal capital, recognized for tiled facades, yellow trams, miradouros, and neighborhoods such as Alfama and Bairro Alto. Belém's monuments, the riverside, and nearby Sintra make varied day plans possible. Metro, trams, buses, and walking cover most central sights, although steep cobbled streets reward comfortable footwear. Local food traditions include grilled seafood and pastéis de nata.",
}

IMAGE_URLS = {
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1431274172761-fca41d930114?auto=format&fit=crop&w=1400&q=85",
    ],
    "tokyo": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1519501025264-65ba15a82390?auto=format&fit=crop&w=1400&q=85",
    ],
    "new york": [
        "https://images.unsplash.com/photo-1522083165195-3424ed129620?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1496588152823-86ff7695e68f?auto=format&fit=crop&w=1400&q=85",
    ],
    "kyoto": [
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1493997181344-712f2f19d87a?auto=format&fit=crop&w=1400&q=85",
        "https://images.unsplash.com/photo-1528360983277-13d401cdc186?auto=format&fit=crop&w=1400&q=85",
    ],
}
FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1400&q=85",
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1400&q=85",
    "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?auto=format&fit=crop&w=1400&q=85",
]


async def search_city(city: str, settings: Settings) -> str:
    if settings.data_mode == "mock":
        await asyncio.sleep(settings.mock_latency_seconds)
        return MOCK_SUMMARIES.get(
            city.casefold(),
            f"{city} is being served through the mock web-search adapter. This offline fallback intentionally avoids inventing local landmarks or transport details. Switch DATA_MODE to live for a current Wikipedia-backed overview, weather data, and Wikimedia images for {city}.",
        )
    return await asyncio.to_thread(_wikipedia_summary, city, settings.api_timeout_seconds)


def _wikipedia_summary(city: str, timeout: float) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(city)}"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "CityScope/0.1"})
    response.raise_for_status()
    summary = response.json().get("extract")
    if not summary:
        raise RuntimeError(f"Wikipedia returned no summary for {city}")
    return summary


async def get_weather(
    city: str, settings: Settings
) -> tuple[CurrentWeather, list[WeatherPoint]]:
    if settings.data_mode == "mock":
        await asyncio.sleep(settings.mock_latency_seconds)
        return _mock_weather(city)
    return await asyncio.to_thread(_open_meteo_weather, city, settings.api_timeout_seconds)


def _mock_weather(city: str) -> tuple[CurrentWeather, list[WeatherPoint]]:
    seed = int.from_bytes(hashlib.sha256(city.casefold().encode()).digest()[:4], "big")
    base = 14 + seed % 15
    conditions = ["Clear", "Partly cloudy", "Cloudy", "Light rain", "Clear", "Breezy", "Partly cloudy"]
    forecast = []
    for offset in range(7):
        variation = ((seed >> (offset * 2)) % 7) - 3
        low = float(base + variation - 4)
        forecast.append(
            WeatherPoint(
                date=datetime.now(UTC).date() + timedelta(days=offset + 1),
                temperature_min_c=low,
                temperature_max_c=low + 7 + (offset % 2),
                condition=conditions[(seed + offset) % len(conditions)],
            )
        )
    current = CurrentWeather(
        temperature_c=float(base),
        condition=conditions[seed % len(conditions)],
        humidity_percent=45 + seed % 41,
    )
    return current, forecast


def _open_meteo_weather(city: str, timeout: float) -> tuple[CurrentWeather, list[WeatherPoint]]:
    geocode = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=timeout,
    )
    geocode.raise_for_status()
    locations = geocode.json().get("results") or []
    if not locations:
        raise RuntimeError(f"Open-Meteo could not locate {city}")
    location = locations[0]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload["current"]
    daily = payload["daily"]
    current_weather = CurrentWeather(
        temperature_c=current["temperature_2m"],
        condition=_weather_code(current["weather_code"]),
        humidity_percent=current["relative_humidity_2m"],
    )
    forecast = [
        WeatherPoint(
            date=day,
            temperature_min_c=low,
            temperature_max_c=high,
            condition=_weather_code(code),
        )
        for day, low, high, code in zip(
            daily["time"], daily["temperature_2m_min"], daily["temperature_2m_max"], daily["weather_code"]
        )
    ]
    return current_weather, forecast


def _weather_code(code: int) -> str:
    if code == 0:
        return "Clear"
    if code <= 3:
        return "Partly cloudy"
    if code <= 48:
        return "Foggy"
    if code <= 67:
        return "Rain"
    if code <= 77:
        return "Snow"
    if code <= 82:
        return "Showers"
    return "Thunderstorms"


async def get_images(city: str, settings: Settings) -> list[str]:
    if settings.data_mode == "mock":
        await asyncio.sleep(settings.mock_latency_seconds)
        return IMAGE_URLS.get(city.casefold(), FALLBACK_IMAGES)
    return await asyncio.to_thread(_wikimedia_images, city, settings.api_timeout_seconds)


def _wikimedia_images(city: str, timeout: float) -> list[str]:
    response = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{city} city landmark filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 8,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1400,
            "format": "json",
            "origin": "*",
        },
        timeout=timeout,
        headers={"User-Agent": "CityScope/0.1"},
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    urls = [
        page["imageinfo"][0].get("thumburl", page["imageinfo"][0]["url"])
        for page in pages.values()
        if page.get("imageinfo")
    ]
    if not urls:
        raise RuntimeError(f"Wikimedia returned no images for {city}")
    return urls[:3]
