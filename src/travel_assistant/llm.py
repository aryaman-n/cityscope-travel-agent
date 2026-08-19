from __future__ import annotations

import json
import re

from openai import APIError, OpenAI

from .config import Settings

FOLLOW_UP_RE = re.compile(r"\b(next week|forecast|weather|temperature|rain|warmer|colder)\b", re.IGNORECASE)
CITY_PATTERNS = [
    re.compile(r"\b(?:about|in|for|visit|visiting|to)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:[?.!,]|$)", re.IGNORECASE),
    re.compile(r"^\s*([A-Za-z][A-Za-z .'-]{1,40}?)\s*[?.!]*$"),
]


def _clean_city(value: str) -> str:
    value = re.sub(
        r"\b(?:what|about|the|next week|this week|weather|forecast|please)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return " ".join(word.capitalize() for word in value.strip(" ,.!?").split())


def deterministic_city(query: str) -> str | None:
    for pattern in CITY_PATTERNS:
        match = pattern.search(query)
        if match:
            city = _clean_city(match.group(1))
            if city and city.casefold() not in {"me", "there", "it", "the", "weather", "next week"}:
                return city
    return None


def resolve_city(query: str, previous_city: str | None, settings: Settings) -> tuple[str, str]:
    explicit_city = deterministic_city(query)
    if explicit_city:
        return explicit_city, "full"
    if previous_city and FOLLOW_UP_RE.search(query):
        return previous_city, "weather_only"
    if settings.use_llm and settings.openai_api_key:
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": 'Extract the requested city. Return JSON: {"city": string}.',
                    },
                    {"role": "user", "content": query},
                ],
            )
            city = _clean_city(
                json.loads(response.choices[0].message.content or "{}").get("city", "")
            )
            if city:
                return city, "full"
        except (APIError, KeyError, TypeError, ValueError):
            pass
    raise ValueError("Please include a city, for example: 'Tell me about Kyoto'.")


def summarize(city: str, context: str, settings: Settings) -> str:
    if not settings.use_llm or not settings.openai_api_key:
        return context.strip()
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "Write a factual 90-130 word travel overview using only the supplied context. Do not invent facts.",
                },
                {"role": "user", "content": f"City: {city}\n\nContext:\n{context}"},
            ],
        )
        return (response.choices[0].message.content or context).strip()
    except (APIError, KeyError, TypeError, ValueError):
        return context.strip()
