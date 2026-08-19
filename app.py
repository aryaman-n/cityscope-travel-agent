from __future__ import annotations

import asyncio
from uuid import uuid4

import pandas as pd
import streamlit as st

from travel_assistant import TravelAssistant
from travel_assistant.schemas import TravelResponse

st.set_page_config(page_title="CityScope", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f5f0; color: #172a2b; }
    [data-testid="stHeader"] { background: rgba(247,245,240,.88); }
    .hero { padding: 2.6rem 0 1.3rem; max-width: 850px; }
    .eyebrow { color:#a0472e; font-size:.82rem; letter-spacing:.16em; font-weight:700; text-transform:uppercase; }
    .hero h1 { font-family: Georgia, serif; font-size: clamp(3rem,7vw,5.5rem); line-height:.94; letter-spacing:-.045em; margin:.45rem 0 1rem; }
    .hero p { color:#5d6969; font-size:1.12rem; max-width:700px; }
    .source-pill { display:inline-block; background:#eae4d9; color:#79412f; padding:.28rem .72rem; border-radius:99px; font-size:.78rem; font-weight:700; }
    .summary { font-family: Georgia, serif; font-size:1.25rem; line-height:1.72; max-width:900px; }
    .block-title { margin-top:2.5rem; font-family: Georgia, serif; font-size:2rem; }
    [data-testid="stMetric"] { background:white; border:1px solid #ded8cd; padding:1rem; border-radius:14px; }
    [data-testid="stImage"] img { border-radius:16px; aspect-ratio:4/3; object-fit:cover; }
    .stButton button { border-radius:999px; border-color:#9e5b44; color:#7d402d; }
    .stTextInput label { color:#5d6969 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_assistant() -> TravelAssistant:
    return TravelAssistant()


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def render_response(result: TravelResponse) -> None:
    st.markdown(
        f'<span class="source-pill">{"INTERNAL KNOWLEDGE" if result.source == "vector_store" else "WEB SEARCH"}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"## {result.city}")
    st.markdown(result.city_summary)

    for warning in result.warnings:
        st.warning(warning)

    st.markdown('<div class="block-title">Weather outlook</div>', unsafe_allow_html=True)
    if result.current_weather:
        c1, c2, c3 = st.columns(3)
        c1.metric("Now", f"{result.current_weather.temperature_c:.0f} °C")
        c2.metric("Conditions", result.current_weather.condition)
        c3.metric("Humidity", f"{result.current_weather.humidity_percent}%")

    frame = pd.DataFrame(
        [
            {
                "Date": point.date,
                "Low °C": point.temperature_min_c,
                "High °C": point.temperature_max_c,
                "Condition": point.condition,
            }
            for point in result.weather_forecast
        ]
    ).set_index("Date")
    st.line_chart(frame[["Low °C", "High °C"]], color=["#7d9b9a", "#a0472e"])
    with st.expander("Daily details"):
        st.dataframe(frame, width="stretch")

    if result.image_urls:
        st.markdown('<div class="block-title">A glimpse of the city</div>', unsafe_allow_html=True)
        columns = st.columns(min(3, len(result.image_urls)))
        for index, image_url in enumerate(result.image_urls):
            columns[index % len(columns)].image(str(image_url), width="stretch")

    with st.expander("Structured agent output"):
        st.json(result.model_dump(mode="json"))


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "query_value" not in st.session_state:
    st.session_state.query_value = "Tell me about Tokyo"

with st.sidebar:
    st.subheader("How routing works")
    st.write("Paris, Tokyo, and New York use the local vector store. Other cities use the search adapter.")
    st.caption("Follow up with ‘What about next week?’ to reuse the current city and refresh only weather.")
    if st.button("Start a new trip", width="stretch"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.pop("result", None)
        st.rerun()
    st.divider()
    mode = get_assistant().settings.data_mode
    st.caption(f"Data mode: {mode.upper()}")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Multi-source travel intelligence</div>
      <h1>Meet the city<br>before you arrive.</h1>
      <p>One request, intelligently routed. CityScope combines curated knowledge or live search with a seven-day weather outlook and visual inspiration.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("travel_query", border=False):
    query = st.text_input(
        "Where are you curious about?",
        value=st.session_state.query_value,
        placeholder="Try: Tell me about Kyoto",
    )
    submitted = st.form_submit_button("Explore city  →", type="primary", width="stretch")

if submitted:
    st.session_state.query_value = query
    try:
        with st.spinner("Planning the best route to your answer…"):
            st.session_state.result = run_async(
                get_assistant().ask(query, thread_id=st.session_state.thread_id)
            )
    except Exception as exc:  # noqa: BLE001 - keep the interactive UI alive on provider errors
        st.error(f"I couldn't complete that request: {exc}")

if result := st.session_state.get("result"):
    st.divider()
    render_response(result)
