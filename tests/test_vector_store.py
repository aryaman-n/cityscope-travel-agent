from pathlib import Path

from travel_assistant.vector_store import CityVectorStore


def test_vector_store_routes_known_and_unknown_cities(tmp_path: Path) -> None:
    store = CityVectorStore(tmp_path / "chroma")

    tokyo_hits = store.search("Travel information about Tokyo")
    kyoto_hits = store.search("Travel information about Kyoto")

    assert store.collection.count() == 3
    assert tokyo_hits[0]["city"] == "Tokyo"
    assert store.has_knowledge("Tokyo", tokyo_hits)
    assert not store.has_knowledge("Kyoto", kyoto_hits)

