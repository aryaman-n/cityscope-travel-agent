from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from pathlib import Path

import chromadb

from .config import PROJECT_ROOT
from .state import VectorHit

TOKEN_RE = re.compile(r"[a-z0-9]+")


def embed_text(text: str, dimensions: int = 512) -> list[float]:
    """Create a stable local feature-hashing embedding without model downloads."""
    tokens = TOKEN_RE.findall(text.lower())
    features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        vector[value % dimensions] += 1.0 if value & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class CityVectorStore:
    collection_name = "city_knowledge"

    def __init__(self, persist_path: Path, data_path: Path | None = None) -> None:
        source_data = PROJECT_ROOT / "data" / "cities"
        packaged_data = Path(__file__).parent / "data" / "cities"
        self.data_path = data_path or (source_data if source_data.exists() else packaged_data)
        persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.ensure_seeded()

    def ensure_seeded(self) -> None:
        files = sorted(self.data_path.glob("*.md"))
        expected_ids = [path.stem for path in files]
        if self.collection.count() == len(files) and set(self.collection.get()["ids"]) == set(expected_ids):
            return
        if self.collection.count():
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        documents = [path.read_text(encoding="utf-8") for path in files]
        cities = [path.stem.replace("_", " ").title() for path in files]
        self.collection.add(
            ids=expected_ids,
            documents=documents,
            metadatas=[{"city": city} for city in cities],
            embeddings=[embed_text(document) for document in documents],
        )

    def search(self, query: str, limit: int = 3) -> list[VectorHit]:
        result = self.collection.query(
            query_embeddings=[embed_text(query)],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            VectorHit(document=document, city=metadata["city"], distance=float(distance))
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def has_knowledge(self, city: str, hits: list[VectorHit]) -> bool:
        """Use retrieval evidence and a conservative distance threshold, not a city allowlist."""
        if not hits:
            return False
        normalized = city.casefold().strip()
        best = hits[0]
        metadata_match = best["city"].casefold() == normalized
        return metadata_match and best["distance"] <= 0.97
