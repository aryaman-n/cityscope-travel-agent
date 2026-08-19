from travel_assistant.config import Settings
from travel_assistant.vector_store import CityVectorStore

if __name__ == "__main__":
    settings = Settings.from_env()
    store = CityVectorStore(settings.vector_db_path)
    print(f"Ready: {store.collection.count()} city documents at {settings.vector_db_path}")

