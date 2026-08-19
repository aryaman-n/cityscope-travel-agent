.PHONY: install init run test lint graph

install:
	python -m pip install -e '.[dev]'

init:
	python scripts/init_vector_store.py

run:
	streamlit run app.py

test:
	pytest -q

lint:
	ruff check .

graph:
	python scripts/render_graph.py

