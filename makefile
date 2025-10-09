.PHONY: run api ui test fmt lint ingest

run: api

api:
	uvicorn app.main:app --reload

ui:
	streamlit run ui/streamlit_app.py

ingest:
	python -m scripts.build_vectorstore

test:
	pytest -q

fmt:
	black app tests

lint:
	ruff check app tests
