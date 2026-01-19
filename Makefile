.PHONY: setup train serve test docker-build docker-run

setup:
	python -m venv .venv
	pip install -r requirements.txt

train:
	python -m src.train

serve:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t nyc-ot .

docker-run:
	docker run -p 8000:8000 nyc-ot