# Web & PDF RAG on LangChain


## Requirements

- Python 3.8+
- Install dependencies:

```sh
pip install -r requirements.txt
```

## Setup

1. Copy or create a `.env` file at the repo root if environment variables are required.
2. Verify `requirements.txt` is installed.

## Run

Edit and start the main application:

```sh
python main.py
```

## Project structure

- [agent.py](agent.py) — Agent implementation.
- [configs.py](configs.py) — Configuration values and helpers.
- [main.py](main.py) — Entry point for running the application.
- [retriever_builder.py](retriever_builder.py) — Retriever construction logic.
- [requirements.txt](requirements.txt) — Python dependencies.
