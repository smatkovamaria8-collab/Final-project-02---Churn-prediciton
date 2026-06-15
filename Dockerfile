FROM python:3.13-slim

ENV APP_HOME=/app \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR $APP_HOME

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --only main

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "02_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
