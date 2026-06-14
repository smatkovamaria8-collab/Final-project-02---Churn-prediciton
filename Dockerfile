FROM python:3.11-slim

ENV APP_HOME=/app \
POETRY_VIRTUALENVS_CREATE=false

WORKDIR $APP_HOME

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "02_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
