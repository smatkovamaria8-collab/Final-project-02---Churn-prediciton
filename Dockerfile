FROM python:3.11-slim

ENV APP_HOME=/app

WORKDIR $APP_HOME

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "02_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
