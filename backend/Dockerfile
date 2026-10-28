FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Entrypoint: init ChromaDB on first run, then start the server
RUN printf '#!/bin/sh\npython -m scripts.init_chroma\nexec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1\n' > /srv/entrypoint.sh \
    && chmod +x /srv/entrypoint.sh

# One worker only: embedded Chroma is single-process.
CMD ["/srv/entrypoint.sh"]