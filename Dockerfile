# DeskPilot AI - serving container
# Multi-stage-free, minimal image: the classical model is a few hundred KB, so a
# heavy base image / GPU runtime is unnecessary - this is intentionally a small,
# fast-starting container suitable for Azure Container Apps / Azure App Service.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps \
    fastapi uvicorn[standard] pydantic numpy pandas scikit-learn scipy joblib

COPY src/ ./src/
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/schema/ ./data/schema/

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
