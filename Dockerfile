FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY artifacts /app/artifacts

ENV PORT=7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]