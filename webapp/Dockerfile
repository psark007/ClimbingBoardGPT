FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY images ./images
COPY webapp ./webapp

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic \
    && pip install --no-cache-dir -e .

EXPOSE 8055

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8055"]