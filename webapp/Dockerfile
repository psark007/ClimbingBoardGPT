FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only PyTorch first (separate index, ~200MB vs ~2.5GB)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install only runtime dependencies (no matplotlib, jupyter, etc.)
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e . \
    --no-deps \
    && pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    scikit-learn \
    fastapi \
    "uvicorn[standard]" \
    pydantic

COPY configs ./configs
COPY images ./images
COPY webapp ./webapp

EXPOSE 8055

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8055"]