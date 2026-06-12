FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy source
COPY . .

RUN chmod +x start.sh

ENV PATH="/app/.venv/bin:$PATH"

CMD ["bash", "start.sh"]
