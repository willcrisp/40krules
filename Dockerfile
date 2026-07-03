FROM python:3.12-slim

# uv - fast Python package installer/resolver
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml ./
COPY config.toml ./
COPY src ./src

# Drop `--extra embeddings` here for a much smaller/faster image if you're
# fine running keyword-only search (the app degrades gracefully — see
# README "Embeddings"). Kept in by default to match the design's bge-small
# default and give real semantic (paraphrase-tolerant) retrieval.
RUN uv sync --no-dev --extra embeddings

# Pre-download the embedding model at build time so container cold starts
# never depend on reaching Hugging Face at runtime.
RUN uv run python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5')"

ENV PATH="/app/.venv/bin:${PATH}"

# Railway (and most PaaS) inject $PORT at runtime; __main__.py reads it.
EXPOSE 8000

CMD ["python", "-m", "rulehound.api"]
