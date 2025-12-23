# ============================================================================
# STAGE 1: BUILDER
# ============================================================================
FROM python:3.11.7-slim-bookworm AS builder

# Build arguments
ARG INSTALL_LLM=false
ARG BUILD_ENV=production
ARG PIP_TIMEOUT=600

# Environment for build stage
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install NumPy (prevents conflicts)
RUN pip install --no-cache-dir numpy==1.26.4

# Build Python wheels for faster installation
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    --timeout=${PIP_TIMEOUT} -r requirements.txt

# ============================================================================
# STAGE 2: RUNTIME
# ============================================================================
FROM python:3.11.7-slim-bookworm AS runtime

# Metadata labels
LABEL maintainer="lexai-team@company.com" \
      version="2.0.0" \
      description="LexAI Regulatory Compliance Analysis Platform" \
      org.opencontainers.image.source="https://github.com/yourorg/lexai"

# Build arguments
ARG INSTALL_LLM=false
ARG BUILD_ENV=production

# Runtime environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    USE_LOCAL_ENGINE=true \
    COMPLIANCE_ENGINE=local \
    PYTHONHASHSEED=random

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    curl \
    ca-certificates \
    # Document processing
    libmagic1 \
    # OCR support (CRITICAL FOR DOCUMENT ANALYSIS)
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    # PDF processing
    poppler-utils \
    # Image processing (runtime libraries only)
    libjpeg62-turbo \
    libpng16-16 \
    # XML processing (runtime libraries only)
    libxml2 \
    libxslt1.1 \
    # Math libraries (runtime only)
    libblas3 \
    liblapack3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy and install pre-built wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Create non-root user and group
RUN groupadd -r lexai --gid=1000 \
    && useradd -r -g lexai --uid=1000 --home-dir=/app --shell=/sbin/nologin lexai \
    && mkdir -p /app/logs \
    && chown -R lexai:lexai /app

# Copy application files with correct ownership
# Following the actual project structure:
# ├── assets/
# ├── data/
# ├── docs/
# ├── utils/
# └── root files (*.py, *.toml, *.sh, *.md, *.txt)

COPY --chown=lexai:lexai assets/ ./assets/
COPY --chown=lexai:lexai data/ ./data/
COPY --chown=lexai:lexai utils/ ./utils/
COPY --chown=lexai:lexai docs/ ./docs/
COPY --chown=lexai:lexai *.py ./
COPY --chown=lexai:lexai *.txt ./
COPY --chown=lexai:lexai *.toml ./
COPY --chown=lexai:lexai *.sh ./
COPY --chown=lexai:lexai *.md ./

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER lexai

# Streamlit configuration
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]