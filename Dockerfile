FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Capture git commit hash at build time
ARG GITHUB_SHA=unknown
ENV GITHUB_SHA=${GITHUB_SHA}

# Configure environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/mesh/.venv/bin:$PATH"

# Install the project into `/app`
WORKDIR /app

# Install supervisor for process management, curl for healthchecks, git for dependency installation, libpq-dev and gcc for psycopg2
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache && \
    apt-get update && \
    apt-get install -yqq --no-install-recommends \
    supervisor \
    curl \
    git \
    libpq-dev \
    gcc \
    libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy mesh project files for dependency installation for better caching
COPY mesh/pyproject.toml mesh/uv.lock ./mesh/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    cd mesh && \
    uv sync --frozen --no-install-project --no-dev

# Add supervisor configuration
COPY .docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Add the rest of the project source code
COPY . .

# Make the env-setup script executable
RUN chmod +x /app/.docker/env-setup.sh

# Run requirements check to verify all dependencies are installed correctly
RUN python .docker/requirements_checker.py && echo "Requirements check passed!" || (echo "Requirements check failed!" && exit 1)

# Reset the entrypoint
ENTRYPOINT []

# Run supervisor as the main process
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
