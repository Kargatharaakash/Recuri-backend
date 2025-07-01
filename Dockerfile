# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools and dependencies
# build-essential is typically for compiling C extensions, might not be strictly needed for all python packages.
# Keeping it as it was in the original.
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (chromium only)
# Ensure Playwright and its browsers are installed correctly.
RUN pip install playwright && playwright install chromium

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages and app code
# This ensures a minimal final image.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY main.py .
COPY agent.py .

# Copy Playwright browser binaries
# This path is crucial for Playwright to find its browsers.
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Corrected ENTRYPOINT and CMD for robust shell variable expansion
# ENTRYPOINT sets the interpreter for the CMD.
ENTRYPOINT ["/bin/sh", "-c"]
# CMD provides the command string to be executed by the ENTRYPOINT shell.
# The shell will expand ${PORT:-8000} before passing it to uvicorn.
CMD ["uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
