FROM python:3.11-slim

WORKDIR /app

# Install only what’s needed for build
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install requirements with no cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy only necessary app files
COPY main.py .
COPY agent.py .

# Remove unused cache and build files
RUN apt-get purge -y build-essential && apt-get autoremove -y && apt-get clean

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
