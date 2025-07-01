#!/bin/sh

# Set the PORT environment variable if it's not already set.
# This ensures that uvicorn always receives a valid integer.
# Railway (and similar platforms) will typically set the PORT variable for you.
# This line acts as a fallback for local testing or if the variable isn't set.
PORT=${PORT:-8000}

echo "Starting uvicorn on port: $PORT"

# Execute the uvicorn command.
# Using 'exec' replaces the current shell process with the uvicorn process,
# which is good practice for Docker entrypoints as it ensures signals (like SIGTERM)
# are correctly passed to the application.
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
