#!/bin/bash
# Stop the kt-llamafactory container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping kt-llamafactory container..."
docker compose down

echo "Container stopped."
