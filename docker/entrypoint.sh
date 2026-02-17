#!/bin/bash
set -e

# Wait for MinIO to be ready
echo "Waiting for MinIO..."
until curl -sf http://minio:9000/minio/health/live; do
  sleep 2
done
echo "MinIO is ready"

# Create warehouse bucket if it doesn't exist
mc alias set myminio http://minio:9000 minioadmin minioadmin 2>/dev/null || true
mc mb myminio/warehouse 2>/dev/null || true

echo "MinIO bucket 'warehouse' ready"
exec "$@"
