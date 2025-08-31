#!/bin/bash
# Setup Docker services for production readiness

set -e

echo "🐳 Setting up Docker services for production..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing Docker..."
    
    # Install Docker on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        echo "Or install via Homebrew: brew install --cask docker"
        exit 1
    fi
    
    # Install Docker on Linux
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Please log out and back in for Docker permissions to take effect"
    exit 1
fi

echo "✅ Docker is installed"

# Start Docker if not running
if ! docker info &> /dev/null; then
    echo "🔄 Starting Docker..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker
        echo "⏳ Waiting for Docker to start..."
        while ! docker info &> /dev/null; do
            sleep 2
        done
    else
        sudo systemctl start docker
    fi
fi

echo "✅ Docker is running"

# Create Docker network
echo "🌐 Creating Docker network..."
docker network create shubham-api-network 2>/dev/null || echo "Network already exists"

# Start ScyllaDB
echo "🚀 Starting ScyllaDB..."
docker run -d \
    --name scylladb-node \
    --network shubham-api-network \
    -p 9042:9042 \
    -p 9160:9160 \
    -p 10000:10000 \
    --memory="2g" \
    --cpus="2" \
    scylladb/scylla:latest \
    --smp 2 \
    --memory 1G \
    --overprovisioned 1 \
    --api-address 0.0.0.0

echo "⏳ Waiting for ScyllaDB to be ready..."
sleep 30

# Wait for ScyllaDB to be ready
for i in {1..30}; do
    if docker exec scylladb-node cqlsh -e "SELECT now() FROM system.local" &> /dev/null; then
        echo "✅ ScyllaDB is ready"
        break
    fi
    echo "⏳ Waiting for ScyllaDB... ($i/30)"
    sleep 5
done

# Create keyspace and tables
echo "📊 Setting up ScyllaDB schema..."
docker exec scylladb-node cqlsh -e "
CREATE KEYSPACE IF NOT EXISTS global_api 
WITH REPLICATION = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

USE global_api;

CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY,
    title TEXT,
    content TEXT,
    author_id UUID,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_deleted BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT,
    name TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_deleted BOOLEAN DEFAULT false
);
"

echo "✅ ScyllaDB setup complete"

# Start Redis for caching
echo "🚀 Starting Redis..."
docker run -d \
    --name redis-cache \
    --network shubham-api-network \
    -p 6379:6379 \
    --memory="512m" \
    redis:alpine \
    redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

echo "✅ Redis setup complete"

# Show running containers
echo "📋 Running containers:"
docker ps --filter "network=shubham-api-network" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎉 Docker services setup complete!"
echo "📝 Services available:"
echo "   - ScyllaDB: localhost:9042"
echo "   - Redis: localhost:6379"
echo ""
echo "🔧 To stop services: docker stop scylladb-node redis-cache"
echo "🗑️  To remove services: docker rm scylladb-node redis-cache"