#!/bin/bash
# Production startup script

set -e

echo "🚀 Starting production environment..."

# Load production environment
export NODE_ENV=production
source .env.production

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2)
REQUIRED_VERSION="18.0.0"
if ! node -e "process.exit(require('semver').gte('$NODE_VERSION', '$REQUIRED_VERSION') ? 0 : 1)" 2>/dev/null; then
    echo "❌ Node.js version $REQUIRED_VERSION or higher required. Current: $NODE_VERSION"
    exit 1
fi
echo "✅ Node.js version: $NODE_VERSION"

# Check if services are running
echo "🔍 Checking services..."

# Check MongoDB
if ! nc -z localhost 27017 2>/dev/null; then
    echo "⚠️  MongoDB not running on localhost:27017"
    echo "   Starting MongoDB..."
    # Try to start MongoDB (adjust for your system)
    if command -v brew &> /dev/null; then
        brew services start mongodb-community
    elif command -v systemctl &> /dev/null; then
        sudo systemctl start mongod
    else
        echo "   Please start MongoDB manually"
    fi
fi

# Check ScyllaDB
if ! nc -z localhost 9042 2>/dev/null; then
    echo "⚠️  ScyllaDB not running. Starting Docker services..."
    ./scripts/setup-docker-services.sh
fi

# Check Redis
if ! nc -z localhost 6379 2>/dev/null; then
    echo "⚠️  Redis not running. It should be started by Docker services."
fi

# Build application
echo "🔨 Building application..."
npm run build

# Run database migrations
echo "📊 Running database migrations..."
npm run migrate

# Start the application
echo "🚀 Starting application in production mode..."
npm start