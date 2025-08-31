#!/bin/bash

echo "🚀 STARTING SCYLLADB ULTRA-PERFORMANCE DATABASE"
echo "==============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing ScyllaDB containers
echo "🛑 Stopping existing ScyllaDB containers..."
docker-compose -f docker-compose.scylladb.yml down

# Start ScyllaDB
echo "🚀 Starting ScyllaDB..."
docker-compose -f docker-compose.scylladb.yml up -d scylladb

# Wait for ScyllaDB to be ready
echo "⏳ Waiting for ScyllaDB to be ready..."
echo "   This may take 1-2 minutes for first startup..."

# Function to check if ScyllaDB is ready
check_scylla_ready() {
    docker exec scylladb-ultra-performance cqlsh -e "SELECT now() FROM system.local;" > /dev/null 2>&1
    return $?
}

# Wait up to 3 minutes for ScyllaDB to be ready
TIMEOUT=180
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    if check_scylla_ready; then
        echo "✅ ScyllaDB is ready!"
        break
    fi
    
    echo "   Still waiting... (${ELAPSED}s/${TIMEOUT}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "❌ ScyllaDB failed to start within ${TIMEOUT} seconds"
    echo "   Check logs: docker logs scylladb-ultra-performance"
    exit 1
fi

# Show connection info
echo ""
echo "🎉 SCYLLADB STARTED SUCCESSFULLY!"
echo "================================"
echo "📊 Connection Details:"
echo "   Host: localhost"
echo "   Port: 9042"
echo "   Datacenter: datacenter1"
echo "   Username: cassandra"
echo "   Password: cassandra"
echo ""
echo "🔧 Management Commands:"
echo "   Connect via CQL: docker exec -it scylladb-ultra-performance cqlsh"
echo "   View logs: docker logs scylladb-ultra-performance"
echo "   Stop: docker-compose -f docker-compose.scylladb.yml down"
echo ""
echo "⚡ Performance Features:"
echo "   ✅ Ultra-low latency (sub-millisecond)"
echo "   ✅ High throughput (millions of ops/sec)"
echo "   ✅ C++ native performance"
echo "   ✅ Cassandra-compatible CQL"
echo "   ✅ Automatic sharding and replication"
echo ""
echo "🚀 Next Steps:"
echo "   1. Initialize database: npm run setup:scylladb"
echo "   2. Start your app: npm run dev"
echo "   3. Test performance: cd cpp-system && ./simple_data_generator"

# Optional: Start monitoring if requested
if [ "$1" = "--with-monitoring" ]; then
    echo ""
    echo "📊 Starting ScyllaDB Monitoring..."
    docker-compose -f docker-compose.scylladb.yml --profile monitoring up -d
    echo "✅ Monitoring available at:"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
fi