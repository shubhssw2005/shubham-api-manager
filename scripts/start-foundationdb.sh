#!/bin/bash

echo "🚀 STARTING FOUNDATIONDB ULTRA-PERFORMANCE DATABASE"
echo "==================================================="
echo "🔥 Apple's ACID-compliant distributed database"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing FoundationDB containers
echo "🛑 Stopping existing FoundationDB containers..."
docker stop foundationdb-ultra 2>/dev/null || true
docker rm foundationdb-ultra 2>/dev/null || true

# Start FoundationDB
echo "🚀 Starting FoundationDB..."
docker run -d \
    --name foundationdb-ultra \
    -p 4500:4500 \
    -p 4501:4501 \
    -e FDB_NETWORKING_MODE=host \
    -e FDB_COORDINATOR=auto \
    foundationdb/foundationdb:7.1.27

# Wait for FoundationDB to be ready
echo "⏳ Waiting for FoundationDB to be ready..."
echo "   This may take 30-60 seconds for initialization..."

# Function to check if FoundationDB is ready
check_fdb_ready() {
    docker exec foundationdb-ultra fdbcli --exec "status" > /dev/null 2>&1
    return $?
}

# Wait up to 2 minutes for FoundationDB to be ready
TIMEOUT=120
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    if check_fdb_ready; then
        echo "✅ FoundationDB is ready!"
        break
    fi
    
    echo "   Still waiting... (${ELAPSED}s/${TIMEOUT}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "❌ FoundationDB failed to start within ${TIMEOUT} seconds"
    echo "   Check logs: docker logs foundationdb-ultra"
    exit 1
fi

# Show connection info
echo ""
echo "🎉 FOUNDATIONDB STARTED SUCCESSFULLY!"
echo "===================================="
echo "📊 Connection Details:"
echo "   Host: localhost"
echo "   Port: 4500 (client)"
echo "   API Version: 720"
echo "   Cluster File: Auto-generated"
echo ""
echo "🔧 Management Commands:"
echo "   Connect via CLI: docker exec -it foundationdb-ultra fdbcli"
echo "   Check status: docker exec foundationdb-ultra fdbcli --exec 'status'"
echo "   View logs: docker logs foundationdb-ultra"
echo "   Stop: docker stop foundationdb-ultra"
echo ""
echo "⚡ FOUNDATIONDB FEATURES:"
echo "   ✅ ACID transactions with strict consistency"
echo "   ✅ Multi-version concurrency control (MVCC)"
echo "   ✅ Distributed transactions across nodes"
echo "   ✅ Apple-grade reliability and performance"
echo "   ✅ Automatic conflict resolution"
echo "   ✅ Linear scalability up to 1000+ nodes"
echo ""
echo "🏆 PERFORMANCE CHARACTERISTICS:"
echo "   ✅ <1ms latency for simple transactions"
echo "   ✅ 10M+ transactions/second at scale"
echo "   ✅ Strict serializability guarantees"
echo "   ✅ Automatic sharding and replication"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start ScyllaDB: ./scripts/start-scylladb.sh"
echo "   2. Initialize databases: npm run setup:ultra-distributed"
echo "   3. Start your app: npm run dev"
echo "   4. Test ultra-performance: cd cpp-system && ./ultra_distributed_generator"

# Create cluster file for local development
echo ""
echo "📁 Creating local cluster configuration..."
mkdir -p ~/.foundationdb
docker exec foundationdb-ultra cat /etc/foundationdb/fdb.cluster > ~/.foundationdb/fdb.cluster 2>/dev/null || echo "docker:docker@localhost:4500" > ~/.foundationdb/fdb.cluster

echo "✅ FoundationDB cluster file created at ~/.foundationdb/fdb.cluster"