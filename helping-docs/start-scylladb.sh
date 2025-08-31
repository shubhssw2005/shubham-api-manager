#!/bin/bash

# ScyllaDB Setup and Start Script
# High-performance NoSQL database (Cassandra-compatible)

set -e

echo "🚀 Starting ScyllaDB Setup..."

# Configuration
SCYLLA_VERSION="5.4"
SCYLLA_DATA_DIR="./scylladb/data"
SCYLLA_LOG_DIR="./scylladb/logs"
SCYLLA_CONFIG_DIR="./scylladb/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Setup ScyllaDB directories
setup_directories() {
    print_status "Setting up ScyllaDB directories..."
    
    mkdir -p "$SCYLLA_DATA_DIR"
    mkdir -p "$SCYLLA_LOG_DIR"
    mkdir -p "$SCYLLA_CONFIG_DIR"
    
    print_status "Directories created"
}

# Install ScyllaDB via Docker
install_scylladb_docker() {
    print_status "Installing ScyllaDB via Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required for ScyllaDB installation"
        print_status "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if ScyllaDB container already exists
    if docker ps -a | grep scylladb-node &> /dev/null; then
        print_status "ScyllaDB container already exists"
        
        # Start if stopped
        if ! docker ps | grep scylladb-node &> /dev/null; then
            print_status "Starting existing ScyllaDB container..."
            docker start scylladb-node
        else
            print_status "ScyllaDB container already running"
        fi
        return
    fi
    
    # Pull ScyllaDB image
    print_status "Pulling ScyllaDB Docker image..."
    docker pull scylladb/scylla:$SCYLLA_VERSION
    
    # Create and start ScyllaDB container
    print_status "Creating ScyllaDB container..."
    docker run -d \
        --name scylladb-node \
        --hostname scylladb-node \
        -p 9042:9042 \
        -p 9160:9160 \
        -p 10000:10000 \
        -v "$(pwd)/$SCYLLA_DATA_DIR:/var/lib/scylla" \
        -v "$(pwd)/$SCYLLA_LOG_DIR:/var/log/scylla" \
        --restart unless-stopped \
        scylladb/scylla:$SCYLLA_VERSION \
        --seeds=scylladb-node \
        --smp 2 \
        --memory 2G \
        --overprovisioned 1 \
        --api-address 0.0.0.0
    
    print_status "ScyllaDB container created and started"
}

# Wait for ScyllaDB to be ready
wait_for_scylladb() {
    print_status "Waiting for ScyllaDB to be ready..."
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec scylladb-node cqlsh -e "DESCRIBE KEYSPACES" &> /dev/null; then
            print_status "✅ ScyllaDB is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
    
    print_error "❌ ScyllaDB failed to start within expected time"
    exit 1
}

# Initialize ScyllaDB schema
initialize_schema() {
    print_status "Initializing ScyllaDB schema..."
    
    # Create keyspace and tables
    docker exec scylladb-node cqlsh -e "
        CREATE KEYSPACE IF NOT EXISTS global_api
        WITH REPLICATION = {
            'class': 'SimpleStrategy',
            'replication_factor': 1
        };
        
        USE global_api;
        
        -- Posts table
        CREATE TABLE IF NOT EXISTS posts (
            id UUID PRIMARY KEY,
            title TEXT,
            slug TEXT,
            content TEXT,
            excerpt TEXT,
            status TEXT,
            published_at TIMESTAMP,
            tags SET<TEXT>,
            author_id UUID,
            view_count COUNTER,
            like_count COUNTER,
            comment_count COUNTER,
            featured BOOLEAN,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by UUID,
            updated_by UUID,
            is_deleted BOOLEAN,
            deleted_at TIMESTAMP,
            deleted_by UUID,
            tombstone_reason TEXT
        );
        
        -- Products table
        CREATE TABLE IF NOT EXISTS products (
            id UUID PRIMARY KEY,
            name TEXT,
            slug TEXT,
            description TEXT,
            short_description TEXT,
            sku TEXT,
            price DECIMAL,
            compare_at_price DECIMAL,
            cost DECIMAL,
            quantity INT,
            status TEXT,
            visibility TEXT,
            featured BOOLEAN,
            tags SET<TEXT>,
            category_id UUID,
            brand_id UUID,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by UUID,
            updated_by UUID,
            is_deleted BOOLEAN,
            deleted_at TIMESTAMP,
            deleted_by UUID,
            tombstone_reason TEXT
        );
        
        -- Orders table
        CREATE TABLE IF NOT EXISTS orders (
            id UUID PRIMARY KEY,
            order_number TEXT,
            customer_id UUID,
            subtotal DECIMAL,
            tax DECIMAL,
            shipping DECIMAL,
            discount DECIMAL,
            total DECIMAL,
            currency TEXT,
            status TEXT,
            payment_status TEXT,
            fulfillment_status TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by UUID,
            updated_by UUID,
            is_deleted BOOLEAN,
            deleted_at TIMESTAMP,
            deleted_by UUID,
            tombstone_reason TEXT
        );
        
        -- Customers table
        CREATE TABLE IF NOT EXISTS customers (
            id UUID PRIMARY KEY,
            email TEXT,
            first_name TEXT,
            last_name TEXT,
            phone TEXT,
            is_active BOOLEAN,
            email_verified BOOLEAN,
            phone_verified BOOLEAN,
            accepts_marketing BOOLEAN,
            total_spent DECIMAL,
            order_count INT,
            last_order_at TIMESTAMP,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by UUID,
            updated_by UUID,
            is_deleted BOOLEAN,
            deleted_at TIMESTAMP,
            deleted_by UUID,
            tombstone_reason TEXT
        );
        
        -- Categories table
        CREATE TABLE IF NOT EXISTS categories (
            id UUID PRIMARY KEY,
            name TEXT,
            slug TEXT,
            description TEXT,
            parent_id UUID,
            is_active BOOLEAN,
            sort_order INT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by UUID,
            updated_by UUID,
            is_deleted BOOLEAN,
            deleted_at TIMESTAMP,
            deleted_by UUID,
            tombstone_reason TEXT
        );
    "
    
    # Create indexes
    docker exec scylladb-node cqlsh -k global_api -e "
        CREATE INDEX IF NOT EXISTS ON posts (status);
        CREATE INDEX IF NOT EXISTS ON posts (author_id);
        CREATE INDEX IF NOT EXISTS ON posts (published_at);
        CREATE INDEX IF NOT EXISTS ON products (status);
        CREATE INDEX IF NOT EXISTS ON products (category_id);
        CREATE INDEX IF NOT EXISTS ON products (sku);
        CREATE INDEX IF NOT EXISTS ON orders (customer_id);
        CREATE INDEX IF NOT EXISTS ON orders (status);
        CREATE INDEX IF NOT EXISTS ON customers (email);
        CREATE INDEX IF NOT EXISTS ON categories (parent_id);
    "
    
    print_status "Schema initialized successfully"
}

# Show connection information
show_connection_info() {
    echo ""
    echo "🔗 ScyllaDB Connection Information:"
    echo "   Host: localhost"
    echo "   Port: 9042 (CQL)"
    echo "   Keyspace: global_api"
    echo "   Docker Container: scylladb-node"
    echo ""
    echo "📝 Usage Examples:"
    echo "   docker exec -it scylladb-node cqlsh                    # Interactive CQL shell"
    echo "   docker exec scylladb-node cqlsh -e 'DESCRIBE TABLES'   # List tables"
    echo "   docker logs scylladb-node                               # View logs"
    echo "   docker exec scylladb-node nodetool status              # Cluster status"
    echo ""
    echo "🌐 Web Interface:"
    echo "   Monitoring: http://localhost:10000"
    echo ""
}

# Verify installation
verify_installation() {
    print_status "Verifying ScyllaDB installation..."
    
    # Check if container is running
    if ! docker ps | grep scylladb-node &> /dev/null; then
        print_error "❌ ScyllaDB container is not running"
        exit 1
    fi
    
    # Check if CQL is accessible
    if docker exec scylladb-node cqlsh -e "SELECT now() FROM system.local" &> /dev/null; then
        print_status "✅ ScyllaDB is running and accessible"
        
        # Show cluster status
        echo ""
        echo "📊 ScyllaDB Cluster Status:"
        docker exec scylladb-node nodetool status
        
    else
        print_error "❌ ScyllaDB is not accessible via CQL"
        exit 1
    fi
}

# Main execution
main() {
    echo "🏗️  ScyllaDB Setup Script"
    echo "========================="
    
    detect_os
    setup_directories
    install_scylladb_docker
    wait_for_scylladb
    initialize_schema
    verify_installation
    show_connection_info
    
    print_status "🎉 ScyllaDB setup completed successfully!"
}

# Run main function
main "$@"