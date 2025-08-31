#!/bin/bash

# FoundationDB Setup and Start Script
# This script sets up and starts FoundationDB for development/testing

set -e

echo "🚀 Starting FoundationDB Setup..."

# Configuration
FDB_VERSION="7.1.38"
FDB_CLUSTER_FILE="/etc/foundationdb/fdb.cluster"
FDB_DATA_DIR="/var/lib/foundationdb/data"
FDB_LOG_DIR="/var/log/foundationdb"

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

# Install FoundationDB
install_foundationdb() {
    print_status "Installing FoundationDB $FDB_VERSION..."
    
    if [[ "$OS" == "macos" ]]; then
        # macOS installation - download official packages
        if ! command -v fdbserver &> /dev/null; then
            print_status "Downloading FoundationDB for macOS..."
            
            # Create temp directory
            TEMP_DIR=$(mktemp -d)
            cd "$TEMP_DIR"
            
            # Download FoundationDB installer
            print_status "Downloading from GitHub releases..."
            curl -L -o "FoundationDB-${FDB_VERSION}.pkg" \
                "https://github.com/apple/foundationdb/releases/download/${FDB_VERSION}/FoundationDB-${FDB_VERSION}-x86_64.pkg"
            
            # Check if download was successful
            if [[ ! -f "FoundationDB-${FDB_VERSION}.pkg" ]] || [[ ! -s "FoundationDB-${FDB_VERSION}.pkg" ]]; then
                print_warning "GitHub download failed, trying alternative method..."
                # Try Docker approach instead
                print_status "Using Docker to run FoundationDB..."
                if ! command -v docker &> /dev/null; then
                    print_error "Docker is required for this installation method"
                    exit 1
                fi
                
                # Pull and run FoundationDB Docker container
                docker pull foundationdb/foundationdb:${FDB_VERSION}
                
                # Create Docker network if it doesn't exist
                docker network create fdb-network 2>/dev/null || true
                
                # Run FoundationDB container
                docker run -d \
                    --name foundationdb \
                    --network fdb-network \
                    -p 4500:4500 \
                    -v fdb-data:/var/fdb/data \
                    foundationdb/foundationdb:${FDB_VERSION}
                
                print_status "FoundationDB started in Docker container"
                return
            fi
            
            print_status "Installing FoundationDB package..."
            sudo installer -pkg "FoundationDB-${FDB_VERSION}.pkg" -target /
            
            # Return to original directory
            cd - > /dev/null
            rm -rf "$TEMP_DIR"
        else
            print_status "FoundationDB already installed"
        fi
        
    elif [[ "$OS" == "linux" ]]; then
        # Linux installation
        if ! command -v fdbserver &> /dev/null; then
            print_status "Downloading and installing FoundationDB for Linux..."
            
            # Download FoundationDB packages
            wget -q "https://github.com/apple/foundationdb/releases/download/${FDB_VERSION}/foundationdb-clients_${FDB_VERSION}-1_amd64.deb"
            wget -q "https://github.com/apple/foundationdb/releases/download/${FDB_VERSION}/foundationdb-server_${FDB_VERSION}-1_amd64.deb"
            
            # Install packages
            sudo dpkg -i foundationdb-clients_${FDB_VERSION}-1_amd64.deb
            sudo dpkg -i foundationdb-server_${FDB_VERSION}-1_amd64.deb
            
            # Clean up downloaded files
            rm foundationdb-clients_${FDB_VERSION}-1_amd64.deb
            rm foundationdb-server_${FDB_VERSION}-1_amd64.deb
        else
            print_status "FoundationDB already installed"
        fi
    fi
}

# Setup FoundationDB directories and permissions
setup_directories() {
    print_status "Setting up FoundationDB directories..."
    
    if [[ "$OS" == "linux" ]]; then
        # Create necessary directories
        sudo mkdir -p "$FDB_DATA_DIR"
        sudo mkdir -p "$FDB_LOG_DIR"
        sudo mkdir -p "$(dirname "$FDB_CLUSTER_FILE")"
        
        # Set proper ownership
        sudo chown -R foundationdb:foundationdb "$FDB_DATA_DIR"
        sudo chown -R foundationdb:foundationdb "$FDB_LOG_DIR"
    fi
}

# Configure FoundationDB cluster
configure_cluster() {
    print_status "Configuring FoundationDB cluster..."
    
    if [[ "$OS" == "macos" ]]; then
        FDB_CLUSTER_FILE="/usr/local/etc/foundationdb/fdb.cluster"
    fi
    
    # Check if cluster file exists
    if [[ ! -f "$FDB_CLUSTER_FILE" ]]; then
        print_status "Creating cluster configuration..."
        
        # Generate cluster ID
        CLUSTER_ID=$(openssl rand -hex 8)
        
        # Create cluster file
        if [[ "$OS" == "macos" ]]; then
            sudo mkdir -p "$(dirname "$FDB_CLUSTER_FILE")"
            echo "docker:docker@127.0.0.1:4500" | sudo tee "$FDB_CLUSTER_FILE" > /dev/null
        else
            echo "docker:docker@127.0.0.1:4500" | sudo tee "$FDB_CLUSTER_FILE" > /dev/null
        fi
        
        print_status "Cluster file created at $FDB_CLUSTER_FILE"
    else
        print_status "Cluster file already exists at $FDB_CLUSTER_FILE"
    fi
}

# Start FoundationDB services
start_services() {
    print_status "Starting FoundationDB services..."
    
    # Check if running in Docker
    if docker ps | grep foundationdb &> /dev/null; then
        print_status "FoundationDB already running in Docker"
        return
    fi
    
    if [[ "$OS" == "macos" ]]; then
        # Check if native installation exists
        if [[ -f "/Library/LaunchDaemons/com.foundationdb.fdbmonitor.plist" ]]; then
            # macOS service management using launchctl
            if ! launchctl list | grep com.foundationdb &> /dev/null; then
                print_status "Starting FoundationDB service via launchctl..."
                sudo launchctl load /Library/LaunchDaemons/com.foundationdb.fdbmonitor.plist
            else
                print_status "FoundationDB service already running"
            fi
        else
            print_status "No native installation found, services managed by Docker"
        fi
        
    elif [[ "$OS" == "linux" ]]; then
        # Linux systemd service management
        if ! systemctl is-active --quiet foundationdb; then
            print_status "Starting FoundationDB systemd service..."
            sudo systemctl start foundationdb
            sudo systemctl enable foundationdb
        else
            print_status "FoundationDB service already running"
        fi
    fi
}

# Initialize database
initialize_database() {
    print_status "Initializing FoundationDB database..."
    
    # Wait for service to be ready
    sleep 5
    
    # Check if database is already initialized
    if fdbcli --exec "status" &> /dev/null; then
        print_status "Database already initialized and accessible"
        return
    fi
    
    # Initialize new database
    print_status "Configuring new database..."
    fdbcli --exec "configure new single memory"
    
    # Verify initialization
    if fdbcli --exec "status" &> /dev/null; then
        print_status "Database successfully initialized"
    else
        print_error "Failed to initialize database"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying FoundationDB installation..."
    
    # Check if running in Docker
    if docker ps | grep foundationdb &> /dev/null; then
        print_status "✅ FoundationDB is running in Docker container"
        
        # Test connection via Docker
        echo ""
        echo "📊 FoundationDB Status (Docker):"
        docker exec foundationdb fdbcli --exec "status" || {
            print_warning "Database not yet initialized, waiting..."
            sleep 10
            docker exec foundationdb fdbcli --exec "configure new single memory"
            docker exec foundationdb fdbcli --exec "status"
        }
        return
    fi
    
    # Check if fdbcli is available for native installation
    if ! command -v fdbcli &> /dev/null; then
        print_error "fdbcli command not found and no Docker container running"
        exit 1
    fi
    
    # Check database status
    if fdbcli --exec "status" &> /dev/null; then
        print_status "✅ FoundationDB is running and accessible"
        
        # Show status
        echo ""
        echo "📊 FoundationDB Status:"
        fdbcli --exec "status"
        
    else
        print_error "❌ FoundationDB is not accessible"
        exit 1
    fi
}

# Show connection information
show_connection_info() {
    echo ""
    echo "🔗 Connection Information:"
    echo "   Cluster file: $FDB_CLUSTER_FILE"
    echo "   Default connection: 127.0.0.1:4500"
    echo ""
    echo "📝 Usage Examples:"
    echo "   fdbcli                          # Interactive CLI"
    echo "   fdbcli --exec 'status'          # Check status"
    echo "   fdbcli --exec 'writemode on'    # Enable writes"
    echo ""
}

# Cleanup function
cleanup() {
    print_warning "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    echo "🏗️  FoundationDB Setup Script"
    echo "================================"
    
    detect_os
    install_foundationdb
    setup_directories
    configure_cluster
    start_services
    initialize_database
    verify_installation
    show_connection_info
    
    print_status "🎉 FoundationDB setup completed successfully!"
}

# Run main function
main "$@"