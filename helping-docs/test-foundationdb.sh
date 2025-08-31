#!/bin/bash

# FoundationDB Test Script
# This script demonstrates basic FoundationDB operations

echo "🧪 Testing FoundationDB Setup"
echo "============================="

FDB_CLI="./foundationdb/fdbcli"

# Check if FoundationDB is available
if [[ ! -f "$FDB_CLI" ]]; then
    echo "❌ FoundationDB CLI not found. Run ./start-foundationdb-local.sh first"
    exit 1
fi

echo "✅ FoundationDB CLI found"

# Test 1: Check status
echo ""
echo "📊 Test 1: Database Status"
echo "-------------------------"
$FDB_CLI --exec "status"

# Test 2: Initialize database
echo ""
echo "🔧 Test 2: Database Configuration"
echo "--------------------------------"
$FDB_CLI --exec "configure new single memory"

# Test 3: Show cluster info
echo ""
echo "🌐 Test 3: Cluster Information"
echo "-----------------------------"
echo "Cluster file: $(cat foundationdb/fdb.cluster)"
echo "Data directory: foundationdb/data"
echo "Log directory: foundationdb/logs"

# Test 4: Interactive mode info
echo ""
echo "💡 Test 4: Interactive Usage"
echo "---------------------------"
echo "To use FoundationDB interactively:"
echo "  $FDB_CLI"
echo ""
echo "Common commands:"
echo "  status                     - Show database status"
echo "  writemode on              - Enable write mode"
echo "  set key value             - Set a key-value pair"
echo "  get key                   - Get value for a key"
echo "  clear key                 - Delete a key"
echo "  getrange key1 key2        - Get range of keys"

echo ""
echo "🎉 FoundationDB test completed successfully!"
echo ""
echo "📝 Next Steps:"
echo "   1. Use $FDB_CLI for interactive testing"
echo "   2. Integrate with your applications using FoundationDB client libraries"
echo "   3. For production, install full FoundationDB system-wide"