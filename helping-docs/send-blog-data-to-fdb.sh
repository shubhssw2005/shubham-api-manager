#!/bin/bash

# Send Blog Data to FoundationDB Test Script
# This script demonstrates storing blog data in FoundationDB

set -e

echo "📝 Sending Blog Data to FoundationDB"
echo "===================================="

FDB_CLI="./foundationdb/fdbcli"

# Check if FoundationDB is available
if [[ ! -f "$FDB_CLI" ]]; then
    echo "❌ FoundationDB CLI not found. Run ./start-foundationdb-local.sh first"
    exit 1
fi

echo "✅ FoundationDB CLI found"

# Enable write mode
echo ""
echo "🔧 Enabling write mode..."
$FDB_CLI --exec "writemode on"

# Blog Post 1: Tech Article
echo ""
echo "📄 Storing Blog Post 1: Tech Article"
echo "------------------------------------"

$FDB_CLI --exec "set blog:1:title 'Ultra-Low Latency Systems with FoundationDB'"
$FDB_CLI --exec "set blog:1:description 'A comprehensive guide to building high-performance distributed systems using FoundationDB for sub-millisecond response times.'"
$FDB_CLI --exec "set blog:1:content 'FoundationDB is a distributed database designed to handle demanding workloads with ACID transactions...'"
$FDB_CLI --exec "set blog:1:author 'John Doe'"
$FDB_CLI --exec "set blog:1:category 'Technology'"
$FDB_CLI --exec "set blog:1:tags 'foundationdb,performance,distributed-systems,low-latency'"
$FDB_CLI --exec "set blog:1:created_at '2025-08-29T13:20:00Z'"
$FDB_CLI --exec "set blog:1:image_url 'https://example.com/images/foundationdb-architecture.jpg'"
$FDB_CLI --exec "set blog:1:pdf_url 'https://example.com/pdfs/foundationdb-guide.pdf'"
$FDB_CLI --exec "set blog:1:status 'published'"

# Blog Post 2: Tutorial
echo ""
echo "📄 Storing Blog Post 2: Tutorial"
echo "--------------------------------"

$FDB_CLI --exec "set blog:2:title 'Building Real-time Analytics with C++'"
$FDB_CLI --exec "set blog:2:description 'Learn how to implement real-time data processing pipelines using modern C++ techniques and lock-free data structures.'"
$FDB_CLI --exec "set blog:2:content 'Real-time analytics requires careful consideration of memory management, CPU cache efficiency...'"
$FDB_CLI --exec "set blog:2:author 'Jane Smith'"
$FDB_CLI --exec "set blog:2:category 'Programming'"
$FDB_CLI --exec "set blog:2:tags 'cpp,analytics,real-time,performance'"
$FDB_CLI --exec "set blog:2:created_at '2025-08-28T10:15:00Z'"
$FDB_CLI --exec "set blog:2:image_url 'https://example.com/images/cpp-analytics-pipeline.png'"
$FDB_CLI --exec "set blog:2:pdf_url 'https://example.com/pdfs/cpp-analytics-tutorial.pdf'"
$FDB_CLI --exec "set blog:2:status 'published'"

# Blog Post 3: Case Study
echo ""
echo "📄 Storing Blog Post 3: Case Study"
echo "----------------------------------"

$FDB_CLI --exec "set blog:3:title 'Scaling to 1M Transactions per Second'"
$FDB_CLI --exec "set blog:3:description 'A detailed case study of how we achieved 1 million transactions per second using FoundationDB and optimized C++ microservices.'"
$FDB_CLI --exec "set blog:3:content 'Our journey to handle massive scale began with identifying bottlenecks in our existing architecture...'"
$FDB_CLI --exec "set blog:3:author 'Mike Johnson'"
$FDB_CLI --exec "set blog:3:category 'Case Study'"
$FDB_CLI --exec "set blog:3:tags 'scaling,performance,microservices,foundationdb'"
$FDB_CLI --exec "set blog:3:created_at '2025-08-27T14:30:00Z'"
$FDB_CLI --exec "set blog:3:image_url 'https://example.com/images/scaling-architecture.svg'"
$FDB_CLI --exec "set blog:3:pdf_url 'https://example.com/pdfs/scaling-case-study.pdf'"
$FDB_CLI --exec "set blog:3:status 'draft'"

# Store metadata
echo ""
echo "📊 Storing Blog Metadata"
echo "-----------------------"

$FDB_CLI --exec "set blog:count '3'"
$FDB_CLI --exec "set blog:last_updated '2025-08-29T13:20:00Z'"
$FDB_CLI --exec "set blog:categories 'Technology,Programming,Case Study'"

# Store media information
echo ""
echo "🖼️ Storing Media Information"
echo "---------------------------"

$FDB_CLI --exec "set media:image:1 '{\"id\":1,\"type\":\"image\",\"url\":\"https://example.com/images/foundationdb-architecture.jpg\",\"size\":\"1024x768\",\"format\":\"jpg\"}'"
$FDB_CLI --exec "set media:image:2 '{\"id\":2,\"type\":\"image\",\"url\":\"https://example.com/images/cpp-analytics-pipeline.png\",\"size\":\"1920x1080\",\"format\":\"png\"}'"
$FDB_CLI --exec "set media:image:3 '{\"id\":3,\"type\":\"image\",\"url\":\"https://example.com/images/scaling-architecture.svg\",\"size\":\"vector\",\"format\":\"svg\"}'"

$FDB_CLI --exec "set media:pdf:1 '{\"id\":1,\"type\":\"pdf\",\"url\":\"https://example.com/pdfs/foundationdb-guide.pdf\",\"size\":\"2.5MB\",\"pages\":45}'"
$FDB_CLI --exec "set media:pdf:2 '{\"id\":2,\"type\":\"pdf\",\"url\":\"https://example.com/pdfs/cpp-analytics-tutorial.pdf\",\"size\":\"1.8MB\",\"pages\":32}'"
$FDB_CLI --exec "set media:pdf:3 '{\"id\":3,\"type\":\"pdf\",\"url\":\"https://example.com/pdfs/scaling-case-study.pdf\",\"size\":\"3.2MB\",\"pages\":58}'"

echo ""
echo "✅ All blog data stored successfully!"

# Verify data storage
echo ""
echo "🔍 Verifying Stored Data"
echo "========================"

echo ""
echo "📊 Database Status:"
$FDB_CLI --exec "status"

echo ""
echo "📝 Blog Posts:"
echo "-------------"
for i in {1..3}; do
    echo "Blog Post $i:"
    echo "  Title: $($FDB_CLI --exec "get blog:$i:title" | grep -o "'.*'" | tr -d "'")"
    echo "  Author: $($FDB_CLI --exec "get blog:$i:author" | grep -o "'.*'" | tr -d "'")"
    echo "  Category: $($FDB_CLI --exec "get blog:$i:category" | grep -o "'.*'" | tr -d "'")"
    echo "  Status: $($FDB_CLI --exec "get blog:$i:status" | grep -o "'.*'" | tr -d "'")"
    echo ""
done

echo "🖼️ Media Files:"
echo "-------------"
echo "Images: $($FDB_CLI --exec "getrange media:image: media:image:z" | wc -l | tr -d ' ') files"
echo "PDFs: $($FDB_CLI --exec "getrange media:pdf: media:pdf:z" | wc -l | tr -d ' ') files"

echo ""
echo "📈 Blog Statistics:"
echo "-----------------"
echo "Total Posts: $($FDB_CLI --exec "get blog:count" | grep -o "'.*'" | tr -d "'")"
echo "Last Updated: $($FDB_CLI --exec "get blog:last_updated" | grep -o "'.*'" | tr -d "'")"
echo "Categories: $($FDB_CLI --exec "get blog:categories" | grep -o "'.*'" | tr -d "'")"

echo ""
echo "🎉 Blog data testing completed successfully!"
echo ""
echo "💡 Try these commands to explore the data:"
echo "   $FDB_CLI --exec \"getrange blog: blog:z\"           # List all blog keys"
echo "   $FDB_CLI --exec \"get blog:1:title\"                # Get specific blog title"
echo "   $FDB_CLI --exec \"getrange media: media:z\"         # List all media files"