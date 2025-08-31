#!/bin/bash

# Global API v2 Test Script
# Tests ScyllaDB + FoundationDB CRUD operations

set -e

# Configuration
API_BASE="http://localhost:3005/api/v2/universal"
TEST_DATA_DIR="./test-data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Test variables
CREATED_POST_ID=""
CREATED_PRODUCT_ID=""
CREATED_CUSTOMER_ID=""

# Test 1: Health Check
test_health_check() {
    print_test "Testing health check endpoint"
    
    local response=$(curl -s -w "%{http_code}" "$API_BASE/health")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Health check passed"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Health check failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 2: Create Post
test_create_post() {
    print_test "Testing POST creation"
    
    local post_data='{
        "title": "Test Blog Post via API v2",
        "content": "This is a test blog post created via the new ScyllaDB + FoundationDB API",
        "excerpt": "Test excerpt for the blog post",
        "status": "published",
        "tags": ["test", "api", "scylladb"],
        "author_id": "550e8400-e29b-41d4-a716-446655440000",
        "featured": true
    }'
    
    local response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$post_data" \
        "$API_BASE/posts")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "201" ]]; then
        print_status "✅ Post created successfully"
        CREATED_POST_ID=$(echo "$body" | jq -r '.data.id')
        echo "Created Post ID: $CREATED_POST_ID"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Post creation failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 3: Get Post by ID
test_get_post() {
    if [[ -z "$CREATED_POST_ID" ]]; then
        print_warning "Skipping get post test - no post ID available"
        return 0
    fi
    
    print_test "Testing GET post by ID"
    
    local response=$(curl -s -w "%{http_code}" "$API_BASE/posts/$CREATED_POST_ID")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Post retrieved successfully"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Post retrieval failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 4: Update Post
test_update_post() {
    if [[ -z "$CREATED_POST_ID" ]]; then
        print_warning "Skipping update post test - no post ID available"
        return 0
    fi
    
    print_test "Testing PUT post update"
    
    local update_data='{
        "title": "Updated Test Blog Post via API v2",
        "content": "This post has been updated via the ScyllaDB + FoundationDB API",
        "tags": ["test", "api", "scylladb", "updated"]
    }'
    
    local response=$(curl -s -w "%{http_code}" -X PUT \
        -H "Content-Type: application/json" \
        -d "$update_data" \
        "$API_BASE/posts/$CREATED_POST_ID")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Post updated successfully"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Post update failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 5: List Posts
test_list_posts() {
    print_test "Testing GET posts list"
    
    local response=$(curl -s -w "%{http_code}" "$API_BASE/posts?limit=10")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Posts listed successfully"
        local count=$(echo "$body" | jq '.data | length' 2>/dev/null || echo "0")
        echo "Found $count posts"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Posts listing failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 6: Search Posts
test_search_posts() {
    print_test "Testing GET posts search"
    
    local response=$(curl -s -w "%{http_code}" "$API_BASE/posts/search?q=test")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Posts search completed"
        local count=$(echo "$body" | jq '.data | length' 2>/dev/null || echo "0")
        echo "Found $count posts matching 'test'"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Posts search failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 7: Create Product
test_create_product() {
    print_test "Testing Product creation"
    
    local product_data='{
        "name": "Test Product via API v2",
        "price": 99.99,
        "description": "This is a test product created via the new API",
        "short_description": "Test product",
        "sku": "TEST-PROD-001",
        "status": "active",
        "visibility": "public",
        "tags": ["test", "product", "api"],
        "category_id": "550e8400-e29b-41d4-a716-446655440001",
        "quantity": 100,
        "featured": false
    }'
    
    local response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$product_data" \
        "$API_BASE/products")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "201" ]]; then
        print_status "✅ Product created successfully"
        CREATED_PRODUCT_ID=$(echo "$body" | jq -r '.data.id')
        echo "Created Product ID: $CREATED_PRODUCT_ID"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Product creation failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 8: Create Customer
test_create_customer() {
    print_test "Testing Customer creation"
    
    local customer_data='{
        "email": "test@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "phone": "+1234567890",
        "is_active": true,
        "email_verified": false,
        "accepts_marketing": true
    }'
    
    local response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$customer_data" \
        "$API_BASE/customers")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "201" ]]; then
        print_status "✅ Customer created successfully"
        CREATED_CUSTOMER_ID=$(echo "$body" | jq -r '.data.id')
        echo "Created Customer ID: $CREATED_CUSTOMER_ID"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Customer creation failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 9: Get Statistics
test_get_stats() {
    print_test "Testing statistics endpoints"
    
    for table in posts products customers; do
        echo "Getting stats for $table..."
        local response=$(curl -s -w "%{http_code}" "$API_BASE/$table/stats")
        local http_code="${response: -3}"
        local body="${response%???}"
        
        if [[ "$http_code" == "200" ]]; then
            print_status "✅ Stats for $table retrieved"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_warning "⚠️  Stats for $table failed (HTTP $http_code)"
            echo "$body"
        fi
        echo ""
    done
}

# Test 10: Soft Delete
test_soft_delete() {
    if [[ -z "$CREATED_POST_ID" ]]; then
        print_warning "Skipping soft delete test - no post ID available"
        return 0
    fi
    
    print_test "Testing soft delete"
    
    local delete_data='{"reason": "Test deletion via API v2"}'
    
    local response=$(curl -s -w "%{http_code}" -X DELETE \
        -H "Content-Type: application/json" \
        -d "$delete_data" \
        "$API_BASE/posts/$CREATED_POST_ID")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Post soft deleted successfully"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Soft delete failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 11: Restore Deleted Record
test_restore() {
    if [[ -z "$CREATED_POST_ID" ]]; then
        print_warning "Skipping restore test - no post ID available"
        return 0
    fi
    
    print_test "Testing restore deleted record"
    
    local response=$(curl -s -w "%{http_code}" -X POST \
        "$API_BASE/posts/$CREATED_POST_ID/restore")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        print_status "✅ Post restored successfully"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "❌ Restore failed (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Test 12: Bulk Operations
test_bulk_operations() {
    print_test "Testing bulk delete operations"
    
    # Create multiple test records first
    local ids=()
    
    for i in {1..3}; do
        local post_data="{
            \"title\": \"Bulk Test Post $i\",
            \"content\": \"Content for bulk test post $i\",
            \"author_id\": \"550e8400-e29b-41d4-a716-446655440000\"
        }"
        
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$post_data" \
            "$API_BASE/posts")
        
        local id=$(echo "$response" | jq -r '.data.id' 2>/dev/null)
        if [[ "$id" != "null" && -n "$id" ]]; then
            ids+=("$id")
        fi
    done
    
    if [[ ${#ids[@]} -gt 0 ]]; then
        # Perform bulk delete
        local bulk_data=$(printf '{"ids": ["%s"], "reason": "Bulk test cleanup"}' "$(IFS='","'; echo "${ids[*]}")")
        
        local response=$(curl -s -w "%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$bulk_data" \
            "$API_BASE/posts/bulk-delete")
        
        local http_code="${response: -3}"
        local body="${response%???}"
        
        if [[ "$http_code" == "200" ]]; then
            print_status "✅ Bulk delete completed"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_error "❌ Bulk delete failed (HTTP $http_code)"
            echo "$body"
        fi
    else
        print_warning "⚠️  No records created for bulk delete test"
    fi
}

# Performance Test
test_performance() {
    print_test "Testing API performance"
    
    local start_time=$(date +%s%N)
    
    # Perform multiple operations
    for i in {1..10}; do
        curl -s "$API_BASE/posts?limit=5" > /dev/null
    done
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    print_status "✅ Performance test completed"
    echo "10 list operations took: ${duration}ms"
    echo "Average per operation: $((duration / 10))ms"
}

# Cleanup function
cleanup_test_data() {
    print_test "Cleaning up test data"
    
    # Hard delete test records
    for id in "$CREATED_POST_ID" "$CREATED_PRODUCT_ID" "$CREATED_CUSTOMER_ID"; do
        if [[ -n "$id" && "$id" != "null" ]]; then
            curl -s -X DELETE "$API_BASE/posts/$id/hard" > /dev/null 2>&1 || true
            curl -s -X DELETE "$API_BASE/products/$id/hard" > /dev/null 2>&1 || true
            curl -s -X DELETE "$API_BASE/customers/$id/hard" > /dev/null 2>&1 || true
        fi
    done
    
    print_status "✅ Cleanup completed"
}

# Main test execution
main() {
    echo "🧪 Global API v2 Test Suite"
    echo "============================"
    echo "Testing ScyllaDB + FoundationDB CRUD operations"
    echo ""
    
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found - JSON output will be raw"
    fi
    
    # Run tests
    local failed_tests=0
    
    test_health_check || ((failed_tests++))
    echo ""
    
    test_create_post || ((failed_tests++))
    echo ""
    
    test_get_post || ((failed_tests++))
    echo ""
    
    test_update_post || ((failed_tests++))
    echo ""
    
    test_list_posts || ((failed_tests++))
    echo ""
    
    test_search_posts || ((failed_tests++))
    echo ""
    
    test_create_product || ((failed_tests++))
    echo ""
    
    test_create_customer || ((failed_tests++))
    echo ""
    
    test_get_stats || ((failed_tests++))
    echo ""
    
    test_soft_delete || ((failed_tests++))
    echo ""
    
    test_restore || ((failed_tests++))
    echo ""
    
    test_bulk_operations || ((failed_tests++))
    echo ""
    
    test_performance || ((failed_tests++))
    echo ""
    
    # Cleanup
    cleanup_test_data
    echo ""
    
    # Summary
    echo "📊 Test Summary:"
    echo "================"
    if [[ $failed_tests -eq 0 ]]; then
        print_status "🎉 All tests passed!"
    else
        print_error "❌ $failed_tests test(s) failed"
        exit 1
    fi
}

# Run main function
main "$@"