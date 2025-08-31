#!/bin/bash

# Blog Data Query Script
# Comprehensive script for querying blog data from various sources

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/test-data"
FDB_CLI="$SCRIPT_DIR/foundationdb/fdbcli"

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

print_query() {
    echo -e "${BLUE}[QUERY]${NC} $1"
}

# Show usage information
show_usage() {
    echo "Blog Data Query Tool"
    echo "==================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  # READ Operations"
    echo "  list                    - List all available blog posts"
    echo "  search <term>          - Search blog posts by term"
    echo "  get <id>               - Get specific blog post by ID"
    echo "  stats                  - Show blog statistics"
    echo "  recent [count]         - Show recent posts (default: 5)"
    echo "  tags                   - List all tags"
    echo "  by-tag <tag>           - Get posts by tag"
    echo ""
    echo "  # CREATE Operations"
    echo "  create                 - Create a new blog post (interactive)"
    echo "  create-from-file <file> - Create post from JSON file"
    echo ""
    echo "  # UPDATE Operations"
    echo "  update <id>            - Update existing blog post (interactive)"
    echo "  update-field <id> <field> <value> - Update specific field"
    echo "  publish <id>           - Publish a draft post"
    echo "  unpublish <id>         - Convert published post to draft"
    echo ""
    echo "  # DELETE Operations"
    echo "  delete <id>            - Delete a blog post"
    echo "  delete-by-tag <tag>    - Delete all posts with specific tag"
    echo ""
    echo "  # UTILITY Operations"
    echo "  export [format]        - Export data (json|csv|xml)"
    echo "  benchmark              - Run query performance benchmark"
    echo "  fdb-test               - Test FoundationDB integration"
    echo ""
    echo "Options:"
    echo "  --format <json|table>  - Output format (default: table)"
    echo "  --limit <n>            - Limit results (default: 10)"
    echo "  --verbose              - Verbose output"
    echo ""
}

# Check if required files exist
check_dependencies() {
    if [[ ! -d "$DATA_DIR" ]]; then
        print_error "Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    if [[ ! -f "$DATA_DIR/sample-blog-post.json" ]]; then
        print_warning "Sample blog post not found, creating one..."
        create_sample_data
    fi
}

# Create sample blog data if it doesn't exist
create_sample_data() {
    mkdir -p "$DATA_DIR"
    
    cat > "$DATA_DIR/sample-blog-post.json" << 'EOF'
{
  "id": "1",
  "title": "Getting Started with Ultra-Low Latency Systems",
  "content": "Building high-performance systems requires careful attention to latency optimization...",
  "author": "Tech Team",
  "tags": ["performance", "cpp", "optimization"],
  "created_at": "2025-08-29T10:00:00Z",
  "updated_at": "2025-08-29T10:00:00Z",
  "status": "published",
  "views": 1250,
  "likes": 45
}
EOF

    cat > "$DATA_DIR/blog-posts.json" << 'EOF'
[
  {
    "id": "1",
    "title": "Getting Started with Ultra-Low Latency Systems",
    "content": "Building high-performance systems requires careful attention to latency optimization, memory management, and CPU cache efficiency.",
    "author": "Tech Team",
    "tags": ["performance", "cpp", "optimization"],
    "created_at": "2025-08-29T10:00:00Z",
    "status": "published",
    "views": 1250,
    "likes": 45
  },
  {
    "id": "2", 
    "title": "FoundationDB Integration Patterns",
    "content": "Learn how to integrate FoundationDB with high-performance applications for ACID transactions and horizontal scaling.",
    "author": "Database Team",
    "tags": ["database", "foundationdb", "scaling"],
    "created_at": "2025-08-28T15:30:00Z",
    "status": "published",
    "views": 890,
    "likes": 32
  },
  {
    "id": "3",
    "title": "Memory Management in C++ Systems",
    "content": "Advanced techniques for memory allocation, NUMA awareness, and lock-free data structures in modern C++ applications.",
    "author": "Performance Team",
    "tags": ["cpp", "memory", "performance"],
    "created_at": "2025-08-27T09:15:00Z",
    "status": "published",
    "views": 2100,
    "likes": 78
  },
  {
    "id": "4",
    "title": "Kubernetes Deployment Strategies",
    "content": "Best practices for deploying high-performance applications on Kubernetes with proper resource management.",
    "author": "DevOps Team",
    "tags": ["kubernetes", "deployment", "devops"],
    "created_at": "2025-08-26T14:20:00Z",
    "status": "draft",
    "views": 0,
    "likes": 0
  }
]
EOF

    print_status "Sample blog data created"
}

# List all blog posts
list_posts() {
    local format=${1:-"table"}
    local limit=${2:-10}
    
    print_query "Listing blog posts (limit: $limit, format: $format)"
    
    if [[ "$format" == "json" ]]; then
        jq --argjson limit "$limit" '.[:$limit]' "$DATA_DIR/blog-posts.json"
    else
        echo "ID | Title | Author | Status | Views | Likes"
        echo "---|-------|--------|--------|-------|------"
        jq -r --argjson limit "$limit" '.[:$limit][] | "\(.id) | \(.title) | \(.author) | \(.status) | \(.views) | \(.likes)"' "$DATA_DIR/blog-posts.json"
    fi
}

# Search blog posts
search_posts() {
    local term="$1"
    local format=${2:-"table"}
    
    if [[ -z "$term" ]]; then
        print_error "Search term required"
        return 1
    fi
    
    print_query "Searching for: '$term'"
    
    if [[ "$format" == "json" ]]; then
        jq --arg term "$term" '[.[] | select(.title | test($term; "i")) or select(.content | test($term; "i")) or select(.tags[] | test($term; "i"))]' "$DATA_DIR/blog-posts.json"
    else
        echo "Search Results for '$term':"
        echo "ID | Title | Author | Tags"
        echo "---|-------|--------|-----"
        jq -r --arg term "$term" '.[] | select(.title | test($term; "i")) or select(.content | test($term; "i")) or select(.tags[] | test($term; "i")) | "\(.id) | \(.title) | \(.author) | \(.tags | join(", "))"' "$DATA_DIR/blog-posts.json"
    fi
}

# Get specific blog post
get_post() {
    local id="$1"
    local format=${2:-"json"}
    
    if [[ -z "$id" ]]; then
        print_error "Post ID required"
        return 1
    fi
    
    print_query "Getting post ID: $id"
    
    local post=$(jq --arg id "$id" '.[] | select(.id == $id)' "$DATA_DIR/blog-posts.json")
    
    if [[ -z "$post" ]]; then
        print_error "Post not found: $id"
        return 1
    fi
    
    if [[ "$format" == "json" ]]; then
        echo "$post" | jq '.'
    else
        echo "$post" | jq -r '"Title: " + .title + "\nAuthor: " + .author + "\nStatus: " + .status + "\nViews: " + (.views | tostring) + "\nLikes: " + (.likes | tostring) + "\nTags: " + (.tags | join(", ")) + "\nContent: " + .content'
    fi
}

# Show blog statistics
show_stats() {
    print_query "Calculating blog statistics"
    
    local total=$(jq 'length' "$DATA_DIR/blog-posts.json")
    local published=$(jq '[.[] | select(.status == "published")] | length' "$DATA_DIR/blog-posts.json")
    local drafts=$(jq '[.[] | select(.status == "draft")] | length' "$DATA_DIR/blog-posts.json")
    local total_views=$(jq '[.[] | .views] | add' "$DATA_DIR/blog-posts.json")
    local total_likes=$(jq '[.[] | .likes] | add' "$DATA_DIR/blog-posts.json")
    local avg_views=$(jq '[.[] | .views] | add / length' "$DATA_DIR/blog-posts.json")
    
    echo "Blog Statistics:"
    echo "==============="
    echo "Total Posts: $total"
    echo "Published: $published"
    echo "Drafts: $drafts"
    echo "Total Views: $total_views"
    echo "Total Likes: $total_likes"
    echo "Average Views: $(printf "%.1f" "$avg_views")"
    
    echo ""
    echo "Top Posts by Views:"
    jq -r 'sort_by(-.views) | .[:3][] | "  \(.title) - \(.views) views"' "$DATA_DIR/blog-posts.json"
}

# Show recent posts
show_recent() {
    local count=${1:-5}
    local format=${2:-"table"}
    
    print_query "Showing $count most recent posts"
    
    if [[ "$format" == "json" ]]; then
        jq --argjson count "$count" 'sort_by(.created_at) | reverse | .[:$count]' "$DATA_DIR/blog-posts.json"
    else
        echo "Recent Posts:"
        echo "ID | Title | Author | Created"
        echo "---|-------|--------|--------"
        jq -r --argjson count "$count" 'sort_by(.created_at) | reverse | .[:$count][] | "\(.id) | \(.title) | \(.author) | \(.created_at)"' "$DATA_DIR/blog-posts.json"
    fi
}

# List all tags
list_tags() {
    print_query "Listing all tags"
    
    echo "Available Tags:"
    jq -r '[.[] | .tags[]] | unique | sort | .[]' "$DATA_DIR/blog-posts.json" | while read -r tag; do
        local count=$(jq --arg tag "$tag" '[.[] | select(.tags[] == $tag)] | length' "$DATA_DIR/blog-posts.json")
        echo "  $tag ($count posts)"
    done
}

# Get posts by tag
get_by_tag() {
    local tag="$1"
    local format=${2:-"table"}
    
    if [[ -z "$tag" ]]; then
        print_error "Tag required"
        return 1
    fi
    
    print_query "Getting posts with tag: '$tag'"
    
    if [[ "$format" == "json" ]]; then
        jq --arg tag "$tag" '[.[] | select(.tags[] == $tag)]' "$DATA_DIR/blog-posts.json"
    else
        echo "Posts tagged '$tag':"
        echo "ID | Title | Author | Views"
        echo "---|-------|--------|------"
        jq -r --arg tag "$tag" '.[] | select(.tags[] == $tag) | "\(.id) | \(.title) | \(.author) | \(.views)"' "$DATA_DIR/blog-posts.json"
    fi
}

# Export data in different formats
export_data() {
    local format=${1:-"json"}
    local output_file="blog-export-$(date +%Y%m%d-%H%M%S).$format"
    
    print_query "Exporting data as $format to $output_file"
    
    case "$format" in
        "json")
            jq '.' "$DATA_DIR/blog-posts.json" > "$output_file"
            ;;
        "csv")
            echo "id,title,author,status,views,likes,tags,created_at" > "$output_file"
            jq -r '.[] | [.id, .title, .author, .status, .views, .likes, (.tags | join(";")), .created_at] | @csv' "$DATA_DIR/blog-posts.json" >> "$output_file"
            ;;
        "xml")
            echo '<?xml version="1.0" encoding="UTF-8"?>' > "$output_file"
            echo '<blog_posts>' >> "$output_file"
            jq -r '.[] | "<post><id>\(.id)</id><title>\(.title)</title><author>\(.author)</author><status>\(.status)</status><views>\(.views)</views><likes>\(.likes)</likes><tags>\(.tags | join(","))</tags><created_at>\(.created_at)</created_at></post>"' "$DATA_DIR/blog-posts.json" >> "$output_file"
            echo '</blog_posts>' >> "$output_file"
            ;;
        *)
            print_error "Unsupported format: $format"
            return 1
            ;;
    esac
    
    print_status "Data exported to $output_file"
}

# Run performance benchmark
run_benchmark() {
    print_query "Running query performance benchmark"
    
    echo "Benchmark Results:"
    echo "=================="
    
    # Test 1: List all posts
    local start_time=$(date +%s%N)
    list_posts "json" 100 > /dev/null
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    echo "List 100 posts: ${duration}ms"
    
    # Test 2: Search operation
    start_time=$(date +%s%N)
    search_posts "performance" "json" > /dev/null
    end_time=$(date +%s%N)
    duration=$(( (end_time - start_time) / 1000000 ))
    echo "Search operation: ${duration}ms"
    
    # Test 3: Statistics calculation
    start_time=$(date +%s%N)
    show_stats > /dev/null
    end_time=$(date +%s%N)
    duration=$(( (end_time - start_time) / 1000000 ))
    echo "Statistics calculation: ${duration}ms"
    
    # Test 4: Tag listing
    start_time=$(date +%s%N)
    list_tags > /dev/null
    end_time=$(date +%s%N)
    duration=$(( (end_time - start_time) / 1000000 ))
    echo "Tag listing: ${duration}ms"
}

# CREATE Operations

# Create a new blog post interactively
create_post() {
    print_query "Creating new blog post"
    
    # Get next available ID
    local next_id=$(jq '[.[] | .id | tonumber] | max + 1' "$DATA_DIR/blog-posts.json")
    
    echo "Enter blog post details:"
    read -p "Title: " title
    read -p "Author: " author
    read -p "Content: " content
    read -p "Tags (comma-separated): " tags_input
    read -p "Status (published/draft) [draft]: " status
    
    # Set defaults
    status=${status:-"draft"}
    
    # Parse tags
    IFS=',' read -ra tag_array <<< "$tags_input"
    local tags_json="["
    for i in "${!tag_array[@]}"; do
        tag_array[i]=$(echo "${tag_array[i]}" | xargs) # trim whitespace
        if [[ $i -gt 0 ]]; then
            tags_json+=","
        fi
        tags_json+="\"${tag_array[i]}\""
    done
    tags_json+="]"
    
    # Create new post object
    local new_post=$(cat << EOF
{
  "id": "$next_id",
  "title": "$title",
  "content": "$content",
  "author": "$author",
  "tags": $tags_json,
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "$status",
  "views": 0,
  "likes": 0
}
EOF
)
    
    # Add to blog posts array
    local updated_posts=$(jq --argjson new_post "$new_post" '. + [$new_post]' "$DATA_DIR/blog-posts.json")
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Blog post created with ID: $next_id"
    echo "$new_post" | jq '.'
}

# Create post from JSON file
create_from_file() {
    local file="$1"
    
    if [[ -z "$file" ]]; then
        print_error "JSON file path required"
        return 1
    fi
    
    if [[ ! -f "$file" ]]; then
        print_error "File not found: $file"
        return 1
    fi
    
    print_query "Creating post from file: $file"
    
    # Validate JSON
    if ! jq empty "$file" 2>/dev/null; then
        print_error "Invalid JSON in file: $file"
        return 1
    fi
    
    # Get next ID and add it to the post
    local next_id=$(jq '[.[] | .id | tonumber] | max + 1' "$DATA_DIR/blog-posts.json")
    local new_post=$(jq --arg id "$next_id" --arg created "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '. + {id: $id, created_at: $created, updated_at: $created}' "$file")
    
    # Add to blog posts array
    local updated_posts=$(jq --argjson new_post "$new_post" '. + [$new_post]' "$DATA_DIR/blog-posts.json")
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Blog post created from file with ID: $next_id"
}

# UPDATE Operations

# Update existing blog post interactively
update_post() {
    local id="$1"
    
    if [[ -z "$id" ]]; then
        print_error "Post ID required"
        return 1
    fi
    
    # Check if post exists
    local existing_post=$(jq --arg id "$id" '.[] | select(.id == $id)' "$DATA_DIR/blog-posts.json")
    if [[ -z "$existing_post" ]]; then
        print_error "Post not found: $id"
        return 1
    fi
    
    print_query "Updating post ID: $id"
    
    # Show current values
    echo "Current post details:"
    echo "$existing_post" | jq -r '"Title: " + .title + "\nAuthor: " + .author + "\nStatus: " + .status + "\nTags: " + (.tags | join(", "))'
    echo ""
    
    # Get updates
    echo "Enter new values (press Enter to keep current):"
    read -p "Title: " new_title
    read -p "Author: " new_author
    read -p "Content: " new_content
    read -p "Tags (comma-separated): " new_tags_input
    read -p "Status (published/draft): " new_status
    
    # Build update object
    local updates="{}"
    
    if [[ -n "$new_title" ]]; then
        updates=$(echo "$updates" | jq --arg title "$new_title" '. + {title: $title}')
    fi
    
    if [[ -n "$new_author" ]]; then
        updates=$(echo "$updates" | jq --arg author "$new_author" '. + {author: $author}')
    fi
    
    if [[ -n "$new_content" ]]; then
        updates=$(echo "$updates" | jq --arg content "$new_content" '. + {content: $content}')
    fi
    
    if [[ -n "$new_status" ]]; then
        updates=$(echo "$updates" | jq --arg status "$new_status" '. + {status: $status}')
    fi
    
    if [[ -n "$new_tags_input" ]]; then
        IFS=',' read -ra tag_array <<< "$new_tags_input"
        local tags_json="["
        for i in "${!tag_array[@]}"; do
            tag_array[i]=$(echo "${tag_array[i]}" | xargs)
            if [[ $i -gt 0 ]]; then
                tags_json+=","
            fi
            tags_json+="\"${tag_array[i]}\""
        done
        tags_json+="]"
        updates=$(echo "$updates" | jq --argjson tags "$tags_json" '. + {tags: $tags}')
    fi
    
    # Add updated_at timestamp
    updates=$(echo "$updates" | jq --arg updated "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '. + {updated_at: $updated}')
    
    # Apply updates
    local updated_posts=$(jq --arg id "$id" --argjson updates "$updates" 'map(if .id == $id then . + $updates else . end)' "$DATA_DIR/blog-posts.json")
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Post updated successfully"
    
    # Show updated post
    jq --arg id "$id" '.[] | select(.id == $id)' "$DATA_DIR/blog-posts.json"
}

# Update specific field
update_field() {
    local id="$1"
    local field="$2"
    local value="$3"
    
    if [[ -z "$id" || -z "$field" || -z "$value" ]]; then
        print_error "Usage: update-field <id> <field> <value>"
        return 1
    fi
    
    # Check if post exists
    local existing_post=$(jq --arg id "$id" '.[] | select(.id == $id)' "$DATA_DIR/blog-posts.json")
    if [[ -z "$existing_post" ]]; then
        print_error "Post not found: $id"
        return 1
    fi
    
    print_query "Updating field '$field' for post ID: $id"
    
    # Handle different field types
    local update_value
    case "$field" in
        "views"|"likes")
            update_value="$value"
            ;;
        "tags")
            # Parse comma-separated tags
            IFS=',' read -ra tag_array <<< "$value"
            local tags_json="["
            for i in "${!tag_array[@]}"; do
                tag_array[i]=$(echo "${tag_array[i]}" | xargs)
                if [[ $i -gt 0 ]]; then
                    tags_json+=","
                fi
                tags_json+="\"${tag_array[i]}\""
            done
            tags_json+="]"
            update_value="$tags_json"
            ;;
        *)
            update_value="\"$value\""
            ;;
    esac
    
    # Apply update
    local updated_posts
    if [[ "$field" == "tags" ]]; then
        updated_posts=$(jq --arg id "$id" --argjson value "$update_value" --arg updated "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 'map(if .id == $id then .[$field] = $value | .updated_at = $updated else . end)' "$DATA_DIR/blog-posts.json")
    else
        updated_posts=$(jq --arg id "$id" --arg field "$field" --argjson value "$update_value" --arg updated "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 'map(if .id == $id then .[$field] = $value | .updated_at = $updated else . end)' "$DATA_DIR/blog-posts.json")
    fi
    
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Field '$field' updated to: $value"
}

# Publish a draft post
publish_post() {
    local id="$1"
    
    if [[ -z "$id" ]]; then
        print_error "Post ID required"
        return 1
    fi
    
    update_field "$id" "status" "published"
    print_status "Post ID $id published"
}

# Unpublish a post (convert to draft)
unpublish_post() {
    local id="$1"
    
    if [[ -z "$id" ]]; then
        print_error "Post ID required"
        return 1
    fi
    
    update_field "$id" "status" "draft"
    print_status "Post ID $id converted to draft"
}

# DELETE Operations

# Delete a blog post
delete_post() {
    local id="$1"
    
    if [[ -z "$id" ]]; then
        print_error "Post ID required"
        return 1
    fi
    
    # Check if post exists
    local existing_post=$(jq --arg id "$id" '.[] | select(.id == $id)' "$DATA_DIR/blog-posts.json")
    if [[ -z "$existing_post" ]]; then
        print_error "Post not found: $id"
        return 1
    fi
    
    print_query "Deleting post ID: $id"
    
    # Show post to be deleted
    echo "Post to be deleted:"
    echo "$existing_post" | jq -r '"Title: " + .title + "\nAuthor: " + .author'
    
    # Confirm deletion
    read -p "Are you sure you want to delete this post? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "Deletion cancelled"
        return 0
    fi
    
    # Remove post from array
    local updated_posts=$(jq --arg id "$id" 'map(select(.id != $id))' "$DATA_DIR/blog-posts.json")
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Post deleted successfully"
}

# Delete posts by tag
delete_by_tag() {
    local tag="$1"
    
    if [[ -z "$tag" ]]; then
        print_error "Tag required"
        return 1
    fi
    
    # Find posts with the tag
    local posts_to_delete=$(jq --arg tag "$tag" '[.[] | select(.tags[] == $tag)]' "$DATA_DIR/blog-posts.json")
    local count=$(echo "$posts_to_delete" | jq 'length')
    
    if [[ "$count" == "0" ]]; then
        print_warning "No posts found with tag: $tag"
        return 0
    fi
    
    print_query "Found $count posts with tag '$tag'"
    
    # Show posts to be deleted
    echo "Posts to be deleted:"
    echo "$posts_to_delete" | jq -r '.[] | "  ID: " + .id + " - " + .title'
    
    # Confirm deletion
    read -p "Are you sure you want to delete all $count posts? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "Deletion cancelled"
        return 0
    fi
    
    # Remove posts with the tag
    local updated_posts=$(jq --arg tag "$tag" 'map(select(.tags[] != $tag or (.tags | length) > 1))' "$DATA_DIR/blog-posts.json")
    echo "$updated_posts" > "$DATA_DIR/blog-posts.json"
    
    print_status "Deleted $count posts with tag '$tag'"
}

# Test FoundationDB integration
test_fdb_integration() {
    print_query "Testing FoundationDB integration"
    
    if [[ ! -f "$FDB_CLI" ]]; then
        print_warning "FoundationDB CLI not found. Run ./start-foundationdb-local.sh first"
        return 1
    fi
    
    echo "FoundationDB Integration Test:"
    echo "============================="
    
    # Test 1: Check FDB status
    echo "1. Database Status:"
    $FDB_CLI --exec "status"
    
    echo ""
    echo "2. Simulating blog data storage in FoundationDB:"
    
    # Simulate storing blog posts in FDB
    jq -r '.[] | "set blog:\(.id) \(.title)"' "$DATA_DIR/blog-posts.json" | head -3 | while read -r cmd; do
        echo "   $cmd"
    done
    
    echo ""
    echo "3. Simulating blog data retrieval:"
    echo "   get blog:1"
    echo "   get blog:2"
    echo "   getrange blog: blog;"
    
    echo ""
    print_status "FoundationDB integration test completed"
    print_status "Use '$FDB_CLI' for interactive testing"
}

# Parse command line arguments
parse_args() {
    local format="table"
    local limit=10
    local verbose=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --format)
                format="$2"
                shift 2
                ;;
            --limit)
                limit="$2"
                shift 2
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Export variables for use in functions
    export OUTPUT_FORMAT="$format"
    export RESULT_LIMIT="$limit"
    export VERBOSE_MODE="$verbose"
}

# Main execution
main() {
    local command="$1"
    shift || true
    
    # Parse global options first
    parse_args "$@"
    
    # Remove parsed options from arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --format|--limit)
                shift 2
                ;;
            --verbose)
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    check_dependencies
    
    case "$command" in
        # READ Operations
        "list")
            list_posts "$OUTPUT_FORMAT" "$RESULT_LIMIT"
            ;;
        "search")
            search_posts "$1" "$OUTPUT_FORMAT"
            ;;
        "get")
            get_post "$1" "$OUTPUT_FORMAT"
            ;;
        "stats")
            show_stats
            ;;
        "recent")
            show_recent "${1:-5}" "$OUTPUT_FORMAT"
            ;;
        "tags")
            list_tags
            ;;
        "by-tag")
            get_by_tag "$1" "$OUTPUT_FORMAT"
            ;;
        # CREATE Operations
        "create")
            create_post
            ;;
        "create-from-file")
            create_from_file "$1"
            ;;
        # UPDATE Operations
        "update")
            update_post "$1"
            ;;
        "update-field")
            update_field "$1" "$2" "$3"
            ;;
        "publish")
            publish_post "$1"
            ;;
        "unpublish")
            unpublish_post "$1"
            ;;
        # DELETE Operations
        "delete")
            delete_post "$1"
            ;;
        "delete-by-tag")
            delete_by_tag "$1"
            ;;
        # UTILITY Operations
        "export")
            export_data "${1:-json}"
            ;;
        "benchmark")
            run_benchmark
            ;;
        "fdb-test")
            test_fdb_integration
            ;;
        "help"|"--help"|"-h"|"")
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"