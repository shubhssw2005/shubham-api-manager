#!/bin/bash

# Ultra C++ System Rollback Script
# This script provides emergency rollback capabilities for the ultra-low-latency C++ system

set -euo pipefail

# Configuration
NAMESPACE="ultra-cpp"
APP_NAME="ultra-cpp-gateway"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current deployment color
get_current_color() {
    local current_service=$(kubectl get service "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "")
    
    if [[ "$current_service" == "blue" ]]; then
        echo "blue"
    elif [[ "$current_service" == "green" ]]; then
        echo "green"
    else
        echo "none"
    fi
}

# Get available deployments
get_available_deployments() {
    kubectl get deployments -n "$NAMESPACE" -l app="$APP_NAME" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo ""
}

# Check if deployment is healthy
check_deployment_health() {
    local deployment_name=$1
    
    # Check if deployment exists and is ready
    local ready_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local desired_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [[ "$ready_replicas" == "$desired_replicas" && "$ready_replicas" -gt 0 ]]; then
        return 0
    else
        return 1
    fi
}

# Emergency rollback
emergency_rollback() {
    log_warning "Performing emergency rollback..."
    
    local current_color=$(get_current_color)
    local available_deployments=$(get_available_deployments)
    
    log_info "Current active color: $current_color"
    log_info "Available deployments: $available_deployments"
    
    # Find the other deployment
    local target_deployment=""
    for deployment in $available_deployments; do
        if [[ "$deployment" == "$APP_NAME-blue" && "$current_color" != "blue" ]]; then
            target_deployment="$deployment"
            break
        elif [[ "$deployment" == "$APP_NAME-green" && "$current_color" != "green" ]]; then
            target_deployment="$deployment"
            break
        fi
    done
    
    if [[ -z "$target_deployment" ]]; then
        log_error "No alternative deployment found for rollback"
        exit 1
    fi
    
    local target_color="${target_deployment#*-}"
    
    # Check if target deployment is healthy
    if ! check_deployment_health "$target_deployment"; then
        log_error "Target deployment '$target_deployment' is not healthy"
        exit 1
    fi
    
    log_info "Rolling back to deployment: $target_deployment"
    
    # Switch traffic
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$target_color'"}}}'
    
    log_success "Emergency rollback completed - traffic switched to $target_color"
}

# List deployment history
list_deployment_history() {
    log_info "Deployment history for $APP_NAME:"
    
    local deployments=$(get_available_deployments)
    
    if [[ -z "$deployments" ]]; then
        log_warning "No deployments found"
        return
    fi
    
    for deployment in $deployments; do
        local color="${deployment#*-}"
        local image=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")
        local ready_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
        local creation_time=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "unknown")
        
        local status="UNHEALTHY"
        if [[ "$ready_replicas" == "$desired_replicas" && "$ready_replicas" -gt 0 ]]; then
            status="HEALTHY"
        fi
        
        local active=""
        local current_color=$(get_current_color)
        if [[ "$color" == "$current_color" ]]; then
            active=" (ACTIVE)"
        fi
        
        echo "  $deployment$active:"
        echo "    Color: $color"
        echo "    Image: $image"
        echo "    Status: $status ($ready_replicas/$desired_replicas replicas)"
        echo "    Created: $creation_time"
        echo ""
    done
}

# Rollback to specific deployment
rollback_to_deployment() {
    local target_deployment=$1
    
    if [[ ! "$target_deployment" =~ ^$APP_NAME-(blue|green)$ ]]; then
        log_error "Invalid deployment name: $target_deployment"
        log_info "Valid format: $APP_NAME-{blue|green}"
        exit 1
    fi
    
    # Check if deployment exists
    if ! kubectl get deployment "$target_deployment" -n "$NAMESPACE" &>/dev/null; then
        log_error "Deployment '$target_deployment' does not exist"
        exit 1
    fi
    
    # Check if deployment is healthy
    if ! check_deployment_health "$target_deployment"; then
        log_error "Target deployment '$target_deployment' is not healthy"
        log_info "Deployment status:"
        kubectl get deployment "$target_deployment" -n "$NAMESPACE"
        exit 1
    fi
    
    local target_color="${target_deployment#*-}"
    local current_color=$(get_current_color)
    
    if [[ "$target_color" == "$current_color" ]]; then
        log_warning "Deployment '$target_deployment' is already active"
        exit 0
    fi
    
    log_info "Rolling back to deployment: $target_deployment"
    
    # Switch traffic
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$target_color'"}}}'
    
    # Wait for service to update
    sleep 5
    
    # Verify the switch
    local new_current_color=$(get_current_color)
    if [[ "$new_current_color" == "$target_color" ]]; then
        log_success "Rollback completed - traffic switched to $target_color"
    else
        log_error "Rollback failed - service selector not updated correctly"
        exit 1
    fi
}

# Show current status
show_status() {
    log_info "Current deployment status:"
    
    local current_color=$(get_current_color)
    echo "  Active deployment color: $current_color"
    
    if [[ "$current_color" != "none" ]]; then
        local active_deployment="$APP_NAME-$current_color"
        echo "  Active deployment: $active_deployment"
        
        # Show deployment details
        kubectl get deployment "$active_deployment" -n "$NAMESPACE" 2>/dev/null || echo "  Deployment not found"
        
        # Show service details
        echo ""
        echo "Service details:"
        kubectl get service "$APP_NAME" -n "$NAMESPACE" 2>/dev/null || echo "  Service not found"
    else
        echo "  No active deployment found"
    fi
    
    echo ""
    list_deployment_history
}

# Show usage
usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  emergency                 Perform emergency rollback to the other deployment"
    echo "  to DEPLOYMENT            Rollback to specific deployment (e.g., ultra-cpp-gateway-blue)"
    echo "  status                   Show current deployment status"
    echo "  history                  Show deployment history"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 emergency             # Emergency rollback"
    echo "  $0 to ultra-cpp-gateway-blue  # Rollback to blue deployment"
    echo "  $0 status                # Show current status"
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    case $1 in
        emergency)
            emergency_rollback
            ;;
        to)
            if [[ $# -lt 2 ]]; then
                log_error "Missing deployment name"
                usage
                exit 1
            fi
            rollback_to_deployment "$2"
            ;;
        status)
            show_status
            ;;
        history)
            list_deployment_history
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

# Run main function
main "$@"