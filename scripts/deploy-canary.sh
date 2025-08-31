#!/bin/bash

# Canary Deployment Script for AWS Deployment System
# This script manages canary deployments using Argo Rollouts

set -euo pipefail

# Configuration
NAMESPACE="production"
SERVICES=("api-service" "media-service" "worker-service")
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${GITHUB_SHA:-latest}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if argo rollouts plugin is available
    if ! kubectl argo rollouts version &> /dev/null; then
        log_error "Argo Rollouts kubectl plugin is not installed"
        log_info "Install with: curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy analysis templates
deploy_analysis_templates() {
    log_info "Deploying analysis templates..."
    
    kubectl apply -f k8s/argo-rollouts/analysis-templates.yaml
    
    log_success "Analysis templates deployed"
}

# Deploy Istio traffic routing
deploy_istio_config() {
    log_info "Deploying Istio traffic routing configuration..."
    
    kubectl apply -f k8s/argo-rollouts/istio-traffic-routing.yaml
    
    log_success "Istio configuration deployed"
}

# Deploy a specific service
deploy_service() {
    local service=$1
    local image_tag=$2
    
    log_info "Deploying $service with image tag: $image_tag"
    
    # Apply the rollout configuration
    kubectl apply -f "k8s/argo-rollouts/${service}-rollout.yaml"
    
    # Set the new image
    kubectl argo rollouts set image "$service" \
        "$service=${ECR_REGISTRY}/${service}:${image_tag}" \
        -n "$NAMESPACE"
    
    log_success "$service rollout initiated"
}

# Monitor rollout progress
monitor_rollout() {
    local service=$1
    
    log_info "Monitoring $service rollout progress..."
    
    # Wait for rollout to complete or fail
    if kubectl argo rollouts get rollout "$service" -n "$NAMESPACE" --watch --timeout=1800s; then
        log_success "$service rollout completed successfully"
        return 0
    else
        log_error "$service rollout failed or timed out"
        return 1
    fi
}

# Get rollout status
get_rollout_status() {
    local service=$1
    
    local status=$(kubectl argo rollouts get rollout "$service" -n "$NAMESPACE" -o json | jq -r '.status.phase')
    echo "$status"
}

# Promote rollout
promote_rollout() {
    local service=$1
    
    log_info "Promoting $service rollout..."
    
    kubectl argo rollouts promote "$service" -n "$NAMESPACE"
    
    log_success "$service rollout promoted"
}

# Abort rollout
abort_rollout() {
    local service=$1
    
    log_warning "Aborting $service rollout..."
    
    kubectl argo rollouts abort "$service" -n "$NAMESPACE"
    
    log_warning "$service rollout aborted"
}

# Rollback to previous version
rollback_service() {
    local service=$1
    
    log_warning "Rolling back $service to previous version..."
    
    kubectl argo rollouts undo "$service" -n "$NAMESPACE"
    
    log_warning "$service rolled back"
}

# Health check after deployment
health_check() {
    local service=$1
    
    log_info "Performing health check for $service..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app="$service" -n "$NAMESPACE" --timeout=300s
    
    # Check if service is responding
    local service_url="http://${service}.${NAMESPACE}.svc.cluster.local"
    if [[ "$service" == "api-service" ]]; then
        service_url="${service_url}:3000/health"
    elif [[ "$service" == "media-service" ]]; then
        service_url="${service_url}:8080/health"
    fi
    
    # Use a test pod to check service health
    kubectl run health-check-${service} --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "$service_url" || {
        log_error "Health check failed for $service"
        return 1
    }
    
    log_success "Health check passed for $service"
}

# Main deployment function
deploy_all_services() {
    local image_tag=${1:-$IMAGE_TAG}
    
    log_info "Starting canary deployment for all services with tag: $image_tag"
    
    # Deploy prerequisites
    deploy_analysis_templates
    deploy_istio_config
    
    # Deploy services
    for service in "${SERVICES[@]}"; do
        deploy_service "$service" "$image_tag"
    done
    
    # Monitor deployments
    local failed_services=()
    for service in "${SERVICES[@]}"; do
        if ! monitor_rollout "$service"; then
            failed_services+=("$service")
        fi
    done
    
    # Check if any deployments failed
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "The following services failed to deploy: ${failed_services[*]}"
        
        # Ask user if they want to rollback failed services
        read -p "Do you want to rollback failed services? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for service in "${failed_services[@]}"; do
                rollback_service "$service"
            done
        fi
        
        exit 1
    fi
    
    log_success "All services deployed successfully!"
}

# Show rollout status for all services
show_status() {
    log_info "Current rollout status:"
    
    for service in "${SERVICES[@]}"; do
        local status=$(get_rollout_status "$service")
        echo "  $service: $status"
    done
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    
    # Remove any temporary test pods
    kubectl delete pod --field-selector=status.phase==Succeeded -n "$NAMESPACE" 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_prerequisites
        deploy_all_services "${2:-}"
        ;;
    "status")
        show_status
        ;;
    "promote")
        if [[ -z "${2:-}" ]]; then
            log_error "Service name required for promote command"
            exit 1
        fi
        promote_rollout "$2"
        ;;
    "abort")
        if [[ -z "${2:-}" ]]; then
            log_error "Service name required for abort command"
            exit 1
        fi
        abort_rollout "$2"
        ;;
    "rollback")
        if [[ -z "${2:-}" ]]; then
            log_error "Service name required for rollback command"
            exit 1
        fi
        rollback_service "$2"
        ;;
    "health-check")
        if [[ -z "${2:-}" ]]; then
            log_error "Service name required for health-check command"
            exit 1
        fi
        health_check "$2"
        ;;
    *)
        echo "Usage: $0 {deploy|status|promote|abort|rollback|health-check} [service-name|image-tag]"
        echo ""
        echo "Commands:"
        echo "  deploy [image-tag]     - Deploy all services with canary strategy"
        echo "  status                 - Show current rollout status"
        echo "  promote <service>      - Promote a service rollout"
        echo "  abort <service>        - Abort a service rollout"
        echo "  rollback <service>     - Rollback a service to previous version"
        echo "  health-check <service> - Perform health check on a service"
        exit 1
        ;;
esac