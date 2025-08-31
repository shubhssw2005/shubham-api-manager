#!/bin/bash

set -euo pipefail

# Configuration
NAMESPACE=${NAMESPACE:-production}
ENVIRONMENT=${ENVIRONMENT:-production}
HELM_TIMEOUT=${HELM_TIMEOUT:-600s}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    command -v istioctl >/dev/null 2>&1 || error "istioctl is required but not installed"
    
    # Check cluster connectivity
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    log "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log "Creating namespace: $NAMESPACE"
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "$NAMESPACE" istio-injection=enabled --overwrite
    
    log "Namespace $NAMESPACE is ready"
}

# Install Istio service mesh
install_istio() {
    log "Installing Istio service mesh..."
    
    # Check if Istio is already installed
    if kubectl get namespace istio-system >/dev/null 2>&1; then
        warn "Istio namespace already exists, skipping installation"
        return
    fi
    
    # Install Istio
    istioctl install --set values.defaultRevision=default -y
    
    # Apply Istio configurations
    kubectl apply -f k8s/istio/
    
    log "Istio service mesh installed"
}

# Deploy Helm charts
deploy_services() {
    log "Deploying services..."
    
    # Update Helm dependencies
    helm dependency update k8s/helm/api-service/
    helm dependency update k8s/helm/media-service/
    helm dependency update k8s/helm/worker-service/
    
    # Deploy API service
    log "Deploying API service..."
    helm upgrade --install api-service k8s/helm/api-service/ \
        --namespace "$NAMESPACE" \
        --timeout "$HELM_TIMEOUT" \
        --values "k8s/helm/api-service/values-${ENVIRONMENT}.yaml" \
        --wait
    
    # Deploy Media service
    log "Deploying Media service..."
    helm upgrade --install media-service k8s/helm/media-service/ \
        --namespace "$NAMESPACE" \
        --timeout "$HELM_TIMEOUT" \
        --values "k8s/helm/media-service/values-${ENVIRONMENT}.yaml" \
        --wait
    
    # Deploy Worker service
    log "Deploying Worker service..."
    helm upgrade --install worker-service k8s/helm/worker-service/ \
        --namespace "$NAMESPACE" \
        --timeout "$HELM_TIMEOUT" \
        --values "k8s/helm/worker-service/values-${ENVIRONMENT}.yaml" \
        --wait
    
    log "All services deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check services
    kubectl get services -n "$NAMESPACE"
    
    # Check HPA status
    kubectl get hpa -n "$NAMESPACE"
    
    # Check Istio configuration
    istioctl proxy-status -n "$NAMESPACE"
    
    log "Deployment verification completed"
}

# Main deployment function
main() {
    log "Starting deployment to $ENVIRONMENT environment..."
    
    check_prerequisites
    create_namespace
    install_istio
    deploy_services
    verify_deployment
    
    log "Deployment completed successfully!"
    log "Access your services at:"
    log "  API: https://api.company.com"
    log "  Media: https://media.company.com"
}

# Run main function
main "$@"