#!/bin/bash

# Deploy Observability Stack
# This script deploys Prometheus, Grafana, Jaeger, and Alertmanager

set -e

# Configuration
NAMESPACE=${NAMESPACE:-observability}
ENVIRONMENT=${ENVIRONMENT:-development}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-observability}
VALUES_FILE=${VALUES_FILE:-values.yaml}

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
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
    
    # Label namespace for monitoring
    kubectl label namespace "$NAMESPACE" monitoring=enabled --overwrite
}

# Install Prometheus Operator CRDs
install_prometheus_crds() {
    log_info "Installing Prometheus Operator CRDs..."
    
    # Add prometheus-community helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install kube-prometheus-stack for CRDs
    if helm list -n "$NAMESPACE" | grep -q "prometheus-operator"; then
        log_warning "Prometheus Operator already installed"
    else
        helm install prometheus-operator prometheus-community/kube-prometheus-stack \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --set prometheus.enabled=false \
            --set grafana.enabled=false \
            --set alertmanager.enabled=false \
            --set kubeStateMetrics.enabled=false \
            --set nodeExporter.enabled=false \
            --set prometheusOperator.enabled=true
        
        log_success "Prometheus Operator CRDs installed"
    fi
}

# Deploy Node Exporter
deploy_node_exporter() {
    log_info "Deploying Node Exporter..."
    
    if helm list -n "$NAMESPACE" | grep -q "node-exporter"; then
        log_warning "Node Exporter already deployed"
    else
        helm install node-exporter prometheus-community/prometheus-node-exporter \
            --namespace "$NAMESPACE" \
            --set service.annotations."prometheus\.io/scrape"="true" \
            --set service.annotations."prometheus\.io/port"="9100"
        
        log_success "Node Exporter deployed"
    fi
}

# Deploy kube-state-metrics
deploy_kube_state_metrics() {
    log_info "Deploying kube-state-metrics..."
    
    if helm list -n "$NAMESPACE" | grep -q "kube-state-metrics"; then
        log_warning "kube-state-metrics already deployed"
    else
        helm install kube-state-metrics prometheus-community/kube-state-metrics \
            --namespace "$NAMESPACE"
        
        log_success "kube-state-metrics deployed"
    fi
}

# Deploy observability stack
deploy_observability_stack() {
    log_info "Deploying observability stack..."
    
    # Determine values file based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        VALUES_FILE="values-production.yaml"
    else
        VALUES_FILE="values.yaml"
    fi
    
    local values_path="../helm/observability/$VALUES_FILE"
    
    if [ ! -f "$values_path" ]; then
        log_error "Values file not found: $values_path"
        exit 1
    fi
    
    # Deploy or upgrade the observability stack
    if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE_NAME"; then
        log_info "Upgrading existing observability stack..."
        helm upgrade "$HELM_RELEASE_NAME" ../helm/observability \
            --namespace "$NAMESPACE" \
            --values "$values_path" \
            --timeout 10m \
            --wait
    else
        log_info "Installing new observability stack..."
        helm install "$HELM_RELEASE_NAME" ../helm/observability \
            --namespace "$NAMESPACE" \
            --values "$values_path" \
            --timeout 10m \
            --wait
    fi
    
    log_success "Observability stack deployed successfully"
}

# Wait for pods to be ready
wait_for_pods() {
    log_info "Waiting for pods to be ready..."
    
    # Wait for all pods in the namespace to be ready
    kubectl wait --for=condition=ready pod --all -n "$NAMESPACE" --timeout=300s
    
    log_success "All pods are ready"
}

# Create service monitors for application services
create_service_monitors() {
    log_info "Creating service monitors for application services..."
    
    # Apply service monitors
    kubectl apply -f ../helm/observability/templates/servicemonitor-api.yaml -n "$NAMESPACE"
    kubectl apply -f ../helm/observability/templates/servicemonitor-media.yaml -n "$NAMESPACE"
    
    log_success "Service monitors created"
}

# Setup port forwarding for local access
setup_port_forwarding() {
    if [ "$ENVIRONMENT" != "production" ]; then
        log_info "Setting up port forwarding for local access..."
        
        # Kill existing port forwards
        pkill -f "kubectl port-forward" || true
        
        # Grafana
        kubectl port-forward -n "$NAMESPACE" svc/grafana 3000:80 &
        log_info "Grafana available at: http://localhost:3000"
        
        # Prometheus
        kubectl port-forward -n "$NAMESPACE" svc/prometheus-server 9090:80 &
        log_info "Prometheus available at: http://localhost:9090"
        
        # Jaeger
        kubectl port-forward -n "$NAMESPACE" svc/jaeger-query 16686:16686 &
        log_info "Jaeger available at: http://localhost:16686"
        
        # Alertmanager
        kubectl port-forward -n "$NAMESPACE" svc/alertmanager 9093:9093 &
        log_info "Alertmanager available at: http://localhost:9093"
        
        log_success "Port forwarding setup complete"
        log_info "Use 'pkill -f \"kubectl port-forward\"' to stop port forwarding"
    fi
}

# Display deployment status
show_deployment_status() {
    log_info "Deployment Status:"
    echo
    
    # Show helm releases
    log_info "Helm Releases:"
    helm list -n "$NAMESPACE"
    echo
    
    # Show pods
    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE"
    echo
    
    # Show services
    log_info "Services:"
    kubectl get services -n "$NAMESPACE"
    echo
    
    # Show ingresses if any
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Ingresses:"
        kubectl get ingress -n "$NAMESPACE"
        echo
    fi
    
    # Show service monitors
    log_info "Service Monitors:"
    kubectl get servicemonitor -n "$NAMESPACE" 2>/dev/null || log_warning "No ServiceMonitors found"
    echo
    
    # Show prometheus rules
    log_info "Prometheus Rules:"
    kubectl get prometheusrule -n "$NAMESPACE" 2>/dev/null || log_warning "No PrometheusRules found"
}

# Main deployment function
main() {
    log_info "Starting observability stack deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Helm Release: $HELM_RELEASE_NAME"
    echo
    
    check_prerequisites
    create_namespace
    install_prometheus_crds
    deploy_node_exporter
    deploy_kube_state_metrics
    deploy_observability_stack
    wait_for_pods
    create_service_monitors
    setup_port_forwarding
    
    echo
    log_success "Observability stack deployment completed successfully!"
    echo
    show_deployment_status
    
    if [ "$ENVIRONMENT" != "production" ]; then
        echo
        log_info "Default Credentials:"
        log_info "Grafana - admin:admin123 (change in production)"
        echo
        log_info "To access the services locally, use the port forwarding URLs above"
        log_info "To stop port forwarding: pkill -f 'kubectl port-forward'"
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        show_deployment_status
        ;;
    "port-forward")
        setup_port_forwarding
        ;;
    "cleanup")
        log_info "Cleaning up observability stack..."
        helm uninstall "$HELM_RELEASE_NAME" -n "$NAMESPACE" || true
        helm uninstall node-exporter -n "$NAMESPACE" || true
        helm uninstall kube-state-metrics -n "$NAMESPACE" || true
        helm uninstall prometheus-operator -n "$NAMESPACE" || true
        kubectl delete namespace "$NAMESPACE" || true
        pkill -f "kubectl port-forward" || true
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|status|port-forward|cleanup}"
        echo "  deploy       - Deploy the observability stack (default)"
        echo "  status       - Show deployment status"
        echo "  port-forward - Setup port forwarding for local access"
        echo "  cleanup      - Remove the observability stack"
        exit 1
        ;;
esac