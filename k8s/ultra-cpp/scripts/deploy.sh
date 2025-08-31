#!/bin/bash

# Ultra C++ System Blue-Green Deployment Script
# This script implements a blue-green deployment strategy for the ultra-low-latency C++ system

set -euo pipefail

# Configuration
NAMESPACE="ultra-cpp"
APP_NAME="ultra-cpp-gateway"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ultra-cpp}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
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

# Get target color (opposite of current)
get_target_color() {
    local current_color=$1
    
    if [[ "$current_color" == "blue" ]]; then
        echo "green"
    elif [[ "$current_color" == "green" ]]; then
        echo "blue"
    else
        echo "blue"  # Default to blue for initial deployment
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment_name=$1
    local timeout=$2
    
    log_info "Waiting for deployment '$deployment_name' to be ready (timeout: ${timeout}s)..."
    
    if kubectl wait --for=condition=available --timeout="${timeout}s" deployment/"$deployment_name" -n "$NAMESPACE"; then
        log_success "Deployment '$deployment_name' is ready"
        return 0
    else
        log_error "Deployment '$deployment_name' failed to become ready within ${timeout}s"
        return 1
    fi
}

# Health check function
perform_health_check() {
    local deployment_name=$1
    local timeout=$2
    
    log_info "Performing health checks for '$deployment_name'..."
    
    # Get pod IPs for the deployment
    local pod_ips=$(kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME",color="${deployment_name#*-}" -o jsonpath='{.items[*].status.podIP}')
    
    if [[ -z "$pod_ips" ]]; then
        log_error "No pods found for deployment '$deployment_name'"
        return 1
    fi
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    for pod_ip in $pod_ips; do
        log_info "Checking health for pod at $pod_ip..."
        
        while [[ $(date +%s) -lt $end_time ]]; do
            # Check liveness
            if kubectl exec -n "$NAMESPACE" deployment/"$deployment_name" -- curl -f -s "http://$pod_ip:8081/health/live" > /dev/null 2>&1; then
                log_info "Liveness check passed for $pod_ip"
                
                # Check readiness
                if kubectl exec -n "$NAMESPACE" deployment/"$deployment_name" -- curl -f -s "http://$pod_ip:8081/health/ready" > /dev/null 2>&1; then
                    log_success "Health check passed for $pod_ip"
                    break
                else
                    log_warning "Readiness check failed for $pod_ip, retrying..."
                fi
            else
                log_warning "Liveness check failed for $pod_ip, retrying..."
            fi
            
            sleep 5
        done
        
        # Final check
        if ! kubectl exec -n "$NAMESPACE" deployment/"$deployment_name" -- curl -f -s "http://$pod_ip:8081/health/ready" > /dev/null 2>&1; then
            log_error "Health check failed for pod at $pod_ip"
            return 1
        fi
    done
    
    log_success "All health checks passed for '$deployment_name'"
    return 0
}

# Performance validation
validate_performance() {
    local deployment_name=$1
    
    log_info "Validating performance for '$deployment_name'..."
    
    # Get a pod from the deployment
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME",color="${deployment_name#*-}" -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for performance validation"
        return 1
    fi
    
    # Run performance test
    log_info "Running performance test on pod '$pod_name'..."
    
    # Simple latency test
    local latency_result=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -w "%{time_total}" -s -o /dev/null "http://localhost:8080/health" 2>/dev/null || echo "999")
    
    # Convert to milliseconds
    local latency_ms=$(echo "$latency_result * 1000" | bc -l 2>/dev/null || echo "999")
    local latency_int=${latency_ms%.*}
    
    log_info "Response latency: ${latency_int}ms"
    
    # Check if latency is acceptable (< 10ms for health endpoint)
    if [[ $latency_int -lt 10 ]]; then
        log_success "Performance validation passed (latency: ${latency_int}ms)"
        return 0
    else
        log_error "Performance validation failed (latency: ${latency_int}ms > 10ms)"
        return 1
    fi
}

# Create deployment manifest
create_deployment_manifest() {
    local color=$1
    local deployment_name="$APP_NAME-$color"
    
    cat <<EOF > "/tmp/${deployment_name}-deployment.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $deployment_name
  namespace: $NAMESPACE
  labels:
    app: $APP_NAME
    color: $color
    version: $IMAGE_TAG
spec:
  replicas: 3
  selector:
    matchLabels:
      app: $APP_NAME
      color: $color
  template:
    metadata:
      labels:
        app: $APP_NAME
        color: $color
        version: $IMAGE_TAG
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeSelector:
        ultra-cpp.io/dpdk-enabled: "true"
        kubernetes.io/arch: amd64
      tolerations:
      - key: ultra-cpp.io/dedicated
        operator: Equal
        value: "true"
        effect: NoSchedule
      containers:
      - name: ultra-cpp-gateway
        image: $IMAGE_REGISTRY/gateway:$IMAGE_TAG
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        - containerPort: 8081
          name: health
          protocol: TCP
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: DEPLOYMENT_COLOR
          value: "$color"
        - name: DPDK_ENABLED
          value: "true"
        - name: WORKER_THREADS
          value: "4"
        - name: MEMORY_POOL_SIZE
          value: "2147483648"
        - name: FALLBACK_UPSTREAM
          value: "http://nodejs-api:3005"
        resources:
          requests:
            memory: "4Gi"
            cpu: "4"
            hugepages-2Mi: "2Gi"
            ultra-cpp.io/dpdk-ports: "2"
          limits:
            memory: "8Gi"
            cpu: "8"
            hugepages-2Mi: "4Gi"
            ultra-cpp.io/dpdk-ports: "2"
        securityContext:
          privileged: true
          capabilities:
            add:
            - IPC_LOCK
            - SYS_NICE
            - NET_ADMIN
            - NET_RAW
        volumeMounts:
        - name: hugepages
          mountPath: /dev/hugepages
        - name: dpdk-devices
          mountPath: /dev/uio
        - name: config
          mountPath: /etc/ultra-cpp
          readOnly: true
        - name: logs
          mountPath: /var/log/ultra-cpp
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 2
          timeoutSeconds: 1
          failureThreshold: 30
      volumes:
      - name: hugepages
        emptyDir:
          medium: HugePages-2Mi
      - name: dpdk-devices
        hostPath:
          path: /dev/uio
          type: Directory
      - name: config
        configMap:
          name: ultra-cpp-config
      - name: logs
        hostPath:
          path: /var/log/ultra-cpp
          type: DirectoryOrCreate
      terminationGracePeriodSeconds: 30
EOF
}

# Switch traffic to new deployment
switch_traffic() {
    local target_color=$1
    
    log_info "Switching traffic to $target_color deployment..."
    
    # Update service selector
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$target_color'"}}}'
    
    # Wait a moment for the change to propagate
    sleep 5
    
    log_success "Traffic switched to $target_color deployment"
}

# Rollback to previous deployment
rollback() {
    local current_color=$1
    local previous_color=$2
    
    log_warning "Rolling back from $current_color to $previous_color..."
    
    # Switch traffic back
    switch_traffic "$previous_color"
    
    # Clean up failed deployment
    kubectl delete deployment "$APP_NAME-$current_color" -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Rollback completed"
}

# Clean up old deployment
cleanup_old_deployment() {
    local old_color=$1
    
    log_info "Cleaning up old $old_color deployment..."
    
    # Scale down old deployment
    kubectl scale deployment "$APP_NAME-$old_color" -n "$NAMESPACE" --replicas=0 || true
    
    # Wait for pods to terminate
    sleep 10
    
    # Delete old deployment
    kubectl delete deployment "$APP_NAME-$old_color" -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Old $old_color deployment cleaned up"
}

# Main deployment function
deploy() {
    log_info "Starting blue-green deployment for $APP_NAME:$IMAGE_TAG"
    
    # Check prerequisites
    check_prerequisites
    
    # Determine current and target colors
    local current_color=$(get_current_color)
    local target_color=$(get_target_color "$current_color")
    
    log_info "Current deployment color: $current_color"
    log_info "Target deployment color: $target_color"
    
    # Create new deployment
    local target_deployment="$APP_NAME-$target_color"
    
    log_info "Creating $target_color deployment..."
    create_deployment_manifest "$target_color"
    kubectl apply -f "/tmp/${target_deployment}-deployment.yaml"
    
    # Wait for deployment to be ready
    if ! wait_for_deployment "$target_deployment" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" && "$current_color" != "none" ]]; then
            rollback "$target_color" "$current_color"
        fi
        exit 1
    fi
    
    # Perform health checks
    if ! perform_health_check "$target_deployment" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Health checks failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" && "$current_color" != "none" ]]; then
            rollback "$target_color" "$current_color"
        fi
        exit 1
    fi
    
    # Validate performance
    if ! validate_performance "$target_deployment"; then
        log_error "Performance validation failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" && "$current_color" != "none" ]]; then
            rollback "$target_color" "$current_color"
        fi
        exit 1
    fi
    
    # Switch traffic to new deployment
    switch_traffic "$target_color"
    
    # Clean up old deployment if it exists
    if [[ "$current_color" != "none" ]]; then
        cleanup_old_deployment "$current_color"
    fi
    
    log_success "Blue-green deployment completed successfully!"
    log_success "Active deployment: $target_color"
    
    # Clean up temporary files
    rm -f "/tmp/${target_deployment}-deployment.yaml"
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --image-tag TAG        Docker image tag to deploy (default: latest)"
    echo "  -r, --registry REGISTRY    Docker registry (default: ultra-cpp)"
    echo "  -t, --timeout SECONDS      Health check timeout (default: 300)"
    echo "  -n, --no-rollback         Disable automatic rollback on failure"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_TAG                 Docker image tag"
    echo "  IMAGE_REGISTRY           Docker registry"
    echo "  HEALTH_CHECK_TIMEOUT     Health check timeout in seconds"
    echo "  ROLLBACK_ON_FAILURE      Enable/disable rollback (true/false)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            IMAGE_REGISTRY="$2"
            shift 2
            ;;
        -t|--timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        -n|--no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run deployment
deploy