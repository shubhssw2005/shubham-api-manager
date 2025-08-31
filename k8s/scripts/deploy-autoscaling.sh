#!/bin/bash

# Deploy Auto-Scaling Components
set -e

CLUSTER_NAME=${1:-""}
AWS_REGION=${2:-"us-east-1"}
PROJECT_NAME=${3:-"strapi-platform"}
ENVIRONMENT=${4:-"production"}

if [ -z "$CLUSTER_NAME" ]; then
    echo "Usage: $0 <cluster-name> [aws-region] [project-name] [environment]"
    echo "Example: $0 my-cluster us-east-1 strapi-platform production"
    exit 1
fi

echo "Deploying auto-scaling components to cluster: $CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "Project: $PROJECT_NAME"
echo "Environment: $ENVIRONMENT"

# Update kubeconfig
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Get cluster OIDC issuer URL
OIDC_ISSUER=$(aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION --query "cluster.identity.oidc.issuer" --output text)
OIDC_ID=$(echo $OIDC_ISSUER | cut -d '/' -f 5)

echo "OIDC Issuer: $OIDC_ISSUER"

# Deploy VPA CRDs first
echo "Deploying VPA CRDs..."
helm upgrade --install vpa-crds k8s/helm/common \
    --set autoscaling.verticalPodAutoscaler.enabled=true \
    --set autoscaling.clusterAutoscaler.enabled=false \
    --set clusterName=$CLUSTER_NAME \
    --set aws.region=$AWS_REGION \
    --namespace kube-system \
    --create-namespace

# Wait for CRDs to be established
echo "Waiting for VPA CRDs to be established..."
kubectl wait --for condition=established --timeout=60s crd/verticalpodautoscalers.autoscaling.k8s.io
kubectl wait --for condition=established --timeout=60s crd/verticalpodautoscalercheckpoints.autoscaling.k8s.io

# Get cluster autoscaler role ARN from Terraform output
CLUSTER_AUTOSCALER_ROLE_ARN=$(terraform -chdir=terraform output -raw cluster_autoscaler_role_arn 2>/dev/null || echo "")

if [ -z "$CLUSTER_AUTOSCALER_ROLE_ARN" ]; then
    echo "Warning: Could not get cluster autoscaler role ARN from Terraform. Please set it manually."
    CLUSTER_AUTOSCALER_ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/${PROJECT_NAME}-${ENVIRONMENT}-cluster-autoscaler-role"
fi

echo "Using Cluster Autoscaler Role ARN: $CLUSTER_AUTOSCALER_ROLE_ARN"

# Deploy Cluster Autoscaler and VPA
echo "Deploying Cluster Autoscaler and VPA..."
helm upgrade --install autoscaling-components k8s/helm/common \
    --set autoscaling.clusterAutoscaler.enabled=true \
    --set autoscaling.verticalPodAutoscaler.enabled=true \
    --set clusterName=$CLUSTER_NAME \
    --set aws.region=$AWS_REGION \
    --set autoscaling.clusterAutoscaler.serviceAccount.roleArn=$CLUSTER_AUTOSCALER_ROLE_ARN \
    --namespace kube-system \
    --create-namespace

# Wait for deployments to be ready
echo "Waiting for autoscaling components to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/cluster-autoscaler -n kube-system
kubectl wait --for=condition=available --timeout=300s deployment/vpa-admission-controller -n kube-system
kubectl wait --for=condition=available --timeout=300s deployment/vpa-recommender -n kube-system
kubectl wait --for=condition=available --timeout=300s deployment/vpa-updater -n kube-system

# Deploy metrics server if not present
if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    echo "Deploying metrics server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Patch metrics server for EKS
    kubectl patch deployment metrics-server -n kube-system --type='json' \
        -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
    
    kubectl wait --for=condition=available --timeout=300s deployment/metrics-server -n kube-system
fi

# Verify cluster autoscaler is working
echo "Verifying cluster autoscaler..."
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=10

# Show VPA status
echo "VPA Components Status:"
kubectl get pods -n kube-system -l app=vpa-admission-controller
kubectl get pods -n kube-system -l app=vpa-recommender  
kubectl get pods -n kube-system -l app=vpa-updater

echo "Auto-scaling components deployed successfully!"
echo ""
echo "Next steps:"
echo "1. Deploy your applications with VPA enabled"
echo "2. Monitor autoscaling behavior in CloudWatch and Kubernetes dashboards"
echo "3. Adjust autoscaling parameters based on workload patterns"
echo ""
echo "Useful commands:"
echo "  kubectl get vpa --all-namespaces"
echo "  kubectl describe vpa <vpa-name> -n <namespace>"
echo "  kubectl logs -n kube-system deployment/cluster-autoscaler"