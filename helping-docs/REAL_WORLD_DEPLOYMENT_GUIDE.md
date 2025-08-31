# 🌍 Real-World Deployment Guide - Production Ready

## Overview

This guide provides the exact steps to deploy your Groot API + Ultra Low-Latency C++ System to production, handling real-world scenarios, traffic, and enterprise requirements.

---

## 🎯 Deployment Scenarios

### Scenario 1: Startup/Small Business (< 10K users)
**Budget**: $200-500/month  
**Traffic**: < 1M requests/month  
**Infrastructure**: Single region, basic monitoring

### Scenario 2: Growing Business (10K-100K users)
**Budget**: $500-2000/month  
**Traffic**: 1M-10M requests/month  
**Infrastructure**: Multi-AZ, advanced monitoring, CDN

### Scenario 3: Enterprise (100K+ users)
**Budget**: $2000+/month  
**Traffic**: 10M+ requests/month  
**Infrastructure**: Multi-region, full observability, compliance

---

## 🚀 Step-by-Step Production Deployment

### Phase 1: Infrastructure Setup (Day 1-2)

#### Step 1: Domain and DNS Setup
```bash
# 1. Purchase domain from registrar (Namecheap, GoDaddy, etc.)
# Domain: yourdomain.com

# 2. Set up CloudFlare (Free plan is sufficient to start)
# - Add domain to CloudFlare
# - Update nameservers at registrar
# - Enable basic security features

# 3. DNS Records to create:
# A     yourdomain.com          → Your server IP
# CNAME www.yourdomain.com      → yourdomain.com
# CNAME api.yourdomain.com      → yourdomain.com
# CNAME admin.yourdomain.com    → yourdomain.com
```

#### Step 2: Cloud Provider Setup (Choose One)

##### Option A: AWS (Recommended for Enterprise)
```bash
# 1. Create AWS Account
# 2. Set up billing alerts
# 3. Create IAM user for deployment

# Create deployment user
aws iam create-user --user-name groot-deploy

# Attach necessary policies
aws iam attach-user-policy --user-name groot-deploy \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

# Create access keys
aws iam create-access-key --user-name groot-deploy

# 4. Create EKS cluster
eksctl create cluster \
  --name groot-production \
  --version 1.28 \
  --region us-east-1 \
  --nodegroup-name groot-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed
```

##### Option B: DigitalOcean (Cost-Effective)
```bash
# 1. Create DigitalOcean Account
# 2. Generate API token
# 3. Create Kubernetes cluster

# Using doctl CLI
doctl kubernetes cluster create groot-production \
  --region nyc1 \
  --version 1.28.2-do.0 \
  --count 3 \
  --size s-2vcpu-2gb \
  --auto-upgrade=true
```

##### Option C: Google Cloud (Good Balance)
```bash
# 1. Create GCP Project
gcloud projects create groot-production-12345

# 2. Enable APIs
gcloud services enable container.googleapis.com

# 3. Create GKE cluster
gcloud container clusters create groot-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10
```

#### Step 3: Database Setup

##### MongoDB Atlas (Recommended)
```bash
# 1. Create MongoDB Atlas account: https://cloud.mongodb.com/
# 2. Create cluster:
#    - Provider: AWS (same region as your app)
#    - Tier: M10 (minimum for production)
#    - Cluster Name: groot-production

# 3. Security Setup:
#    - Database Access: Create user 'groot-app' with readWrite permissions
#    - Network Access: Add your server IPs or 0.0.0.0/0 (with strong auth)

# 4. Get connection string:
mongodb+srv://groot-app:<password>@groot-production.xxxxx.mongodb.net/groot?retryWrites=true&w=majority
```

##### Self-Hosted MongoDB (Cost-Effective)
```yaml
# mongodb-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
spec:
  serviceName: mongodb
  replicas: 3
  template:
    spec:
      containers:
      - name: mongodb
        image: mongo:7
        ports:
        - containerPort: 27017
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          value: "admin"
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: password
        volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
  volumeClaimTemplates:
  - metadata:
      name: mongodb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

#### Step 4: Object Storage Setup

##### AWS S3
```bash
# Create S3 buckets
aws s3 mb s3://groot-media-prod-12345
aws s3 mb s3://groot-backups-prod-12345

# Set bucket policies
aws s3api put-bucket-policy --bucket groot-media-prod-12345 --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::groot-media-prod-12345/*"
  }]
}'

# Enable versioning for backups
aws s3api put-bucket-versioning \
  --bucket groot-backups-prod-12345 \
  --versioning-configuration Status=Enabled
```

##### MinIO (Self-Hosted)
```yaml
# minio-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        ports:
        - containerPort: 9000
        - containerPort: 9001
        env:
        - name: MINIO_ROOT_USER
          value: "admin"
        - name: MINIO_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: password
        command:
        - minio
        - server
        - /data
        - --console-address
        - ":9001"
        volumeMounts:
        - name: minio-data
          mountPath: /data
      volumes:
      - name: minio-data
        persistentVolumeClaim:
          claimName: minio-pvc
```

### Phase 2: Application Deployment (Day 3-4)

#### Step 1: Prepare Application for Production

##### Environment Configuration
```bash
# Create production environment file
cat > .env.production << EOF
# Application
NODE_ENV=production
PORT=3005
API_URL=https://api.yourdomain.com

# Database
MONGODB_URI=mongodb+srv://groot-app:password@groot-production.xxxxx.mongodb.net/groot

# JWT
JWT_SECRET=$(openssl rand -base64 32)
JWT_REFRESH_SECRET=$(openssl rand -base64 32)

# Storage (choose one)
# AWS S3
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BACKUP_BUCKET=groot-backups-prod-12345

# OR MinIO
MINIO_ENDPOINT=minio.default.svc.cluster.local
MINIO_PORT=9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=your-secure-password

# Email
SENDGRID_API_KEY=SG.your-api-key
SENDGRID_FROM_EMAIL=noreply@yourdomain.com

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
GA_MEASUREMENT_ID=G-XXXXXXXXXX

# Redis
REDIS_URL=redis://redis.default.svc.cluster.local:6379
EOF
```

##### Docker Image Build
```dockerfile
# Dockerfile.production
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:18-alpine AS runner
WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

USER nextjs

EXPOSE 3005
ENV PORT 3005

CMD ["npm", "start"]
```

```bash
# Build and push Docker image
docker build -f Dockerfile.production -t groot-api:v1.0.0 .
docker tag groot-api:v1.0.0 your-registry/groot-api:v1.0.0
docker push your-registry/groot-api:v1.0.0
```

#### Step 2: Kubernetes Deployment

##### Create Kubernetes Secrets
```bash
# Create secrets from environment file
kubectl create secret generic app-secrets \
  --from-env-file=.env.production \
  --namespace=production

# Create TLS secret for HTTPS
kubectl create secret tls groot-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  --namespace=production
```

##### Application Deployment
```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: groot-api
  namespace: production
  labels:
    app: groot-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: groot-api
  template:
    metadata:
      labels:
        app: groot-api
        version: v1.0.0
    spec:
      containers:
      - name: groot-api
        image: your-registry/groot-api:v1.0.0
        ports:
        - containerPort: 3005
          name: http
        env:
        - name: NODE_ENV
          value: "production"
        envFrom:
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3005
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 3005
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
      securityContext:
        fsGroup: 1001
```

##### Service and Ingress
```yaml
# k8s/production/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: groot-api-service
  namespace: production
spec:
  selector:
    app: groot-api
  ports:
  - name: http
    port: 80
    targetPort: 3005
    protocol: TCP
  type: ClusterIP

---
# k8s/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: groot-api-ingress
  namespace: production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: groot-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: groot-api-service
            port:
              number: 80
```

##### Auto-Scaling
```yaml
# k8s/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: groot-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: groot-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

#### Step 3: Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/production/

# Verify deployment
kubectl get all -n production

# Check pod logs
kubectl logs -f deployment/groot-api -n production

# Test health endpoint
kubectl port-forward svc/groot-api-service 8080:80 -n production
curl http://localhost:8080/health
```

### Phase 3: Monitoring and Observability (Day 5)

#### Step 1: Install Monitoring Stack
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus + Grafana
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=YourSecurePassword123 \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
```

#### Step 2: Configure Application Metrics
```javascript
// lib/monitoring/metrics.js
const prometheus = require('prom-client');

// Create custom metrics
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
});

const httpRequestsTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

const activeConnections = new prometheus.Gauge({
  name: 'active_connections',
  help: 'Number of active connections'
});

const databaseOperations = new prometheus.Counter({
  name: 'database_operations_total',
  help: 'Total database operations',
  labelNames: ['operation', 'collection', 'status']
});

// Middleware to collect metrics
const metricsMiddleware = (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route?.path || req.path;
    
    httpRequestDuration
      .labels(req.method, route, res.statusCode)
      .observe(duration);
    
    httpRequestsTotal
      .labels(req.method, route, res.statusCode)
      .inc();
  });
  
  next();
};

module.exports = {
  httpRequestDuration,
  httpRequestsTotal,
  activeConnections,
  databaseOperations,
  metricsMiddleware,
  register: prometheus.register
};
```

#### Step 3: Grafana Dashboards
```bash
# Access Grafana
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Login with admin/YourSecurePassword123
# Import dashboards:
# - Node.js Application Metrics (ID: 11159)
# - Kubernetes Cluster Monitoring (ID: 7249)
# - MongoDB Metrics (ID: 2583)
```

### Phase 4: Security Hardening (Day 6)

#### Step 1: SSL/TLS Setup
```bash
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### Step 2: Network Policies
```yaml
# k8s/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: groot-api-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: groot-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 3005
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: default
    ports:
    - protocol: TCP
      port: 27017  # MongoDB
    - protocol: TCP
      port: 6379   # Redis
```

#### Step 3: Security Scanning
```bash
# Install Trivy for vulnerability scanning
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan your Docker image
trivy image your-registry/groot-api:v1.0.0

# Scan Kubernetes manifests
trivy config k8s/production/
```

### Phase 5: Backup and Disaster Recovery (Day 7)

#### Step 1: Database Backup
```bash
# MongoDB Atlas automatic backups are enabled by default
# For self-hosted MongoDB, set up backup cronjob

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mongodb-backup
  namespace: production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: mongodb-backup
            image: mongo:7
            command:
            - /bin/bash
            - -c
            - |
              mongodump --host mongodb:27017 --out /backup/\$(date +%Y%m%d_%H%M%S)
              # Upload to S3 or MinIO
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```

#### Step 2: Application State Backup
```javascript
// lib/backup/disaster-recovery.js
class DisasterRecoveryService {
  constructor() {
    this.backupInterval = 24 * 60 * 60 * 1000; // 24 hours
    this.retentionDays = 30;
  }

  async createFullSystemBackup() {
    const timestamp = new Date().toISOString();
    const backupId = `system-backup-${timestamp}`;

    try {
      // 1. Database backup
      const dbBackup = await this.createDatabaseBackup();
      
      // 2. File system backup
      const filesBackup = await this.createFilesBackup();
      
      // 3. Configuration backup
      const configBackup = await this.createConfigBackup();
      
      // 4. Combine and store
      const fullBackup = {
        id: backupId,
        timestamp,
        database: dbBackup,
        files: filesBackup,
        config: configBackup,
        metadata: {
          version: process.env.APP_VERSION,
          environment: process.env.NODE_ENV
        }
      };

      await this.storeBackup(backupId, fullBackup);
      
      console.log(`Full system backup created: ${backupId}`);
      return backupId;
    } catch (error) {
      console.error('System backup failed:', error);
      throw error;
    }
  }

  async restoreFromBackup(backupId) {
    try {
      const backup = await this.retrieveBackup(backupId);
      
      // 1. Restore database
      await this.restoreDatabase(backup.database);
      
      // 2. Restore files
      await this.restoreFiles(backup.files);
      
      // 3. Restore configuration
      await this.restoreConfig(backup.config);
      
      console.log(`System restored from backup: ${backupId}`);
    } catch (error) {
      console.error('System restore failed:', error);
      throw error;
    }
  }
}
```

### Phase 6: Performance Optimization (Day 8-9)

#### Step 1: CDN Setup
```bash
# CloudFlare CDN configuration
# 1. Go to CloudFlare dashboard
# 2. Add page rules:
#    - /api/* : Cache Level: Bypass
#    - /static/* : Cache Level: Cache Everything, Edge TTL: 1 month
#    - /*.jpg, /*.png, /*.css, /*.js : Cache Level: Cache Everything

# 3. Enable optimizations:
#    - Auto Minify: CSS, JavaScript, HTML
#    - Brotli compression
#    - HTTP/2 and HTTP/3
```

#### Step 2: Database Optimization
```javascript
// Database indexes for production
const createIndexes = async () => {
  const db = mongoose.connection.db;
  
  // Posts collection indexes
  await db.collection('posts').createIndex({ author: 1, createdAt: -1 });
  await db.collection('posts').createIndex({ status: 1, createdAt: -1 });
  await db.collection('posts').createIndex({ tags: 1 });
  await db.collection('posts').createIndex({ 
    title: 'text', 
    content: 'text', 
    excerpt: 'text' 
  });
  
  // Media collection indexes
  await db.collection('media').createIndex({ uploadedBy: 1, createdAt: -1 });
  await db.collection('media').createIndex({ mimeType: 1 });
  
  // Events collection indexes
  await db.collection('events').createIndex({ aggregateId: 1, timestamp: -1 });
  await db.collection('events').createIndex({ processed: 1, createdAt: 1 });
  
  console.log('Database indexes created successfully');
};
```

#### Step 3: Caching Strategy
```javascript
// Multi-layer caching implementation
const Redis = require('ioredis');
const NodeCache = require('node-cache');

class CacheManager {
  constructor() {
    // L1: In-memory cache (fastest)
    this.memoryCache = new NodeCache({ 
      stdTTL: 300,  // 5 minutes
      maxKeys: 10000 
    });
    
    // L2: Redis cache (distributed)
    this.redisCache = new Redis({
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3
    });
  }

  async get(key) {
    // Try L1 cache first
    let value = this.memoryCache.get(key);
    if (value) {
      return JSON.parse(value);
    }

    // Try L2 cache
    value = await this.redisCache.get(key);
    if (value) {
      const parsed = JSON.parse(value);
      // Populate L1 cache
      this.memoryCache.set(key, value);
      return parsed;
    }

    return null;
  }

  async set(key, value, ttl = 300) {
    const serialized = JSON.stringify(value);
    
    // Set in both caches
    this.memoryCache.set(key, serialized, ttl);
    await this.redisCache.setex(key, ttl, serialized);
  }

  async invalidate(pattern) {
    // Clear from memory cache
    this.memoryCache.flushAll();
    
    // Clear from Redis
    const keys = await this.redisCache.keys(pattern);
    if (keys.length > 0) {
      await this.redisCache.del(...keys);
    }
  }
}
```

### Phase 7: Load Testing and Validation (Day 10)

#### Step 1: Load Testing Setup
```bash
# Install k6 load testing tool
curl https://github.com/grafana/k6/releases/download/v0.46.0/k6-v0.46.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1
sudo mv k6 /usr/local/bin/
```

#### Step 2: Load Testing Scripts
```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

export let errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200 users
    { duration: '5m', target: 200 },   // Stay at 200 users
    { duration: '2m', target: 300 },   // Ramp up to 300 users
    { duration: '5m', target: 300 },   // Stay at 300 users
    { duration: '2m', target: 0 },     // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests under 500ms
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    errors: ['rate<0.1'],
  },
};

const BASE_URL = 'https://api.yourdomain.com';

export function setup() {
  // Login and get JWT token
  let loginRes = http.post(`${BASE_URL}/api/auth/login`, {
    email: 'test@example.com',
    password: 'testpassword123'
  });
  
  return { token: loginRes.json('token') };
}

export default function(data) {
  let params = {
    headers: {
      'Authorization': `Bearer ${data.token}`,
      'Content-Type': 'application/json',
    },
  };

  // Test different endpoints
  let responses = http.batch([
    ['GET', `${BASE_URL}/health`, null, params],
    ['GET', `${BASE_URL}/api/posts`, null, params],
    ['GET', `${BASE_URL}/api/media`, null, params],
    ['POST', `${BASE_URL}/api/posts`, JSON.stringify({
      title: 'Load Test Post',
      content: 'This is a load test post',
      status: 'published'
    }), params],
  ]);

  responses.forEach((response, index) => {
    check(response, {
      'status is 200': (r) => r.status === 200,
      'response time < 500ms': (r) => r.timings.duration < 500,
    }) || errorRate.add(1);
  });

  sleep(1);
}

export function teardown(data) {
  // Cleanup test data if needed
}
```

#### Step 3: Run Load Tests
```bash
# Run load test
k6 run load-test.js

# Run with specific configuration
k6 run --vus 50 --duration 30s load-test.js

# Generate HTML report
k6 run --out json=results.json load-test.js
```

### Phase 8: Go-Live Checklist (Day 11)

#### Pre-Launch Checklist
```bash
# ✅ Infrastructure
- [ ] Domain configured and DNS propagated
- [ ] SSL certificates installed and valid
- [ ] Load balancer configured
- [ ] Auto-scaling enabled
- [ ] Backup systems tested

# ✅ Application
- [ ] All environment variables set
- [ ] Database migrations completed
- [ ] Health checks passing
- [ ] Error tracking configured
- [ ] Logging properly configured

# ✅ Security
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Network policies applied
- [ ] Secrets properly managed
- [ ] Vulnerability scans passed

# ✅ Monitoring
- [ ] Metrics collection working
- [ ] Dashboards configured
- [ ] Alerts set up
- [ ] Log aggregation working
- [ ] Performance monitoring active

# ✅ Testing
- [ ] Load testing completed
- [ ] Security testing passed
- [ ] Backup/restore tested
- [ ] Disaster recovery tested
- [ ] User acceptance testing completed
```

#### Launch Commands
```bash
# Final deployment
kubectl set image deployment/groot-api groot-api=your-registry/groot-api:v1.0.0 -n production

# Verify deployment
kubectl rollout status deployment/groot-api -n production

# Check all pods are running
kubectl get pods -n production

# Test all endpoints
curl -f https://api.yourdomain.com/health
curl -f https://api.yourdomain.com/ready

# Monitor logs
kubectl logs -f deployment/groot-api -n production
```

---

## 🔧 Post-Launch Operations

### Daily Operations
```bash
# Check system health
kubectl get pods -n production
kubectl top pods -n production
kubectl top nodes

# Check application logs
kubectl logs --tail=100 deployment/groot-api -n production

# Monitor key metrics
curl -s https://api.yourdomain.com/metrics | grep http_requests_total
```

### Weekly Operations
```bash
# Review performance metrics
# Check error rates and response times in Grafana

# Update dependencies
npm audit
npm update

# Review and rotate secrets
kubectl create secret generic app-secrets-new --from-env-file=.env.production
kubectl patch deployment groot-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"groot-api","envFrom":[{"secretRef":{"name":"app-secrets-new"}}]}]}}}}'
```

### Monthly Operations
```bash
# Security updates
trivy image your-registry/groot-api:latest

# Cost optimization review
kubectl describe nodes
kubectl top pods --all-namespaces

# Backup verification
# Test restore process with recent backup

# Performance optimization
# Review slow query logs
# Analyze cache hit rates
# Check resource utilization
```

---

## 💰 Cost Optimization

### Startup Budget ($200-500/month)
```yaml
# Minimal viable production setup
Infrastructure:
  - DigitalOcean Kubernetes: $72/month (3 nodes, 2GB RAM each)
  - MongoDB Atlas M10: $57/month
  - CloudFlare Free: $0/month
  - Domain: $12/year

Services:
  - SendGrid Free: 100 emails/day
  - Sentry Free: 5K errors/month
  - Basic monitoring: $0 (self-hosted)

Total: ~$140/month + usage-based services
```

### Growing Business ($500-2000/month)
```yaml
Infrastructure:
  - AWS EKS: $200-500/month (depending on usage)
  - MongoDB Atlas M30: $157/month
  - CloudFlare Pro: $20/month
  - AWS S3: $50-100/month

Services:
  - SendGrid Essentials: $19.95/month
  - Sentry Team: $26/month
  - DataDog: $15/host/month

Total: ~$500-800/month
```

### Enterprise ($2000+/month)
```yaml
Infrastructure:
  - Multi-region AWS EKS: $1000-3000/month
  - MongoDB Atlas M60+: $500+/month
  - CloudFlare Business: $200/month
  - AWS S3 + CloudFront: $200-500/month

Services:
  - SendGrid Pro: $89.95/month
  - Sentry Business: $80/month
  - DataDog Pro: $23/host/month
  - PagerDuty: $21/user/month

Total: $2000-5000+/month
```

---

## 🚨 Troubleshooting Common Issues

### Pod Startup Issues
```bash
# Check pod status
kubectl describe pod <pod-name> -n production

# Common issues and fixes:
# 1. ImagePullBackOff - Check image name and registry credentials
# 2. CrashLoopBackOff - Check application logs and environment variables
# 3. Pending - Check resource requests and node capacity

# Fix resource issues
kubectl patch deployment groot-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"groot-api","resources":{"requests":{"memory":"256Mi","cpu":"250m"}}}]}}}}'
```

### Database Connection Issues
```bash
# Test MongoDB connection
kubectl run mongodb-test --rm -it --image=mongo:7 -- mongo "mongodb+srv://groot-app:password@groot-production.xxxxx.mongodb.net/groot"

# Check network policies
kubectl describe networkpolicy groot-api-netpol -n production

# Verify secrets
kubectl get secret app-secrets -n production -o yaml | base64 -d
```

### Performance Issues
```bash
# Check resource usage
kubectl top pods -n production
kubectl top nodes

# Scale up if needed
kubectl scale deployment groot-api --replicas=10 -n production

# Check HPA status
kubectl describe hpa groot-api-hpa -n production
```

### SSL/TLS Issues
```bash
# Check certificate status
kubectl describe certificate groot-api-tls -n production

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manually trigger certificate renewal
kubectl delete certificate groot-api-tls -n production
kubectl apply -f k8s/production/ingress.yaml
```

---

## 📞 Support and Maintenance

### Emergency Contacts
- **Infrastructure**: Your cloud provider support
- **Database**: MongoDB Atlas support (if using)
- **Application**: Your development team
- **Security**: Your security team or consultant

### Monitoring Alerts
Set up alerts for:
- High error rates (>5%)
- High response times (>1s)
- Low success rates (<95%)
- Resource exhaustion (>80% CPU/Memory)
- Certificate expiration (30 days before)

### Maintenance Windows
- **Weekly**: Dependency updates, security patches
- **Monthly**: Major updates, performance optimization
- **Quarterly**: Disaster recovery testing, security audits

---

**🎉 Congratulations! Your Groot API system is now live in production!**

This guide has taken you through the complete journey from code to production deployment. Your system is now:

✅ **Scalable** - Can handle growing traffic  
✅ **Secure** - Protected against common threats  
✅ **Monitored** - Full observability and alerting  
✅ **Reliable** - High availability and disaster recovery  
✅ **Optimized** - Performance tuned for production loads  

Remember to:
1. **Monitor continuously** - Keep an eye on metrics and logs
2. **Update regularly** - Keep dependencies and security patches current
3. **Test thoroughly** - Regular disaster recovery and load testing
4. **Optimize constantly** - Review performance and costs monthly
5. **Document everything** - Maintain runbooks and procedures

Your production system is ready to serve users and scale with your business! 🚀