# 🚀 MinIO Setup Guide

MinIO is an S3-compatible object storage server that's perfect for development and production use. It's lightweight, fast, and can be self-hosted.

## 🐳 Quick Start with Docker (Recommended)

### Step 1: Start MinIO Server

```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  -v minio_data:/data \
  minio/minio server /data --console-address ":9001"
```

### Step 2: Verify MinIO is Running

- **API Endpoint**: http://localhost:9000
- **Console**: http://localhost:9001
- **Username**: minioadmin
- **Password**: minioadmin

### Step 3: Setup Buckets

```bash
npm run setup:minio
```

### Step 4: Test Connection

```bash
npm run test:minio
```

## 📦 Alternative: Local Installation

### macOS (Homebrew)

```bash
brew install minio/stable/minio
minio server ~/minio-data --console-address ":9001"
```

### Linux

```bash
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server ~/minio-data --console-address ":9001"
```

### Windows

```bash
# Download from https://dl.min.io/server/minio/release/windows-amd64/minio.exe
minio.exe server C:\minio-data --console-address ":9001"
```

## ⚙️ Configuration

Your `.env` file is already configured with default values:

```env
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BACKUP_BUCKET=user-data-backups
```

### For Production:

```env
MINIO_ENDPOINT=your-minio-server.com
MINIO_PORT=9000
MINIO_USE_SSL=true
MINIO_ACCESS_KEY=your-secure-access-key
MINIO_SECRET_KEY=your-secure-secret-key
```

## 🧪 Testing the Setup

### 1. Test MinIO Connection

```bash
npm run test:minio
```

### 2. Create a Backup

```bash
# Get JWT token first
JWT_TOKEN=$(curl -s -X POST http://localhost:3005/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "superadmin@example.com", "password": "SuperAdmin@123"}' | \
  jq -r '.token')

# Create backup
curl -X POST http://localhost:3005/api/backup-blog \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json"
```

### 3. View in MinIO Console

1. Go to http://localhost:9001
2. Login with minioadmin/minioadmin
3. Navigate to `user-data-backups` bucket
4. See your backup files

## 🔧 Features

### ✅ What Works Now:

- **Backup Creation**: Blog posts + media metadata → MinIO
- **Presigned URLs**: Secure download links (1 hour expiry)
- **Bucket Management**: Auto-creation and configuration
- **Metadata**: Rich file metadata and user information
- **JSON Format**: Human-readable backup format

### 🚀 Advanced Features:

- **Versioning**: Keep multiple backup versions
- **Lifecycle Policies**: Auto-delete old backups
- **Encryption**: Server-side encryption
- **Replication**: Multi-site backup replication

## 📊 MinIO Console Features

Access at http://localhost:9001:

### Buckets

- View all buckets and objects
- Upload/download files manually
- Set bucket policies and lifecycle rules

### Monitoring

- Storage usage statistics
- API request metrics
- Performance monitoring

### Security

- Access key management
- Bucket policies
- User management

## 🛡️ Security Best Practices

### 1. Change Default Credentials

```bash
# In production, use strong credentials
MINIO_ACCESS_KEY=your-strong-access-key-20-chars
MINIO_SECRET_KEY=your-strong-secret-key-40-chars
```

### 2. Enable HTTPS

```bash
# Generate certificates
openssl req -new -x509 -days 365 -nodes -out server.crt -keyout server.key

# Start MinIO with HTTPS
minio server ~/minio-data --certs-dir ~/.minio/certs --console-address ":9001"
```

### 3. Restrict Access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "AWS": ["arn:aws:iam::*:user/backup-service"] },
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::user-data-backups/*"]
    }
  ]
}
```

## 💰 Storage Costs

MinIO is **free and open source**:

- No per-request charges
- No data transfer fees
- Only pay for your server/storage costs
- Perfect for development and small-scale production

## 🔄 Migration from S3

MinIO is S3-compatible, so you can:

1. Use same S3 SDKs and tools
2. Migrate data with `mc mirror`
3. Switch endpoints without code changes

## 🐛 Troubleshooting

### Connection Refused

```
Error: connect ECONNREFUSED 127.0.0.1:9000
```

**Fix**: Start MinIO server first

### Access Denied

```
Error: Access Denied
```

**Fix**: Check access key/secret key in .env

### Bucket Not Found

```
Error: The specified bucket does not exist
```

**Fix**: Run `npm run setup:minio` to create buckets

### Port Already in Use

```
Error: bind: address already in use
```

**Fix**: Stop existing MinIO or use different ports

## 📚 Useful Commands

```bash
# Start MinIO (Docker)
docker start minio

# Stop MinIO
docker stop minio

# View logs
docker logs minio

# Remove container
docker rm minio

# Setup buckets
npm run setup:minio

# Test connection
npm run test:minio

# Create backup
npm run test:backup
```

## 🎯 Production Deployment

### Docker Compose

```yaml
version: "3.8"
services:
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: your-access-key
      MINIO_ROOT_PASSWORD: your-secret-key
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

volumes:
  minio_data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
        - name: minio
          image: minio/minio
          ports:
            - containerPort: 9000
            - containerPort: 9001
          env:
            - name: MINIO_ROOT_USER
              value: "your-access-key"
            - name: MINIO_ROOT_PASSWORD
              value: "your-secret-key"
          command: ["minio", "server", "/data", "--console-address", ":9001"]
```

---

**Ready to backup your blog data with MinIO! 🚀**

Next: Run `docker run -d --name minio -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"` and then `npm run setup:minio`
