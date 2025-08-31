# AWS S3 Setup Guide

## 🚀 Quick Setup Instructions

### Step 1: Get AWS Credentials

1. **Login to AWS Console**: https://console.aws.amazon.com/
2. **Go to IAM**: Services → IAM → Users
3. **Create or Select User**: 
   - If creating new: Click "Add user" → Enter username → Select "Programmatic access"
   - If using existing: Click on your username
4. **Get Access Keys**: 
   - Go to "Security credentials" tab
   - Click "Create access key"
   - **IMPORTANT**: Download and save the credentials immediately!

### Step 2: Set Required Permissions

Your IAM user needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:ListBucket",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:PutBucketVersioning",
                "s3:PutBucketLifecycleConfiguration",
                "s3:PutBucketPolicy"
            ],
            "Resource": [
                "arn:aws:s3:::*",
                "arn:aws:s3:::*/*"
            ]
        }
    ]
}
```

### Step 3: Update Environment Variables

Edit your `.env` file:
```env
# Replace these with your actual AWS credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# Customize bucket names (must be globally unique)
S3_BACKUP_BUCKET=your-app-user-backups-12345
EVENT_BUCKET_NAME=your-app-events-12345
ARCHIVE_BUCKET_NAME=your-app-archive-12345
```

### Step 4: Run Setup Script

```bash
# This will create and configure your S3 buckets
npm run setup:s3
```

### Step 5: Test the System

```bash
# Start your server
npm run dev

# Test backup creation (replace JWT_TOKEN with actual token)
curl -X POST http://localhost:3005/api/backup-blog \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔧 Manual S3 Bucket Creation (Alternative)

If the script doesn't work, create buckets manually:

### 1. Create Buckets in AWS Console

1. Go to **S3 Console**: https://s3.console.aws.amazon.com/
2. Click **"Create bucket"**
3. Enter bucket name: `your-app-user-backups` (must be globally unique)
4. Select region: `us-east-1` (or your preferred region)
5. **Block Public Access**: Keep all boxes checked (recommended)
6. Click **"Create bucket"**

Repeat for:
- `your-app-events` (for event streaming)
- `your-app-archive` (for archived data)

### 2. Configure Lifecycle Policies

For each bucket:
1. Go to bucket → **Management** tab
2. Click **"Create lifecycle rule"**
3. Rule name: `DataLifecycle`
4. Apply to all objects
5. Add transitions:
   - After 30 days → Standard-IA
   - After 90 days → Glacier
   - After 365 days → Deep Archive

### 3. Enable Versioning

For each bucket:
1. Go to bucket → **Properties** tab
2. Find **"Bucket Versioning"**
3. Click **"Edit"** → **"Enable"**

## 🛡️ Security Best Practices

### 1. Use IAM Roles (Production)
Instead of access keys, use IAM roles when deploying to EC2/ECS/Lambda.

### 2. Rotate Access Keys
Regularly rotate your access keys (every 90 days).

### 3. Monitor Usage
Set up CloudWatch alarms for:
- Unusual API calls
- High storage costs
- Failed requests

### 4. Bucket Policies
The setup script creates basic policies. For production, restrict access further:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "RestrictToApplication",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::YOUR-ACCOUNT:user/your-app-user"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-server-side-encryption": "AES256"
                }
            }
        }
    ]
}
```

## 💰 Cost Optimization

### 1. Lifecycle Policies
- **Standard**: $0.023/GB/month (first 50TB)
- **Standard-IA**: $0.0125/GB/month (after 30 days)
- **Glacier**: $0.004/GB/month (after 90 days)
- **Deep Archive**: $0.00099/GB/month (after 365 days)

### 2. Request Costs
- **PUT/POST**: $0.0005 per 1,000 requests
- **GET**: $0.0004 per 1,000 requests

### 3. Data Transfer
- **Upload**: Free
- **Download**: $0.09/GB (first 1GB free/month)

## 🔍 Monitoring & Troubleshooting

### Common Issues

**1. Access Denied**
```
Error: Access Denied
```
- Check IAM permissions
- Verify access keys are correct
- Ensure bucket policy allows your user

**2. Bucket Already Exists**
```
Error: BucketAlreadyExists
```
- Bucket names must be globally unique
- Try adding random numbers: `your-app-backups-12345`

**3. Invalid Credentials**
```
Error: The AWS Access Key Id you provided does not exist
```
- Double-check access key ID
- Ensure secret key matches
- Check if keys are active in IAM

### Debug Commands

```bash
# Test AWS CLI connection
aws s3 ls

# Check bucket contents
aws s3 ls s3://your-bucket-name

# Test upload
echo "test" | aws s3 cp - s3://your-bucket-name/test.txt
```

## 📊 Usage Examples

### Create Backup
```bash
curl -X POST http://localhost:3005/api/backup-blog \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json"
```

### List Backups
```bash
curl -X GET http://localhost:3005/api/backup-blog \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Download Backup (Local Mode)
```bash
curl -X GET "http://localhost:3005/api/download-backup?key=backup-key" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -o backup.json
```

## 🎯 Production Deployment

### Environment Variables for Production
```env
# Production AWS settings
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Production bucket names
S3_BACKUP_BUCKET=prod-app-user-backups
EVENT_BUCKET_NAME=prod-app-events
ARCHIVE_BUCKET_NAME=prod-app-archive

# Enable S3 features
ENABLE_S3_ARCHIVE=true
```

### Docker Environment
```dockerfile
ENV AWS_REGION=us-east-1
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
```

### Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aws-credentials
type: Opaque
data:
  access-key-id: <base64-encoded-access-key>
  secret-access-key: <base64-encoded-secret-key>
```

## ✅ Verification Checklist

- [ ] AWS credentials obtained
- [ ] IAM permissions configured
- [ ] Environment variables updated
- [ ] S3 buckets created
- [ ] Lifecycle policies configured
- [ ] Versioning enabled
- [ ] Setup script runs successfully
- [ ] Backup API works
- [ ] Download works (local mode)
- [ ] Monitoring set up

## 🆘 Support

If you encounter issues:

1. **Check AWS Status**: https://status.aws.amazon.com/
2. **Review IAM Permissions**: Ensure all required permissions are granted
3. **Test with AWS CLI**: Verify credentials work outside the application
4. **Check Logs**: Look for detailed error messages in application logs
5. **Contact Support**: AWS Support or create GitHub issue

---

**Ready to backup your blog data to AWS S3! 🚀**