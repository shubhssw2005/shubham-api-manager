# 🚀 S3 Setup - Step by Step Guide

## Step 1: Get AWS Credentials (5 minutes)

### 1.1 Login to AWS Console
- Go to: https://console.aws.amazon.com/
- Login with your AWS account

### 1.2 Create IAM User (if you don't have one)
- Go to: **IAM** → **Users** → **Add user**
- Username: `blog-app-user` (or any name you prefer)
- Access type: ✅ **Programmatic access**
- Click **Next: Permissions**

### 1.3 Set Permissions
- Click **Attach existing policies directly**
- Search and select: `AmazonS3FullAccess`
- Click **Next: Tags** → **Next: Review** → **Create user**

### 1.4 Save Credentials
- **IMPORTANT**: Copy and save these immediately:
  - Access Key ID: `AKIA...`
  - Secret Access Key: `...`
- Click **Download .csv** for backup

## Step 2: Update Your .env File

Replace these lines in your `.env` file:

```env
# Replace these with your actual credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA1234567890EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Choose unique bucket names (add random numbers)
S3_BACKUP_BUCKET=your-blog-backups-12345
EVENT_BUCKET_NAME=your-blog-events-12345
ARCHIVE_BUCKET_NAME=your-blog-archive-12345
```

## Step 3: Test AWS Connection

Run this command to test your credentials:
```bash
npm run setup:s3
```

## Step 4: Switch to S3 Service

Once S3 is working, I'll help you switch from local backup to S3 backup.

---

## 🔧 Current Status

✅ **Local Backup Working**: Your backup system is currently working with local storage
✅ **Blog Data Ready**: You have 3 posts and 3 media files ready to backup
✅ **API Functional**: Backup API is responding correctly

## 📋 What You Need to Do Now

1. **Get AWS credentials** (follow Step 1 above)
2. **Update .env file** with your real AWS credentials
3. **Run setup script**: `npm run setup:s3`
4. **Let me know when ready** and I'll help you test S3 backup

## 🎯 After S3 Setup

Once you have S3 configured, we'll:
1. Switch the backup service to use S3
2. Test uploading your blog backup to S3
3. Verify download from S3
4. Set up lifecycle policies for cost optimization

---

**Ready when you are! Let me know once you have your AWS credentials.** 🚀