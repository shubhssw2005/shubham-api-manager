#!/usr/bin/env node

import { S3Client, CreateBucketCommand, PutBucketPolicyCommand, PutBucketVersioningCommand, PutBucketLifecycleConfigurationCommand } from '@aws-sdk/client-s3';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * AWS S3 Setup Script
 * Creates and configures S3 buckets for the Universal Data Management System
 */

const s3Client = new S3Client({
    region: process.env.AWS_REGION || 'us-east-1',
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

const buckets = [
    {
        name: process.env.S3_BACKUP_BUCKET || 'user-data-backups',
        description: 'User blog data backups',
        lifecycle: true
    },
    {
        name: process.env.EVENT_BUCKET_NAME || 'event-stream-bucket',
        description: 'Event sourcing data',
        lifecycle: true
    },
    {
        name: process.env.ARCHIVE_BUCKET_NAME || 'archive-bucket',
        description: 'Archived data storage',
        lifecycle: false
    }
];

async function createBucket(bucketName, region = 'us-east-1') {
    try {
        console.log(`🪣 Creating bucket: ${bucketName}`);
        
        const createParams = {
            Bucket: bucketName
        };
        
        // Add location constraint if not us-east-1
        if (region !== 'us-east-1') {
            createParams.CreateBucketConfiguration = {
                LocationConstraint: region
            };
        }
        
        await s3Client.send(new CreateBucketCommand(createParams));
        console.log(`✅ Bucket created: ${bucketName}`);
        
    } catch (error) {
        if (error.name === 'BucketAlreadyOwnedByYou') {
            console.log(`ℹ️  Bucket already exists: ${bucketName}`);
        } else if (error.name === 'BucketAlreadyExists') {
            console.log(`⚠️  Bucket name taken: ${bucketName}`);
            throw error;
        } else {
            console.error(`❌ Failed to create bucket ${bucketName}:`, error.message);
            throw error;
        }
    }
}

async function configureBucketVersioning(bucketName) {
    try {
        console.log(`🔄 Enabling versioning for: ${bucketName}`);
        
        await s3Client.send(new PutBucketVersioningCommand({
            Bucket: bucketName,
            VersioningConfiguration: {
                Status: 'Enabled'
            }
        }));
        
        console.log(`✅ Versioning enabled: ${bucketName}`);
        
    } catch (error) {
        console.error(`❌ Failed to enable versioning for ${bucketName}:`, error.message);
    }
}

async function configureBucketLifecycle(bucketName) {
    try {
        console.log(`📅 Setting up lifecycle policy for: ${bucketName}`);
        
        const lifecycleConfig = {
            Bucket: bucketName,
            LifecycleConfiguration: {
                Rules: [
                    {
                        ID: 'DataLifecycleRule',
                        Status: 'Enabled',
                        Filter: {},
                        Transitions: [
                            {
                                Days: 30,
                                StorageClass: 'STANDARD_IA'
                            },
                            {
                                Days: 90,
                                StorageClass: 'GLACIER'
                            },
                            {
                                Days: 365,
                                StorageClass: 'DEEP_ARCHIVE'
                            }
                        ]
                    },
                    {
                        ID: 'DeleteOldVersions',
                        Status: 'Enabled',
                        Filter: {},
                        NoncurrentVersionExpiration: {
                            NoncurrentDays: 90
                        }
                    }
                ]
            }
        };
        
        await s3Client.send(new PutBucketLifecycleConfigurationCommand(lifecycleConfig));
        console.log(`✅ Lifecycle policy configured: ${bucketName}`);
        
    } catch (error) {
        console.error(`❌ Failed to configure lifecycle for ${bucketName}:`, error.message);
    }
}

async function setBucketPolicy(bucketName) {
    try {
        console.log(`🔒 Setting bucket policy for: ${bucketName}`);
        
        // Basic policy to allow the application to read/write
        const policy = {
            Version: '2012-10-17',
            Statement: [
                {
                    Sid: 'AllowApplicationAccess',
                    Effect: 'Allow',
                    Principal: {
                        AWS: `arn:aws:iam::*:user/*` // You should replace this with your specific user ARN
                    },
                    Action: [
                        's3:GetObject',
                        's3:PutObject',
                        's3:DeleteObject',
                        's3:ListBucket'
                    ],
                    Resource: [
                        `arn:aws:s3:::${bucketName}`,
                        `arn:aws:s3:::${bucketName}/*`
                    ]
                }
            ]
        };
        
        await s3Client.send(new PutBucketPolicyCommand({
            Bucket: bucketName,
            Policy: JSON.stringify(policy)
        }));
        
        console.log(`✅ Bucket policy set: ${bucketName}`);
        
    } catch (error) {
        console.log(`⚠️  Could not set bucket policy for ${bucketName}: ${error.message}`);
        // This is not critical, so we continue
    }
}

async function testS3Connection() {
    try {
        console.log('🔍 Testing S3 connection...');
        
        // Try to list buckets to test connection
        const { Buckets } = await s3Client.send(new ListBucketsCommand({}));
        
        console.log('✅ S3 connection successful!');
        console.log(`📊 You have ${Buckets.length} buckets in your account`);
        
        return true;
    } catch (error) {
        console.error('❌ S3 connection failed:', error.message);
        return false;
    }
}

async function setupS3() {
    console.log('🚀 Starting S3 setup for Universal Data Management System...\n');
    
    // Check credentials
    if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'your-aws-access-key-here') {
        console.error('❌ AWS credentials not configured!');
        console.log('\n📝 Please update your .env file with:');
        console.log('AWS_ACCESS_KEY_ID=your-actual-access-key');
        console.log('AWS_SECRET_ACCESS_KEY=your-actual-secret-key');
        console.log('AWS_REGION=your-preferred-region');
        console.log('\n💡 Get these from: AWS Console → IAM → Users → Your User → Security Credentials');
        process.exit(1);
    }
    
    // Test connection first
    const connectionOk = await testS3Connection();
    if (!connectionOk) {
        console.error('❌ Cannot proceed without S3 connection');
        process.exit(1);
    }
    
    console.log('\n🪣 Setting up S3 buckets...\n');
    
    const region = process.env.AWS_REGION || 'us-east-1';
    
    for (const bucket of buckets) {
        try {
            console.log(`\n--- Setting up ${bucket.name} (${bucket.description}) ---`);
            
            // Create bucket
            await createBucket(bucket.name, region);
            
            // Configure versioning
            await configureBucketVersioning(bucket.name);
            
            // Configure lifecycle if needed
            if (bucket.lifecycle) {
                await configureBucketLifecycle(bucket.name);
            }
            
            // Set bucket policy
            await setBucketPolicy(bucket.name);
            
            console.log(`✅ ${bucket.name} setup complete\n`);
            
        } catch (error) {
            console.error(`❌ Failed to setup ${bucket.name}:`, error.message);
        }
    }
    
    console.log('🎉 S3 setup completed!\n');
    console.log('📋 Summary:');
    buckets.forEach(bucket => {
        console.log(`  ✅ ${bucket.name} - ${bucket.description}`);
    });
    
    console.log('\n🔧 Next steps:');
    console.log('1. Test the backup system: npm run test:backup');
    console.log('2. Create your first backup: curl -X POST http://localhost:3005/api/backup-blog');
    console.log('3. Monitor your S3 usage in AWS Console');
}

// Import ListBucketsCommand
import { ListBucketsCommand } from '@aws-sdk/client-s3';

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    setupS3().catch(error => {
        console.error('❌ Setup failed:', error);
        process.exit(1);
    });
}

export default setupS3;