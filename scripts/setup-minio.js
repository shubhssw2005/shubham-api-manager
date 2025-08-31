#!/usr/bin/env node

import * as Minio from 'minio';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * MinIO Setup Script
 * Sets up MinIO buckets and tests connection
 */

const minioClient = new Minio.Client({
    endPoint: process.env.MINIO_ENDPOINT || 'localhost',
    port: parseInt(process.env.MINIO_PORT) || 9000,
    useSSL: process.env.MINIO_USE_SSL === 'true',
    accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
    secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
});

const buckets = [
    {
        name: process.env.MINIO_BACKUP_BUCKET || 'user-data-backups',
        description: 'User blog data backups'
    },
    {
        name: 'event-stream-bucket',
        description: 'Event sourcing data'
    },
    {
        name: 'archive-bucket',
        description: 'Archived data storage'
    }
];

async function testMinIOConnection() {
    try {
        console.log('🔍 Testing MinIO connection...');
        console.log(`📍 Endpoint: ${process.env.MINIO_ENDPOINT || 'localhost'}:${process.env.MINIO_PORT || 9000}`);
        console.log(`🔐 Access Key: ${(process.env.MINIO_ACCESS_KEY || 'minioadmin').substring(0, 4)}...`);
        
        const buckets = await minioClient.listBuckets();
        
        console.log('✅ MinIO connection successful!');
        console.log(`📊 Found ${buckets.length} existing buckets:`);
        
        if (buckets.length > 0) {
            buckets.forEach(bucket => {
                console.log(`  - ${bucket.name} (created: ${bucket.creationDate.toISOString().split('T')[0]})`);
            });
        } else {
            console.log('  (No existing buckets - we\'ll create them)');
        }
        
        return true;
    } catch (error) {
        console.error('❌ MinIO connection failed:', error.message);
        
        if (error.code === 'ECONNREFUSED') {
            console.log('\n💡 MinIO server is not running. Start it with:');
            console.log('   Docker: docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"');
            console.log('   Or install locally: https://min.io/download');
        } else if (error.code === 'InvalidAccessKeyId') {
            console.log('\n💡 Check your MINIO_ACCESS_KEY in .env file');
        } else if (error.code === 'SignatureDoesNotMatch') {
            console.log('\n💡 Check your MINIO_SECRET_KEY in .env file');
        }
        
        return false;
    }
}

async function createBucket(bucketName, region = 'us-east-1') {
    try {
        const exists = await minioClient.bucketExists(bucketName);
        
        if (exists) {
            console.log(`ℹ️  Bucket already exists: ${bucketName}`);
            return true;
        }
        
        console.log(`🪣 Creating bucket: ${bucketName}`);
        await minioClient.makeBucket(bucketName, region);
        console.log(`✅ Bucket created: ${bucketName}`);
        
        return true;
    } catch (error) {
        console.error(`❌ Failed to create bucket ${bucketName}:`, error.message);
        return false;
    }
}

async function setBucketPolicy(bucketName) {
    try {
        console.log(`🔒 Setting bucket policy for: ${bucketName}`);
        
        // Public read policy for download URLs
        const policy = {
            Version: '2012-10-17',
            Statement: [
                {
                    Effect: 'Allow',
                    Principal: { AWS: ['*'] },
                    Action: ['s3:GetObject'],
                    Resource: [`arn:aws:s3:::${bucketName}/*`]
                }
            ]
        };
        
        await minioClient.setBucketPolicy(bucketName, JSON.stringify(policy));
        console.log(`✅ Bucket policy set: ${bucketName}`);
        
        return true;
    } catch (error) {
        console.log(`⚠️  Could not set bucket policy for ${bucketName}: ${error.message}`);
        // This is not critical for basic functionality
        return true;
    }
}

async function setupMinIO() {
    console.log('🚀 Starting MinIO setup for Universal Data Management System...\n');
    
    // Test connection first
    const connectionOk = await testMinIOConnection();
    if (!connectionOk) {
        console.error('\n❌ Cannot proceed without MinIO connection');
        console.log('\n🐳 Quick Start with Docker:');
        console.log('docker run -d \\');
        console.log('  --name minio \\');
        console.log('  -p 9000:9000 \\');
        console.log('  -p 9001:9001 \\');
        console.log('  -e "MINIO_ROOT_USER=minioadmin" \\');
        console.log('  -e "MINIO_ROOT_PASSWORD=minioadmin" \\');
        console.log('  minio/minio server /data --console-address ":9001"');
        console.log('\nThen access MinIO Console at: http://localhost:9001');
        process.exit(1);
    }
    
    console.log('\n🪣 Setting up MinIO buckets...\n');
    
    let successCount = 0;
    
    for (const bucket of buckets) {
        try {
            console.log(`\n--- Setting up ${bucket.name} (${bucket.description}) ---`);
            
            // Create bucket
            const created = await createBucket(bucket.name);
            if (created) {
                successCount++;
                
                // Set bucket policy
                await setBucketPolicy(bucket.name);
            }
            
            console.log(`✅ ${bucket.name} setup complete\n`);
            
        } catch (error) {
            console.error(`❌ Failed to setup ${bucket.name}:`, error.message);
        }
    }
    
    console.log('🎉 MinIO setup completed!\n');
    console.log('📋 Summary:');
    console.log(`  ✅ ${successCount}/${buckets.length} buckets configured`);
    buckets.forEach(bucket => {
        console.log(`  - ${bucket.name} - ${bucket.description}`);
    });
    
    console.log('\n🔧 Next steps:');
    console.log('1. Test the backup system: npm run test:minio');
    console.log('2. Create your first backup: curl -X POST http://localhost:3005/api/backup-blog');
    console.log('3. Access MinIO Console: http://localhost:9001 (minioadmin/minioadmin)');
    
    console.log('\n📊 MinIO Console Access:');
    console.log(`URL: http://${process.env.MINIO_ENDPOINT || 'localhost'}:9001`);
    console.log(`Username: ${process.env.MINIO_ACCESS_KEY || 'minioadmin'}`);
    console.log(`Password: ${process.env.MINIO_SECRET_KEY || 'minioadmin'}`);
}

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    setupMinIO().catch(error => {
        console.error('❌ Setup failed:', error);
        process.exit(1);
    });
}

export default setupMinIO;