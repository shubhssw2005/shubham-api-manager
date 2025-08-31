#!/usr/bin/env node

import { S3Client, ListBucketsCommand } from '@aws-sdk/client-s3';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * Simple AWS Connection Test
 */
async function testAWSConnection() {
    console.log('🔍 Testing AWS S3 Connection...\n');
    
    // Check if credentials are set
    if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'your-aws-access-key-here') {
        console.log('❌ AWS credentials not configured in .env file');
        console.log('\n📝 Please update your .env file with:');
        console.log('AWS_ACCESS_KEY_ID=your-actual-access-key');
        console.log('AWS_SECRET_ACCESS_KEY=your-actual-secret-key');
        console.log('AWS_REGION=us-east-1');
        console.log('\n💡 See S3_SETUP_STEPS.md for detailed instructions');
        return false;
    }
    
    console.log('✅ AWS credentials found in .env');
    console.log(`📍 Region: ${process.env.AWS_REGION || 'us-east-1'}`);
    console.log(`🔑 Access Key: ${process.env.AWS_ACCESS_KEY_ID.substring(0, 8)}...`);
    
    // Create S3 client
    const s3Client = new S3Client({
        region: process.env.AWS_REGION || 'us-east-1',
        credentials: {
            accessKeyId: process.env.AWS_ACCESS_KEY_ID,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
        }
    });
    
    try {
        console.log('\n🔄 Connecting to AWS S3...');
        
        // Test connection by listing buckets
        const response = await s3Client.send(new ListBucketsCommand({}));
        
        console.log('✅ Connection successful!');
        console.log(`📊 Found ${response.Buckets.length} existing buckets:`);
        
        if (response.Buckets.length > 0) {
            response.Buckets.forEach(bucket => {
                console.log(`  - ${bucket.Name} (created: ${bucket.CreationDate.toISOString().split('T')[0]})`);
            });
        } else {
            console.log('  (No existing buckets - we\'ll create them)');
        }
        
        console.log('\n🎉 AWS S3 is ready to use!');
        console.log('\n🔧 Next steps:');
        console.log('1. Run: npm run setup:s3');
        console.log('2. Test backup: npm run test:backup');
        
        return true;
        
    } catch (error) {
        console.log('❌ Connection failed!');
        console.log(`Error: ${error.message}`);
        
        if (error.name === 'InvalidAccessKeyId') {
            console.log('\n💡 Fix: Check your AWS_ACCESS_KEY_ID');
        } else if (error.name === 'SignatureDoesNotMatch') {
            console.log('\n💡 Fix: Check your AWS_SECRET_ACCESS_KEY');
        } else if (error.name === 'TokenRefreshRequired') {
            console.log('\n💡 Fix: Your credentials may have expired');
        } else {
            console.log('\n💡 Fix: Verify your AWS credentials and permissions');
        }
        
        console.log('\n📚 See AWS_SETUP_GUIDE.md for troubleshooting');
        return false;
    }
}

// Run test if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testAWSConnection().then(success => {
        process.exit(success ? 0 : 1);
    });
}

export default testAWSConnection;