#!/usr/bin/env node

import MinIOBackupService from '../services/MinIOBackupService.js';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * MinIO Connection Test
 */
async function testMinIO() {
    console.log('🔍 Testing MinIO Backup Service...\n');
    
    try {
        const minioService = new MinIOBackupService();
        
        // Test connection
        const connectionResult = await minioService.testConnection();
        
        if (connectionResult.success) {
            console.log(`\n🌐 MinIO Endpoint: ${connectionResult.endpoint}`);
            console.log(`📊 Available Buckets: ${connectionResult.buckets}`);
            
            // Test bucket creation
            console.log('\n🪣 Testing bucket operations...');
            await minioService.ensureBucketExists();
            
            console.log('\n✅ MinIO is ready for backups!');
            console.log('\n🔧 You can now:');
            console.log('1. Create backups via API');
            console.log('2. View files in MinIO Console');
            console.log('3. Download backups with presigned URLs');
            
            return true;
        }
        
    } catch (error) {
        console.error('❌ MinIO test failed:', error.message);
        
        if (error.code === 'ECONNREFUSED') {
            console.log('\n🐳 Start MinIO with Docker:');
            console.log('docker run -d --name minio -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"');
        }
        
        return false;
    }
}

// Run test if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testMinIO().then(success => {
        process.exit(success ? 0 : 1);
    });
}

export default testMinIO;