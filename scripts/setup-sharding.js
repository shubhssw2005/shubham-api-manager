#!/usr/bin/env node

/**
 * Setup script for database sharding system
 * Demonstrates initialization and basic operations
 */

import { getShardedDatabase, runMigrations, getClusterStats } from '../lib/dbConnect.js';
import TenantModelFactory from '../lib/database/TenantModelFactory.js';
import mongoose from 'mongoose';

async function setupSharding() {
  console.log('🚀 Setting up database sharding system...\n');

  try {
    // Initialize the sharded database
    console.log('1. Initializing sharded database...');
    const dbManager = await getShardedDatabase();
    console.log('✅ Database manager initialized\n');

    // Run migrations
    console.log('2. Running database migrations...');
    const migrationResults = await runMigrations();
    
    let totalMigrations = 0;
    let successfulShards = 0;
    
    for (const [shardId, result] of migrationResults) {
      if (result.success) {
        successfulShards++;
        totalMigrations += result.migrationsRun || 0;
        console.log(`   ✅ Shard ${shardId}: ${result.migrationsRun || 0} migrations completed`);
      } else {
        console.log(`   ❌ Shard ${shardId}: Migration failed - ${result.error}`);
      }
    }
    
    console.log(`✅ Migrations completed: ${totalMigrations} total across ${successfulShards} shards\n`);

    // Register sample schemas
    console.log('3. Registering sample schemas...');
    registerSampleSchemas();
    console.log('✅ Sample schemas registered\n');

    // Demonstrate tenant operations
    console.log('4. Demonstrating tenant operations...');
    await demonstrateTenantOperations();
    console.log('✅ Tenant operations completed\n');

    // Show cluster statistics
    console.log('5. Cluster statistics:');
    const stats = await getClusterStats();
    console.log(`   📊 Total shards: ${stats.totalShards}`);
    console.log(`   👥 Total tenants: ${stats.totalTenants}`);
    
    console.log('   📈 Tenant distribution:');
    for (const [shardId, tenantCount] of stats.shardDistribution) {
      console.log(`      ${shardId}: ${tenantCount} tenants`);
    }
    
    console.log('   🏥 Health status:');
    for (const [shardId, health] of stats.healthStatus) {
      const status = health.status === 'healthy' ? '✅' : '❌';
      console.log(`      ${shardId}: ${status} ${health.status} (${health.latency || 0}ms)`);
    }

    console.log('\n🎉 Sharding setup completed successfully!');
    console.log('\n📝 Next steps:');
    console.log('   - Update your models to use TenantModelFactory');
    console.log('   - Configure environment variables for production shards');
    console.log('   - Set up monitoring and alerting');
    console.log('   - Test failover scenarios');

  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

function registerSampleSchemas() {
  // User schema
  const userSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true },
    name: { type: String, required: true },
    role: { type: String, enum: ['user', 'admin'], default: 'user' },
    status: { type: String, enum: ['active', 'inactive'], default: 'active' },
    lastLoginAt: Date,
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now }
  });

  TenantModelFactory.registerSchema('User', userSchema, {
    tenantIndexes: [
      { tenantId: 1, email: 1 },
      { tenantId: 1, role: 1 },
      { tenantId: 1, status: 1 }
    ]
  });

  // Post schema
  const postSchema = new mongoose.Schema({
    title: { type: String, required: true },
    slug: { type: String, required: true },
    content: { type: String, required: true },
    status: { type: String, enum: ['draft', 'published', 'archived'], default: 'draft' },
    authorId: { type: mongoose.Schema.Types.ObjectId, required: true },
    publishedAt: Date,
    tags: [String],
    viewCount: { type: Number, default: 0 },
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now }
  });

  TenantModelFactory.registerSchema('Post', postSchema, {
    tenantIndexes: [
      { tenantId: 1, status: 1, publishedAt: -1 },
      { tenantId: 1, authorId: 1 },
      { tenantId: 1, tags: 1 },
      { tenantId: 1, slug: 1 }
    ]
  });

  // Media schema
  const mediaSchema = new mongoose.Schema({
    filename: { type: String, required: true },
    originalName: { type: String, required: true },
    mimeType: { type: String, required: true },
    size: { type: Number, required: true },
    path: { type: String, required: true },
    url: { type: String, required: true },
    storageProvider: { type: String, enum: ['local', 's3'], default: 'local' },
    metadata: {
      width: Number,
      height: Number,
      duration: Number
    },
    tags: [String],
    uploadedBy: { type: mongoose.Schema.Types.ObjectId, required: true },
    processingStatus: { type: String, enum: ['pending', 'processing', 'completed', 'failed'], default: 'pending' },
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now }
  });

  TenantModelFactory.registerSchema('Media', mediaSchema, {
    tenantIndexes: [
      { tenantId: 1, mimeType: 1 },
      { tenantId: 1, uploadedBy: 1 },
      { tenantId: 1, processingStatus: 1 },
      { tenantId: 1, tags: 1 }
    ]
  });

  console.log('   📋 Registered schemas: User, Post, Media');
}

async function demonstrateTenantOperations() {
  const sampleTenants = ['tenant_demo_1', 'tenant_demo_2', 'tenant_demo_3'];
  
  for (const tenantId of sampleTenants) {
    console.log(`   👤 Working with ${tenantId}...`);
    
    // Get tenant models
    const User = await TenantModelFactory.getTenantModel(tenantId, 'User');
    const Post = await TenantModelFactory.getTenantModel(tenantId, 'Post');
    const Media = await TenantModelFactory.getTenantModel(tenantId, 'Media');
    
    // Create sample user
    const user = await User.create({
      email: `admin@${tenantId}.com`,
      name: `Admin User for ${tenantId}`,
      role: 'admin'
    });
    
    // Create sample post
    const post = await Post.create({
      title: `Welcome to ${tenantId}`,
      slug: `welcome-to-${tenantId}`,
      content: 'This is a sample post to demonstrate sharding.',
      status: 'published',
      authorId: user._id,
      publishedAt: new Date(),
      tags: ['welcome', 'demo']
    });
    
    // Create sample media
    const media = await Media.create({
      filename: `logo-${tenantId}.png`,
      originalName: 'logo.png',
      mimeType: 'image/png',
      size: 12345,
      path: `/uploads/${tenantId}/logo.png`,
      url: `https://cdn.example.com/${tenantId}/logo.png`,
      uploadedBy: user._id,
      processingStatus: 'completed',
      metadata: { width: 200, height: 100 }
    });
    
    // Verify tenant isolation
    const userCount = await User.countDocuments({});
    const postCount = await Post.countDocuments({});
    const mediaCount = await Media.countDocuments({});
    
    console.log(`      📊 Created: ${userCount} users, ${postCount} posts, ${mediaCount} media files`);
    
    // Get tenant statistics
    const userStats = await User.getTenantStats();
    console.log(`      📈 User stats: ${userStats.totalDocuments} documents, avg size: ${Math.round(userStats.avgSize || 0)} bytes`);
  }
  
  // Demonstrate cross-tenant isolation
  console.log('   🔒 Verifying tenant isolation...');
  
  const User1 = await TenantModelFactory.getTenantModel(sampleTenants[0], 'User');
  const User2 = await TenantModelFactory.getTenantModel(sampleTenants[1], 'User');
  
  const tenant1Users = await User1.find({});
  const tenant2Users = await User2.find({});
  
  console.log(`      ${sampleTenants[0]}: ${tenant1Users.length} users`);
  console.log(`      ${sampleTenants[1]}: ${tenant2Users.length} users`);
  
  // Verify users belong to correct tenants
  for (const user of tenant1Users) {
    if (user.tenantId !== sampleTenants[0]) {
      throw new Error('Tenant isolation violation detected!');
    }
  }
  
  for (const user of tenant2Users) {
    if (user.tenantId !== sampleTenants[1]) {
      throw new Error('Tenant isolation violation detected!');
    }
  }
  
  console.log('      ✅ Tenant isolation verified');
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down gracefully...');
  
  try {
    const { shutdownDatabase } = await import('../lib/dbConnect.js');
    await shutdownDatabase();
    console.log('✅ Database connections closed');
  } catch (error) {
    console.error('❌ Error during shutdown:', error.message);
  }
  
  process.exit(0);
});

// Run the setup
if (import.meta.url === `file://${process.argv[1]}`) {
  setupSharding().catch(error => {
    console.error('❌ Unhandled error:', error);
    process.exit(1);
  });
}

export default setupSharding;