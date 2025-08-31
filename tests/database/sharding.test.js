import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import ShardManager from '../../lib/database/ShardManager.js';
import MigrationManager from '../../lib/database/MigrationManager.js';
import DatabaseManager from '../../lib/database/DatabaseManager.js';
import TenantModelFactory from '../../lib/database/TenantModelFactory.js';
import { getConfig } from '../../lib/database/config.js';
import mongoose from 'mongoose';

describe('Database Sharding System', () => {
  let shardManager;
  let migrationManager;
  let databaseManager;
  let testTenantIds;

  beforeAll(async () => {
    // Use test configuration
    process.env.NODE_ENV = 'test';
    
    const config = getConfig();
    
    // Initialize shard manager
    shardManager = new ShardManager(config.sharding);
    await shardManager.initialize(config.shards);
    
    // Initialize migration manager
    migrationManager = new MigrationManager(shardManager, config.migrations);
    await migrationManager.initialize();
    
    // Initialize database manager
    databaseManager = new DatabaseManager(config);
    await databaseManager.initialize(config.shards);
    
    // Test tenant IDs
    testTenantIds = ['tenant_1', 'tenant_2', 'tenant_3', 'tenant_4', 'tenant_5'];
  });

  afterAll(async () => {
    // Clean up test data
    for (const tenantId of testTenantIds) {
      try {
        const connection = await databaseManager.getConnection(tenantId);
        await connection.db.dropDatabase();
      } catch (error) {
        console.warn(`Failed to cleanup tenant ${tenantId}:`, error.message);
      }
    }
    
    // Shutdown managers
    if (databaseManager) {
      await databaseManager.shutdown();
    }
    if (shardManager) {
      await shardManager.close();
    }
  });

  describe('ShardManager', () => {
    it('should initialize with correct number of shards', () => {
      expect(shardManager.initialized).toBe(true);
      expect(shardManager.shards.size).toBeGreaterThan(0);
    });

    it('should assign tenants to shards consistently', () => {
      const assignments = new Map();
      
      // Test consistent assignment
      for (let i = 0; i < 10; i++) {
        for (const tenantId of testTenantIds) {
          const shardId = shardManager.getShardForTenant(tenantId);
          
          if (assignments.has(tenantId)) {
            expect(shardId).toBe(assignments.get(tenantId));
          } else {
            assignments.set(tenantId, shardId);
          }
        }
      }
      
      expect(assignments.size).toBe(testTenantIds.length);
    });

    it('should provide database connections for tenants', async () => {
      for (const tenantId of testTenantIds) {
        const connection = shardManager.getConnectionForTenant(tenantId);
        expect(connection).toBeDefined();
        expect(connection.readyState).toBe(1); // Connected
      }
    });

    it('should perform health checks on all shards', async () => {
      const healthResults = await shardManager.healthCheck();
      
      expect(healthResults.size).toBe(shardManager.shards.size);
      
      for (const [shardId, result] of healthResults) {
        expect(result.status).toBe('healthy');
        expect(result.latency).toBeGreaterThan(0);
        expect(result.lastCheck).toBeInstanceOf(Number);
      }
    });

    it('should provide cluster statistics', async () => {
      const stats = await shardManager.getClusterStats();
      
      expect(stats.totalShards).toBe(shardManager.shards.size);
      expect(stats.totalTenants).toBeGreaterThanOrEqual(testTenantIds.length);
      expect(stats.shardDistribution).toBeInstanceOf(Map);
      expect(stats.healthStatus).toBeInstanceOf(Map);
    });
  });

  describe('MigrationManager', () => {
    it('should load available migrations', async () => {
      const migrations = await migrationManager.getAvailableMigrations();
      expect(Array.isArray(migrations)).toBe(true);
      expect(migrations.length).toBeGreaterThan(0);
      
      // Check migration structure
      for (const migration of migrations) {
        expect(migration.version).toBeDefined();
        expect(migration.filename).toBeDefined();
        expect(migration.path).toBeDefined();
      }
    });

    it('should identify pending migrations', async () => {
      const shardIds = Array.from(shardManager.shards.keys());
      
      for (const shardId of shardIds) {
        const pendingMigrations = await migrationManager.getPendingMigrations(shardId);
        expect(Array.isArray(pendingMigrations)).toBe(true);
      }
    });

    it('should run migrations on a single shard', async () => {
      const shardIds = Array.from(shardManager.shards.keys());
      const testShardId = shardIds[0];
      
      const result = await migrationManager.runMigrations(testShardId);
      
      expect(result.success).toBe(true);
      expect(result.migrationsRun).toBeGreaterThanOrEqual(0);
    });

    it('should provide migration status', async () => {
      const status = await migrationManager.getMigrationStatus();
      
      expect(status).toBeInstanceOf(Map);
      expect(status.size).toBe(shardManager.shards.size);
      
      for (const [shardId, shardStatus] of status) {
        expect(shardStatus.totalMigrations).toBeGreaterThanOrEqual(0);
        expect(shardStatus.completedMigrations).toBeGreaterThanOrEqual(0);
        expect(shardStatus.pendingMigrations).toBeGreaterThanOrEqual(0);
        expect(typeof shardStatus.isRunning).toBe('boolean');
      }
    });
  });

  describe('DatabaseManager', () => {
    it('should execute queries for tenants', async () => {
      const tenantId = testTenantIds[0];
      
      const result = await databaseManager.executeQuery(
        tenantId,
        (db) => db.admin().ping(),
        'primary'
      );
      
      expect(result.ok).toBe(1);
    });

    it('should handle query timeouts', async () => {
      const tenantId = testTenantIds[0];
      
      // Set a very short timeout for this test
      const originalTimeout = databaseManager.queryTimeout;
      databaseManager.queryTimeout = 1; // 1ms
      
      try {
        await expect(
          databaseManager.executeQuery(
            tenantId,
            (db) => new Promise(resolve => setTimeout(resolve, 100)),
            'primary'
          )
        ).rejects.toThrow('Query timeout');
      } finally {
        databaseManager.queryTimeout = originalTimeout;
      }
    });

    it('should provide shard information for tenants', () => {
      for (const tenantId of testTenantIds) {
        const shardInfo = databaseManager.getShardInfo(tenantId);
        
        expect(shardInfo.shardId).toBeDefined();
        expect(shardInfo.region).toBeDefined();
        expect(shardInfo.status).toBeDefined();
      }
    });

    it('should create tenant models', async () => {
      const tenantId = testTenantIds[0];
      const schema = new mongoose.Schema({
        name: String,
        value: Number
      });
      
      const TenantModel = databaseManager.createTenantModel(
        tenantId,
        'TestModel',
        schema
      );
      
      expect(TenantModel).toBeDefined();
      expect(typeof TenantModel.find).toBe('function');
      expect(typeof TenantModel.insertOne).toBe('function');
    });
  });

  describe('TenantModelFactory', () => {
    beforeEach(() => {
      // Clear caches before each test
      TenantModelFactory.clearAllCaches();
    });

    it('should register schemas', () => {
      const schema = new mongoose.Schema({
        title: String,
        content: String
      });
      
      TenantModelFactory.registerSchema('TestPost', schema, {
        tenantIndexes: [
          { tenantId: 1, title: 1 }
        ]
      });
      
      expect(TenantModelFactory.schemaCache.has('TestPost')).toBe(true);
    });

    it('should create tenant-specific models', async () => {
      const schema = new mongoose.Schema({
        title: String,
        content: String
      });
      
      TenantModelFactory.registerSchema('TestPost', schema);
      
      const tenantId = testTenantIds[0];
      const TestPost = await TenantModelFactory.getTenantModel(tenantId, 'TestPost');
      
      expect(TestPost).toBeDefined();
      expect(TestPost.tenantId).toBe(tenantId);
      expect(typeof TestPost.find).toBe('function');
      expect(typeof TestPost.create).toBe('function');
    });

    it('should cache tenant models', async () => {
      const schema = new mongoose.Schema({
        name: String
      });
      
      TenantModelFactory.registerSchema('CachedModel', schema);
      
      const tenantId = testTenantIds[0];
      const Model1 = await TenantModelFactory.getTenantModel(tenantId, 'CachedModel');
      const Model2 = await TenantModelFactory.getTenantModel(tenantId, 'CachedModel');
      
      expect(Model1).toBe(Model2); // Should be the same cached instance
    });

    it('should provide cache statistics', async () => {
      const schema = new mongoose.Schema({ name: String });
      TenantModelFactory.registerSchema('StatsModel', schema);
      
      await TenantModelFactory.getTenantModel(testTenantIds[0], 'StatsModel');
      await TenantModelFactory.getTenantModel(testTenantIds[1], 'StatsModel');
      
      const stats = TenantModelFactory.getCacheStats();
      
      expect(stats.registeredSchemas).toBeGreaterThan(0);
      expect(stats.cachedTenants).toBeGreaterThan(0);
      expect(stats.totalCachedModels).toBeGreaterThan(0);
    });
  });

  describe('Integration Tests', () => {
    it('should handle CRUD operations across shards', async () => {
      const schema = new mongoose.Schema({
        name: String,
        value: Number,
        createdAt: { type: Date, default: Date.now }
      });
      
      TenantModelFactory.registerSchema('IntegrationTest', schema);
      
      const testData = [];
      
      // Create test documents for different tenants
      for (const tenantId of testTenantIds) {
        const Model = await TenantModelFactory.getTenantModel(tenantId, 'IntegrationTest');
        
        const doc = await Model.create({
          name: `Test Document for ${tenantId}`,
          value: Math.floor(Math.random() * 100)
        });
        
        testData.push({ tenantId, docId: doc._id, model: Model });
      }
      
      // Verify documents can be found
      for (const { tenantId, docId, model } of testData) {
        const foundDoc = await model.findById(docId);
        expect(foundDoc).toBeDefined();
        expect(foundDoc.tenantId).toBe(tenantId);
      }
      
      // Update documents
      for (const { model, docId } of testData) {
        await model.updateOne(
          { _id: docId },
          { $set: { value: 999 } }
        );
      }
      
      // Verify updates
      for (const { model, docId } of testData) {
        const updatedDoc = await model.findById(docId);
        expect(updatedDoc.value).toBe(999);
      }
      
      // Clean up
      for (const { model } of testData) {
        await model.deleteMany({});
      }
    });

    it('should maintain tenant isolation', async () => {
      const schema = new mongoose.Schema({
        secret: String
      });
      
      TenantModelFactory.registerSchema('IsolationTest', schema);
      
      const tenant1Id = testTenantIds[0];
      const tenant2Id = testTenantIds[1];
      
      const Model1 = await TenantModelFactory.getTenantModel(tenant1Id, 'IsolationTest');
      const Model2 = await TenantModelFactory.getTenantModel(tenant2Id, 'IsolationTest');
      
      // Create documents for each tenant
      await Model1.create({ secret: 'tenant1-secret' });
      await Model2.create({ secret: 'tenant2-secret' });
      
      // Verify tenant isolation
      const tenant1Docs = await Model1.find({});
      const tenant2Docs = await Model2.find({});
      
      expect(tenant1Docs.length).toBe(1);
      expect(tenant2Docs.length).toBe(1);
      expect(tenant1Docs[0].secret).toBe('tenant1-secret');
      expect(tenant2Docs[0].secret).toBe('tenant2-secret');
      expect(tenant1Docs[0].tenantId).toBe(tenant1Id);
      expect(tenant2Docs[0].tenantId).toBe(tenant2Id);
      
      // Clean up
      await Model1.deleteMany({});
      await Model2.deleteMany({});
    });

    it('should handle high concurrency operations', async () => {
      const schema = new mongoose.Schema({
        counter: Number,
        timestamp: { type: Date, default: Date.now }
      });
      
      TenantModelFactory.registerSchema('ConcurrencyTest', schema);
      
      const tenantId = testTenantIds[0];
      const Model = await TenantModelFactory.getTenantModel(tenantId, 'ConcurrencyTest');
      
      // Create multiple concurrent operations
      const concurrentOps = [];
      const numOps = 50;
      
      for (let i = 0; i < numOps; i++) {
        concurrentOps.push(
          Model.create({ counter: i })
        );
      }
      
      const results = await Promise.all(concurrentOps);
      
      expect(results.length).toBe(numOps);
      
      // Verify all documents were created
      const allDocs = await Model.find({}).sort({ counter: 1 });
      expect(allDocs.length).toBe(numOps);
      
      for (let i = 0; i < numOps; i++) {
        expect(allDocs[i].counter).toBe(i);
        expect(allDocs[i].tenantId).toBe(tenantId);
      }
      
      // Clean up
      await Model.deleteMany({});
    });
  });
});