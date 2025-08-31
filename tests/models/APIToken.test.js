import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import mongoose from 'mongoose';
import APIToken from '../../models/APIToken.js';
import User from '../../models/User.js';

describe('APIToken Model', () => {
  let testUser;

  beforeEach(async () => {
    // Connect to test database
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/test');
    }
    
    // Clear collections
    await APIToken.deleteMany({});
    await User.deleteMany({});

    // Create test user
    testUser = await User.create({
      email: 'test@example.com',
      password: 'password123',
      name: 'Test User',
      role: 'admin',
      status: 'approved'
    });
  });

  afterEach(async () => {
    // Clean up
    await APIToken.deleteMany({});
    await User.deleteMany({});
  });

  it('should create an API token with valid data', async () => {
    const tokenValue = APIToken.generateToken();
    
    const tokenData = {
      name: 'Test Token',
      token: tokenValue,
      permissions: [
        {
          model: 'User',
          actions: ['read', 'create']
        }
      ],
      createdBy: testUser._id
    };

    const apiToken = new APIToken(tokenData);
    await apiToken.save();

    expect(apiToken._id).toBeDefined();
    expect(apiToken.name).toBe('Test Token');
    expect(apiToken.hashedToken).toBeDefined();
    expect(apiToken.isValid).toBe(true);
  });

  it('should generate and hash tokens correctly', () => {
    const token1 = APIToken.generateToken();
    const token2 = APIToken.generateToken();
    
    expect(token1).not.toBe(token2);
    expect(token1).toHaveLength(64); // 32 bytes * 2 (hex)
    
    const hash1 = APIToken.hashToken(token1);
    const hash2 = APIToken.hashToken(token1);
    
    expect(hash1).toBe(hash2); // Same token should produce same hash
    expect(hash1).toHaveLength(64); // SHA256 hex length
  });

  it('should find token by token value', async () => {
    const tokenValue = APIToken.generateToken();
    
    const apiToken = await APIToken.create({
      name: 'Test Token',
      token: tokenValue,
      permissions: [{ model: 'User', actions: ['read'] }],
      createdBy: testUser._id
    });

    const foundToken = await APIToken.findByToken(tokenValue);
    expect(foundToken).toBeDefined();
    expect(foundToken._id.toString()).toBe(apiToken._id.toString());
  });

  it('should check permissions correctly', async () => {
    const tokenValue = APIToken.generateToken();
    
    const apiToken = await APIToken.create({
      name: 'Test Token',
      token: tokenValue,
      permissions: [
        {
          model: 'User',
          actions: ['read', 'create']
        },
        {
          model: '*',
          actions: ['read']
        }
      ],
      createdBy: testUser._id
    });

    expect(apiToken.hasPermission('User', 'read')).toBe(true);
    expect(apiToken.hasPermission('User', 'create')).toBe(true);
    expect(apiToken.hasPermission('User', 'delete')).toBe(false);
    expect(apiToken.hasPermission('Post', 'read')).toBe(true); // Wildcard permission
    expect(apiToken.hasPermission('Post', 'create')).toBe(false);
  });

  it('should handle token expiration', async () => {
    const tokenValue = APIToken.generateToken();
    const pastDate = new Date(Date.now() - 1000); // 1 second ago
    
    const apiToken = await APIToken.create({
      name: 'Expired Token',
      token: tokenValue,
      permissions: [{ model: 'User', actions: ['read'] }],
      expiresAt: pastDate,
      createdBy: testUser._id
    });

    expect(apiToken.isExpired).toBe(true);
    expect(apiToken.isValid).toBe(false);

    const foundToken = await APIToken.findByToken(tokenValue);
    expect(foundToken).toBeNull(); // Should not find expired tokens
  });

  it('should record token usage', async () => {
    const tokenValue = APIToken.generateToken();
    
    const apiToken = await APIToken.create({
      name: 'Test Token',
      token: tokenValue,
      permissions: [{ model: 'User', actions: ['read'] }],
      createdBy: testUser._id
    });

    expect(apiToken.usage.totalRequests).toBe(0);
    
    await apiToken.recordUsage('127.0.0.1', 'Test User Agent');
    
    expect(apiToken.usage.totalRequests).toBe(1);
    expect(apiToken.usage.lastIP).toBe('127.0.0.1');
    expect(apiToken.usage.lastUserAgent).toBe('Test User Agent');
    expect(apiToken.usage.lastUsed).toBeDefined();
  });

  it('should revoke tokens', async () => {
    const tokenValue = APIToken.generateToken();
    
    const apiToken = await APIToken.create({
      name: 'Test Token',
      token: tokenValue,
      permissions: [{ model: 'User', actions: ['read'] }],
      createdBy: testUser._id
    });

    expect(apiToken.isActive).toBe(true);
    
    await apiToken.revoke();
    
    expect(apiToken.isActive).toBe(false);
    expect(apiToken.isValid).toBe(false);
  });
});