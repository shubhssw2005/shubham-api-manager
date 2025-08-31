import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import mongoose from 'mongoose';
import APIToken from '../../models/APIToken';
import User from '../../models/User';

describe('API Token Integration Tests', () => {
  let testUser;
  let testToken;

  beforeAll(async () => {
    // Connect to test database
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/test');
    }

    // Create a test user
    testUser = new User({
      name: 'Test User',
      email: 'test@example.com',
      password: 'hashedpassword',
      role: 'admin',
      status: 'approved'
    });
    await testUser.save();
  });

  afterAll(async () => {
    // Clean up test data
    if (testToken) {
      await APIToken.findByIdAndDelete(testToken._id);
    }
    if (testUser) {
      await User.findByIdAndDelete(testUser._id);
    }
  });

  it('should create an API token with proper validation', async () => {
    const tokenData = {
      name: 'Test Integration Token',
      token: APIToken.generateToken(),
      permissions: [
        { model: 'User', actions: ['read', 'create'] },
        { model: 'Media', actions: ['read'] }
      ],
      rateLimit: {
        requests: 500,
        window: 3600
      },
      description: 'Integration test token',
      createdBy: testUser._id
    };

    testToken = new APIToken(tokenData);
    await testToken.save();

    expect(testToken._id).toBeDefined();
    expect(testToken.name).toBe('Test Integration Token');
    expect(testToken.permissions).toHaveLength(2);
    expect(testToken.hashedToken).toBeDefined();
    expect(testToken.isValid).toBe(true);
  });

  it('should validate token permissions correctly', async () => {
    expect(testToken.hasPermission('User', 'read')).toBe(true);
    expect(testToken.hasPermission('User', 'create')).toBe(true);
    expect(testToken.hasPermission('User', 'delete')).toBe(false);
    expect(testToken.hasPermission('Media', 'read')).toBe(true);
    expect(testToken.hasPermission('Media', 'create')).toBe(false);
    expect(testToken.hasPermission('NonExistentModel', 'read')).toBe(false);
  });

  it('should find token by token value', async () => {
    const foundToken = await APIToken.findByToken(testToken.token);
    expect(foundToken).toBeTruthy();
    expect(foundToken._id.toString()).toBe(testToken._id.toString());
    expect(foundToken.name).toBe(testToken.name);
  });

  it('should record usage correctly', async () => {
    const initialRequests = testToken.usage.totalRequests;
    
    await testToken.recordUsage('127.0.0.1', 'Test User Agent');
    
    expect(testToken.usage.totalRequests).toBe(initialRequests + 1);
    expect(testToken.usage.lastUsed).toBeDefined();
    expect(testToken.usage.lastIP).toBe('127.0.0.1');
    expect(testToken.usage.lastUserAgent).toBe('Test User Agent');
  });

  it('should revoke token correctly', async () => {
    expect(testToken.isActive).toBe(true);
    
    await testToken.revoke();
    
    expect(testToken.isActive).toBe(false);
    expect(testToken.isValid).toBe(false);
  });

  it('should not find revoked token', async () => {
    const foundToken = await APIToken.findByToken(testToken.token);
    expect(foundToken).toBeNull();
  });

  it('should generate unique tokens', () => {
    const token1 = APIToken.generateToken();
    const token2 = APIToken.generateToken();
    
    expect(token1).not.toBe(token2);
    expect(token1).toHaveLength(64); // 32 bytes * 2 (hex)
    expect(token2).toHaveLength(64);
  });

  it('should hash tokens consistently', () => {
    const token = 'test-token-value';
    const hash1 = APIToken.hashToken(token);
    const hash2 = APIToken.hashToken(token);
    
    expect(hash1).toBe(hash2);
    expect(hash1).toHaveLength(64); // SHA-256 hex output
  });

  it('should validate expiration correctly', async () => {
    // Create token that expires in the past
    const expiredTokenData = {
      name: 'Expired Token',
      token: APIToken.generateToken(),
      permissions: [{ model: 'User', actions: ['read'] }],
      expiresAt: new Date(Date.now() - 1000), // 1 second ago
      createdBy: testUser._id
    };

    const expiredToken = new APIToken(expiredTokenData);
    await expiredToken.save();

    expect(expiredToken.isExpired).toBe(true);
    expect(expiredToken.isValid).toBe(false);

    // Clean up
    await APIToken.findByIdAndDelete(expiredToken._id);
  });

  it('should handle wildcard permissions', async () => {
    const wildcardTokenData = {
      name: 'Wildcard Token',
      token: APIToken.generateToken(),
      permissions: [
        { model: '*', actions: ['read'] }
      ],
      createdBy: testUser._id
    };

    const wildcardToken = new APIToken(wildcardTokenData);
    await wildcardToken.save();

    expect(wildcardToken.hasPermission('User', 'read')).toBe(true);
    expect(wildcardToken.hasPermission('Media', 'read')).toBe(true);
    expect(wildcardToken.hasPermission('AnyModel', 'read')).toBe(true);
    expect(wildcardToken.hasPermission('User', 'create')).toBe(false);

    // Clean up
    await APIToken.findByIdAndDelete(wildcardToken._id);
  });
});