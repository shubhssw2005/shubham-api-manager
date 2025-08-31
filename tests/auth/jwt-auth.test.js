import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import TokenService from '../../lib/auth/TokenService.js';
import PermissionService from '../../lib/auth/PermissionService.js';
import { jwtAuth } from '../../middleware/jwtAuth.js';
import { createMocks } from 'node-mocks-http';
import Redis from 'ioredis';

// Mock Redis for testing
vi.mock('ioredis', () => {
  const mockRedis = {
    setex: vi.fn().mockResolvedValue('OK'),
    get: vi.fn().mockResolvedValue(null),
    del: vi.fn().mockResolvedValue(1),
    sadd: vi.fn().mockResolvedValue(1),
    srem: vi.fn().mockResolvedValue(1),
    smembers: vi.fn().mockResolvedValue([]),
    expire: vi.fn().mockResolvedValue(1),
    exists: vi.fn().mockResolvedValue(0),
    incr: vi.fn().mockResolvedValue(1),
    pipeline: vi.fn().mockReturnValue({
      del: vi.fn().mockReturnThis(),
      exec: vi.fn().mockResolvedValue([])
    })
  };
  
  return {
    default: vi.fn(() => mockRedis)
  };
});

// Mock database connection
vi.mock('../../lib/dbConnect.js', () => ({
  default: vi.fn().mockResolvedValue(true)
}));

// Mock Role model
vi.mock('../../models/Role.js', () => ({
  Role: {
    findOne: vi.fn().mockResolvedValue({
      name: 'user',
      permission: {
        create: true,
        read: true,
        update: true,
        delete: false
      },
      routes: ['/manager', '/media']
    })
  }
}));

// Mock User model
vi.mock('../../models/User.js', () => ({
  default: {
    findById: vi.fn().mockReturnValue({
      select: vi.fn().mockResolvedValue({
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        name: 'Test User',
        role: 'user',
        status: 'active',
        comparePassword: vi.fn().mockResolvedValue(true),
        save: vi.fn().mockResolvedValue(true),
        toObject: vi.fn().mockReturnValue({
          _id: '507f1f77bcf86cd799439011',
          email: 'test@example.com',
          name: 'Test User',
          role: 'user',
          status: 'active'
        })
      })
    }),
    findOne: vi.fn().mockResolvedValue({
      _id: '507f1f77bcf86cd799439011',
      email: 'test@example.com',
      name: 'Test User',
      role: 'user',
      status: 'active',
      comparePassword: vi.fn().mockResolvedValue(true),
      save: vi.fn().mockResolvedValue(true),
      toObject: vi.fn().mockReturnValue({
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        name: 'Test User',
        role: 'user',
        status: 'active'
      })
    })
  }
}));

describe('JWT Authentication and Authorization Service', () => {
  let tokenService;
  let permissionService;
  let mockRedis;

  beforeAll(() => {
    tokenService = new TokenService();
    permissionService = new PermissionService();
    mockRedis = tokenService.redis; // Get the mocked Redis instance
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('TokenService', () => {
    it('should generate token pair successfully', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      const tokenPair = await tokenService.generateTokenPair(user);

      expect(tokenPair).toHaveProperty('accessToken');
      expect(tokenPair).toHaveProperty('refreshToken');
      expect(tokenPair).toHaveProperty('expiresIn');
      expect(tokenPair).toHaveProperty('tokenType', 'Bearer');
      expect(mockRedis.setex).toHaveBeenCalled();
      expect(mockRedis.sadd).toHaveBeenCalled();
    });

    it('should verify access token successfully', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      const tokenPair = await tokenService.generateTokenPair(user);
      const decoded = await tokenService.verifyAccessToken(tokenPair.accessToken);

      expect(decoded).toHaveProperty('userId', user._id);
      expect(decoded).toHaveProperty('email', user.email);
      expect(decoded).toHaveProperty('role', user.role);
      expect(decoded).toHaveProperty('type', 'access');
    });

    it('should verify refresh token successfully', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      // Mock Redis to return stored token data
      mockRedis.get.mockResolvedValueOnce(JSON.stringify({
        userId: user._id,
        tokenId: 'test-token-id',
        createdAt: new Date().toISOString()
      }));

      const tokenPair = await tokenService.generateTokenPair(user);
      const decoded = await tokenService.verifyRefreshToken(tokenPair.refreshToken);

      expect(decoded).toHaveProperty('userId', user._id);
      expect(decoded).toHaveProperty('type', 'refresh');
      expect(decoded).toHaveProperty('sessionData');
    });

    it('should refresh access token successfully', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      // Mock Redis to return stored token data
      mockRedis.get.mockResolvedValueOnce(JSON.stringify({
        userId: user._id,
        tokenId: 'test-token-id',
        createdAt: new Date().toISOString()
      }));

      const tokenPair = await tokenService.generateTokenPair(user);
      const newToken = await tokenService.refreshAccessToken(tokenPair.refreshToken, user);

      expect(newToken).toHaveProperty('accessToken');
      expect(newToken).toHaveProperty('expiresIn');
      expect(newToken).toHaveProperty('tokenType', 'Bearer');
    });

    it('should revoke refresh token successfully', async () => {
      const userId = '507f1f77bcf86cd799439011';
      const tokenId = 'test-token-id';

      const result = await tokenService.revokeRefreshToken(userId, tokenId);

      expect(result).toBe(true);
      expect(mockRedis.del).toHaveBeenCalled();
      expect(mockRedis.srem).toHaveBeenCalled();
    });

    it('should blacklist access token successfully', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      const tokenPair = await tokenService.generateTokenPair(user);
      const result = await tokenService.blacklistAccessToken(tokenPair.accessToken);

      expect(result).toBe(true);
      expect(mockRedis.setex).toHaveBeenCalled();
    });
  });

  describe('PermissionService', () => {
    it('should check user permissions correctly', async () => {
      const user = {
        id: '507f1f77bcf86cd799439011',
        role: 'user'
      };

      const hasPermission = await permissionService.hasPermission(user, 'content.read');
      expect(typeof hasPermission).toBe('boolean');
    });

    it('should grant all permissions to superadmin', async () => {
      const superadmin = {
        id: '507f1f77bcf86cd799439011',
        role: 'superadmin'
      };

      const hasPermission = await permissionService.hasPermission(superadmin, 'any.permission');
      expect(hasPermission).toBe(true);
    });

    it('should check multiple permissions correctly', async () => {
      const user = {
        id: '507f1f77bcf86cd799439011',
        role: 'user'
      };

      const permissions = ['content.read', 'content.create'];
      const hasAny = await permissionService.hasAnyPermission(user, permissions);
      const hasAll = await permissionService.hasAllPermissions(user, permissions);

      expect(typeof hasAny).toBe('boolean');
      expect(typeof hasAll).toBe('boolean');
    });

    it('should get user permissions correctly', async () => {
      const user = {
        id: '507f1f77bcf86cd799439011',
        role: 'user'
      };

      const permissions = await permissionService.getUserPermissions(user);
      expect(Array.isArray(permissions)).toBe(true);
    });

    it('should create permission middleware correctly', async () => {
      const middleware = permissionService.requirePermission('content.read');
      expect(typeof middleware).toBe('function');
    });
  });

  describe('JWT Middleware', () => {
    it('should authenticate valid token', async () => {
      const user = {
        _id: '507f1f77bcf86cd799439011',
        email: 'test@example.com',
        role: 'user'
      };

      const tokenPair = await tokenService.generateTokenPair(user);
      
      const { req, res } = createMocks({
        method: 'GET',
        headers: {
          authorization: `Bearer ${tokenPair.accessToken}`
        }
      });

      const next = vi.fn();

      await jwtAuth(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.user).toBeDefined();
      expect(req.user.email).toBe(user.email);
    });

    it('should reject invalid token', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        headers: {
          authorization: 'Bearer invalid-token'
        }
      });

      const next = vi.fn();

      await jwtAuth(req, res, next);

      expect(next).not.toHaveBeenCalled();
      expect(res._getStatusCode()).toBe(401);
    });

    it('should reject missing token', async () => {
      const { req, res } = createMocks({
        method: 'GET'
      });

      const next = vi.fn();

      await jwtAuth(req, res, next);

      expect(next).not.toHaveBeenCalled();
      expect(res._getStatusCode()).toBe(401);
    });
  });
});