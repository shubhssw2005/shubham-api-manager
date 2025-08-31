import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createMocks } from 'node-mocks-http';
import loginHandler from '../../pages/api/auth/login.js';
import tokenHandler from '../../pages/api/auth/token.js';
import permissionsHandler from '../../pages/api/auth/permissions.js';

// Mock database and Redis
vi.mock('../../lib/dbConnect.js', () => ({
  default: vi.fn().mockResolvedValue(true)
}));

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
        lastLoginAt: new Date(),
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
      lastLoginAt: new Date(),
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

describe('Authentication Endpoints Integration', () => {
  describe('POST /api/auth/login', () => {
    it('should login successfully with valid credentials', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: 'password123'
        },
        headers: {
          'user-agent': 'test-agent',
          'x-forwarded-for': '127.0.0.1'
        }
      });
      
      // Mock connection object
      req.connection = { remoteAddress: '127.0.0.1' };

      await loginHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('accessToken');
      expect(data.data).toHaveProperty('refreshToken');
      expect(data.data.user).toHaveProperty('email', 'test@example.com');
    });

    it('should reject invalid credentials', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: 'wrongpassword'
        }
      });

      // Mock password comparison to fail
      const User = await import('../../models/User.js');
      User.default.findOne.mockResolvedValueOnce({
        comparePassword: vi.fn().mockResolvedValue(false)
      });

      await loginHandler(req, res);

      expect(res._getStatusCode()).toBe(401);
      const data = JSON.parse(res._getData());
      expect(data.code).toBe('INVALID_CREDENTIALS');
    });

    it('should reject missing credentials', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          email: 'test@example.com'
          // missing password
        }
      });

      await loginHandler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.code).toBe('MISSING_CREDENTIALS');
    });
  });

  describe('POST /api/auth/permissions', () => {
    it('should check user permissions', async () => {
      // First login to get a token
      const { req: loginReq, res: loginRes } = createMocks({
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: 'password123'
        },
        headers: {
          'user-agent': 'test-agent',
          'x-forwarded-for': '127.0.0.1'
        }
      });
      
      // Mock connection object
      loginReq.connection = { remoteAddress: '127.0.0.1' };

      await loginHandler(loginReq, loginRes);
      const loginData = JSON.parse(loginRes._getData());
      const accessToken = loginData.data.accessToken;

      // Now check permissions
      const { req, res } = createMocks({
        method: 'POST',
        headers: {
          authorization: `Bearer ${accessToken}`
        },
        body: {
          permissions: ['content.read', 'content.create'],
          checkType: 'any'
        }
      });

      await permissionsHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('hasPermission');
    });

    it('should require authentication for permissions check', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          permissions: ['content.read']
        }
      });

      try {
        await permissionsHandler(req, res);
        expect(res._getStatusCode()).toBe(401);
      } catch (error) {
        // The middleware should handle the error and return 401
        expect(res._getStatusCode()).toBe(401);
      }
    }, 1000); // 1 second timeout
  });

  describe('Error Handling', () => {
    it('should handle method not allowed', async () => {
      const { req, res } = createMocks({
        method: 'GET' // Wrong method
      });

      await loginHandler(req, res);

      expect(res._getStatusCode()).toBe(405);
    });
  });
});