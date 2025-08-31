import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createMocks } from 'node-mocks-http';
import handler from '../../pages/api/settings/tokens';
import dbConnect from '../../lib/dbConnect';
import User from '../../models/User';
import APIToken from '../../models/APIToken';
import AuditLog from '../../models/AuditLog';

// Mock dependencies
vi.mock('../../lib/dbConnect');
vi.mock('../../lib/jwt', () => ({
  verifyToken: vi.fn(),
  getTokenFromRequest: vi.fn()
}));

describe('/api/settings/tokens', () => {
  let mockUser;
  let mockToken;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockUser = {
      _id: 'user123',
      name: 'Test User',
      email: 'test@example.com',
      role: 'admin',
      isApproved: () => true
    };

    mockToken = {
      _id: 'token123',
      name: 'Test Token',
      token: 'test-token-value',
      maskedToken: 'test-tok...alue',
      permissions: [
        { model: 'User', actions: ['read', 'create'] }
      ],
      rateLimit: { requests: 1000, window: 3600 },
      isActive: true,
      createdBy: mockUser._id,
      usage: { totalRequests: 0 },
      toJSON: () => ({ ...mockToken, token: undefined, hashedToken: undefined })
    };

    // Mock database connection
    dbConnect.mockResolvedValue();
    
    // Mock User.findById
    User.findById = vi.fn().mockResolvedValue(mockUser);
    
    // Mock APIToken methods
    APIToken.find = vi.fn();
    APIToken.prototype.save = vi.fn();
    APIToken.findById = vi.fn();
    APIToken.generateToken = vi.fn().mockReturnValue('generated-token-123');
    
    // Mock AuditLog
    AuditLog.logAction = vi.fn().mockResolvedValue();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('GET /api/settings/tokens', () => {
    it('should return list of tokens for admin user', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      // Mock token find
      APIToken.find.mockReturnValue({
        populate: vi.fn().mockReturnValue({
          sort: vi.fn().mockResolvedValue([mockToken])
        })
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.tokens).toHaveLength(1);
      expect(data.tokens[0].name).toBe('Test Token');
    });

    it('should return 401 for non-admin user', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      // Mock non-admin user
      User.findById.mockResolvedValue({
        ...mockUser,
        role: 'user'
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(403);
      const data = JSON.parse(res._getData());
      expect(data.error).toBe('Insufficient permissions');
    });

    it('should return 401 for missing token', async () => {
      const { req, res } = createMocks({
        method: 'GET'
      });

      // Mock JWT verification
      const { getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue(null);

      await handler(req, res);

      expect(res._getStatusCode()).toBe(401);
      const data = JSON.parse(res._getData());
      expect(data.error).toBe('No token provided');
    });
  });

  describe('POST /api/settings/tokens', () => {
    it('should create new API token', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          name: 'New Token',
          permissions: [
            { model: 'User', actions: ['read'] }
          ],
          description: 'Test token'
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      // Mock token creation
      const mockNewToken = {
        ...mockToken,
        name: 'New Token',
        token: 'generated-token-123',
        save: vi.fn().mockResolvedValue(),
        toJSON: () => ({ ...mockToken, name: 'New Token' })
      };

      // Mock APIToken constructor
      global.APIToken = vi.fn().mockImplementation(() => mockNewToken);
      APIToken.mockImplementation(() => mockNewToken);

      await handler(req, res);

      expect(res._getStatusCode()).toBe(201);
      const data = JSON.parse(res._getData());
      expect(data.token.name).toBe('New Token');
      expect(data.token.token).toBe('generated-token-123');
      expect(AuditLog.logAction).toHaveBeenCalledWith(
        expect.objectContaining({
          action: 'token_create',
          resource: 'api_token'
        })
      );
    });

    it('should return 400 for missing required fields', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          // Missing name and permissions
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.error).toBe('Missing required fields');
    });

    it('should return 400 for invalid permissions structure', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          name: 'Test Token',
          permissions: [
            { model: 'User' } // Missing actions
          ]
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.error).toBe('Invalid permissions structure');
    });
  });

  describe('PUT /api/settings/tokens', () => {
    it('should update existing token', async () => {
      const { req, res } = createMocks({
        method: 'PUT',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          id: 'token123',
          name: 'Updated Token',
          isActive: false
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      // Mock token find and update
      const mockUpdatedToken = {
        ...mockToken,
        name: 'Updated Token',
        isActive: false,
        save: vi.fn().mockResolvedValue()
      };

      APIToken.findById.mockResolvedValue(mockUpdatedToken);

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.token.name).toBe('Updated Token');
      expect(AuditLog.logAction).toHaveBeenCalledWith(
        expect.objectContaining({
          action: 'update',
          resource: 'api_token'
        })
      );
    });

    it('should return 404 for non-existent token', async () => {
      const { req, res } = createMocks({
        method: 'PUT',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          id: 'non-existent-token'
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      APIToken.findById.mockResolvedValue(null);

      await handler(req, res);

      expect(res._getStatusCode()).toBe(404);
      const data = JSON.parse(res._getData());
      expect(data.error).toBe('Token not found');
    });
  });

  describe('DELETE /api/settings/tokens', () => {
    it('should revoke token', async () => {
      const { req, res } = createMocks({
        method: 'DELETE',
        headers: {
          authorization: 'Bearer valid-jwt-token'
        },
        body: {
          id: 'token123'
        }
      });

      // Mock JWT verification
      const { verifyToken, getTokenFromRequest } = await import('../../lib/jwt');
      getTokenFromRequest.mockReturnValue('valid-jwt-token');
      verifyToken.mockReturnValue({ userId: 'user123' });

      // Mock token find and revoke
      const mockTokenToRevoke = {
        ...mockToken,
        revoke: vi.fn().mockResolvedValue()
      };

      APIToken.findById.mockResolvedValue(mockTokenToRevoke);

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.message).toBe('Token revoked successfully');
      expect(mockTokenToRevoke.revoke).toHaveBeenCalled();
      expect(AuditLog.logAction).toHaveBeenCalledWith(
        expect.objectContaining({
          action: 'token_revoke',
          resource: 'api_token'
        })
      );
    });
  });
});