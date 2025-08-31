import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createMocks } from 'node-mocks-http';
import handler from '../../pages/api/roles/index.js';
import { Role } from '../../models/Role.js';
import User from '../../models/User.js';
import * as auth from '../../middleware/auth.js';
import * as rbac from '../../lib/rbac.js';

// Mock dependencies
vi.mock('../../models/Role.js');
vi.mock('../../models/User.js');
vi.mock('../../middleware/auth.js');
vi.mock('../../lib/rbac.js');
vi.mock('../../lib/dbConnect.js', () => ({
  default: vi.fn().mockResolvedValue(true)
}));

describe('/api/roles', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/roles', () => {
    it('should return list of roles for authorized user', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const mockRoles = [
        {
          _id: 'role1',
          name: 'admin',
          displayName: 'Admin',
          permission: { create: true, read: true, update: true, delete: true },
          routes: ['*'],
          description: 'Administrator role',
          isSystemRole: true,
          toObject: () => ({
            _id: 'role1',
            name: 'admin',
            displayName: 'Admin',
            permission: { create: true, read: true, update: true, delete: true },
            routes: ['*'],
            description: 'Administrator role',
            isSystemRole: true
          })
        }
      ];

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.find.mockReturnValue({
        sort: vi.fn().mockResolvedValue(mockRoles)
      });
      User.countDocuments.mockResolvedValue(1);

      const { req, res } = createMocks({
        method: 'GET'
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data).toHaveLength(1);
      expect(data.data[0].userCount).toBe(1);
    });

    it('should return 401 for unauthenticated user', async () => {
      auth.requireApprovedUser.mockResolvedValue(null);

      const { req, res } = createMocks({
        method: 'GET'
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(401);
    });

    it('should return 403 for user without permissions', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'user',
        name: 'Test User'
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(false);

      const { req, res } = createMocks({
        method: 'GET'
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(403);
    });
  });

  describe('POST /api/roles', () => {
    it('should create new role for authorized user', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const mockNewRole = {
        _id: 'newrole123',
        name: 'editor',
        displayName: 'Editor',
        permission: { create: true, read: true, update: true, delete: false },
        routes: ['/manager', '/media'],
        description: 'Editor role',
        isSystemRole: false,
        toObject: () => ({
          _id: 'newrole123',
          name: 'editor',
          displayName: 'Editor',
          permission: { create: true, read: true, update: true, delete: false },
          routes: ['/manager', '/media'],
          description: 'Editor role',
          isSystemRole: false
        }),
        save: vi.fn().mockResolvedValue(true)
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findOne.mockResolvedValue(null); // Role doesn't exist
      Role.mockImplementation(() => mockNewRole);

      const { req, res } = createMocks({
        method: 'POST',
        body: {
          name: 'Editor',
          permission: { create: true, read: true, update: true, delete: false },
          routes: ['/manager', '/media'],
          description: 'Editor role'
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(201);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.name).toBe('editor');
    });

    it('should return 400 for missing role name', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);

      const { req, res } = createMocks({
        method: 'POST',
        body: {
          permission: { create: true, read: true, update: true, delete: false }
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Role name is required');
    });

    it('should return 409 for duplicate role name', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const existingRole = {
        _id: 'existing123',
        name: 'editor'
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findOne.mockResolvedValue(existingRole);

      const { req, res } = createMocks({
        method: 'POST',
        body: {
          name: 'Editor',
          permission: { create: true, read: true, update: true, delete: false }
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(409);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Role already exists');
    });
  });

  describe('PUT /api/roles', () => {
    it('should update role for authorized user', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const mockRole = {
        _id: 'role123',
        name: 'editor',
        displayName: 'Editor',
        permission: { create: true, read: true, update: false, delete: false },
        routes: ['/manager'],
        description: 'Editor role',
        isSystemRole: false,
        toObject: () => ({
          _id: 'role123',
          name: 'editor',
          displayName: 'Editor',
          permission: { create: true, read: true, update: true, delete: false },
          routes: ['/manager', '/media'],
          description: 'Updated editor role',
          isSystemRole: false
        }),
        save: vi.fn().mockResolvedValue(true)
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(mockRole);
      User.countDocuments.mockResolvedValue(2);

      const { req, res } = createMocks({
        method: 'PUT',
        body: {
          id: 'role123',
          permission: { create: true, read: true, update: true, delete: false },
          routes: ['/manager', '/media'],
          description: 'Updated editor role'
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(mockRole.save).toHaveBeenCalled();
    });

    it('should return 404 for non-existent role', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(null);

      const { req, res } = createMocks({
        method: 'PUT',
        body: {
          id: 'nonexistent123',
          permission: { create: true, read: true, update: true, delete: false }
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(404);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Role not found');
    });

    it('should return 403 for system role modification', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const systemRole = {
        _id: 'system123',
        name: 'superadmin',
        isSystemRole: true
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(systemRole);

      const { req, res } = createMocks({
        method: 'PUT',
        body: {
          id: 'system123',
          permission: { create: false, read: true, update: false, delete: false }
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(403);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Cannot modify permissions of system roles');
    });
  });

  describe('DELETE /api/roles', () => {
    it('should delete role for authorized user', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const mockRole = {
        _id: 'role123',
        name: 'editor',
        isSystemRole: false
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(mockRole);
      User.countDocuments.mockResolvedValue(0);
      Role.findByIdAndDelete.mockResolvedValue(mockRole);

      const { req, res } = createMocks({
        method: 'DELETE',
        body: {
          id: 'role123'
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.message).toBe('Role deleted successfully');
    });

    it('should return 403 for system role deletion', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const systemRole = {
        _id: 'system123',
        name: 'superadmin',
        isSystemRole: true
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(systemRole);

      const { req, res } = createMocks({
        method: 'DELETE',
        body: {
          id: 'system123'
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(403);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Cannot delete system roles');
    });

    it('should return 409 for role with assigned users', async () => {
      const mockUser = {
        _id: 'user123',
        role: 'admin',
        name: 'Test Admin'
      };

      const mockRole = {
        _id: 'role123',
        name: 'editor',
        isSystemRole: false
      };

      auth.requireApprovedUser.mockResolvedValue(mockUser);
      rbac.hasPermission.mockReturnValue(true);
      Role.findById.mockResolvedValue(mockRole);
      User.countDocuments.mockResolvedValue(3);

      const { req, res } = createMocks({
        method: 'DELETE',
        body: {
          id: 'role123'
        }
      });

      await handler(req, res);

      expect(res._getStatusCode()).toBe(409);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Cannot delete role. 3 users are assigned to this role.');
    });
  });
});