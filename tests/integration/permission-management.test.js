import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createMocks } from 'node-mocks-http';
import rolesHandler from '../../pages/api/roles/index.js';
import usersRolesHandler from '../../pages/api/users/roles.js';
import dbConnect from '../../lib/dbConnect.js';
import { Role } from '../../models/Role.js';
import User from '../../models/User.js';

// Mock database connection
vi.mock('../../lib/dbConnect.js');
vi.mock('../../middleware/auth.js', () => ({
  requireApprovedUser: vi.fn().mockResolvedValue({
    _id: 'admin123',
    role: 'admin',
    name: 'Test Admin',
    email: 'admin@test.com'
  })
}));
vi.mock('../../lib/rbac.js', () => ({
  hasPermission: vi.fn().mockReturnValue(true),
  PERMISSION: {
    CREATE: 'create',
    READ: 'read',
    UPDATE: 'update',
    DELETE: 'delete'
  }
}));

describe('Permission Management Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    dbConnect.mockResolvedValue(true);
  });

  describe('Role Management Workflow', () => {
    it('should create, update, and delete a role', async () => {
      // Mock Role model methods
      const mockRole = {
        _id: 'role123',
        name: 'editor',
        displayName: 'Editor',
        permission: { create: true, read: true, update: true, delete: false },
        routes: ['/manager', '/media'],
        description: 'Editor role',
        isSystemRole: false,
        toObject: function() {
          return {
            _id: this._id,
            name: this.name,
            displayName: this.displayName,
            permission: this.permission,
            routes: this.routes,
            description: this.description,
            isSystemRole: this.isSystemRole
          };
        },
        save: vi.fn().mockResolvedValue(true)
      };

      // Step 1: Create role
      Role.findOne = vi.fn().mockResolvedValue(null); // Role doesn't exist
      Role.find = vi.fn().mockReturnValue({
        sort: vi.fn().mockResolvedValue([])
      });
      User.countDocuments = vi.fn().mockResolvedValue(0);
      
      // Mock Role constructor
      Role.mockImplementation(() => mockRole);

      const { req: createReq, res: createRes } = createMocks({
        method: 'POST',
        body: {
          name: 'Editor',
          permission: { create: true, read: true, update: true, delete: false },
          routes: ['/manager', '/media'],
          description: 'Editor role'
        }
      });

      await rolesHandler(createReq, createRes);

      expect(createRes._getStatusCode()).toBe(201);
      const createData = JSON.parse(createRes._getData());
      expect(createData.success).toBe(true);
      expect(createData.data.name).toBe('editor');

      // Step 2: Update role
      Role.findById = vi.fn().mockResolvedValue(mockRole);
      
      const { req: updateReq, res: updateRes } = createMocks({
        method: 'PUT',
        body: {
          id: 'role123',
          permission: { create: true, read: true, update: true, delete: true },
          routes: ['/manager', '/media', '/setting'],
          description: 'Updated editor role'
        }
      });

      await rolesHandler(updateReq, updateRes);

      expect(updateRes._getStatusCode()).toBe(200);
      const updateData = JSON.parse(updateRes._getData());
      expect(updateData.success).toBe(true);
      expect(mockRole.save).toHaveBeenCalled();

      // Step 3: Delete role
      Role.findByIdAndDelete = vi.fn().mockResolvedValue(mockRole);
      
      const { req: deleteReq, res: deleteRes } = createMocks({
        method: 'DELETE',
        body: {
          id: 'role123'
        }
      });

      await rolesHandler(deleteReq, deleteRes);

      expect(deleteRes._getStatusCode()).toBe(200);
      const deleteData = JSON.parse(deleteRes._getData());
      expect(deleteData.success).toBe(true);
      expect(deleteData.message).toBe('Role deleted successfully');
    });
  });

  describe('User Role Assignment Workflow', () => {
    it('should assign role to user', async () => {
      const mockUser = {
        _id: 'user123',
        name: 'Test User',
        email: 'user@test.com',
        role: 'user',
        toObject: function() {
          return {
            _id: this._id,
            name: this.name,
            email: this.email,
            role: this.role
          };
        },
        save: vi.fn().mockResolvedValue(true)
      };

      const mockRole = {
        _id: 'role123',
        name: 'editor',
        displayName: 'Editor',
        description: 'Editor role',
        isSystemRole: false
      };

      // Mock database queries
      Role.findOne = vi.fn().mockResolvedValue(mockRole);
      User.findById = vi.fn().mockResolvedValue(mockUser);

      const { req, res } = createMocks({
        method: 'PUT',
        body: {
          userId: 'user123',
          newRole: 'editor'
        }
      });

      await usersRolesHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.message).toBe('User role updated successfully');
      expect(mockUser.save).toHaveBeenCalled();
    });

    it('should prevent user from changing their own role', async () => {
      const mockUser = {
        _id: 'admin123', // Same as the authenticated user
        name: 'Test Admin',
        email: 'admin@test.com',
        role: 'admin'
      };

      const mockRole = {
        _id: 'role123',
        name: 'user',
        displayName: 'User',
        description: 'Regular user role',
        isSystemRole: false
      };

      Role.findOne = vi.fn().mockResolvedValue(mockRole);
      User.findById = vi.fn().mockResolvedValue(mockUser);

      const { req, res } = createMocks({
        method: 'PUT',
        body: {
          userId: 'admin123', // Same as authenticated user
          newRole: 'user'
        }
      });

      await usersRolesHandler(req, res);

      expect(res._getStatusCode()).toBe(403);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Cannot change your own role');
    });
  });

  describe('Permission Inheritance and Conflict Resolution', () => {
    it('should handle system role protection', async () => {
      const systemRole = {
        _id: 'system123',
        name: 'superadmin',
        isSystemRole: true
      };

      Role.findById = vi.fn().mockResolvedValue(systemRole);

      // Try to update system role permissions
      const { req: updateReq, res: updateRes } = createMocks({
        method: 'PUT',
        body: {
          id: 'system123',
          permission: { create: false, read: true, update: false, delete: false }
        }
      });

      await rolesHandler(updateReq, updateRes);

      expect(updateRes._getStatusCode()).toBe(403);
      const updateData = JSON.parse(updateRes._getData());
      expect(updateData.success).toBe(false);
      expect(updateData.message).toBe('Cannot modify permissions of system roles');

      // Try to delete system role
      const { req: deleteReq, res: deleteRes } = createMocks({
        method: 'DELETE',
        body: {
          id: 'system123'
        }
      });

      await rolesHandler(deleteReq, deleteRes);

      expect(deleteRes._getStatusCode()).toBe(403);
      const deleteData = JSON.parse(deleteRes._getData());
      expect(deleteData.success).toBe(false);
      expect(deleteData.message).toBe('Cannot delete system roles');
    });

    it('should prevent deletion of roles with assigned users', async () => {
      const roleWithUsers = {
        _id: 'role123',
        name: 'editor',
        isSystemRole: false
      };

      Role.findById = vi.fn().mockResolvedValue(roleWithUsers);
      User.countDocuments = vi.fn().mockResolvedValue(5); // 5 users assigned

      const { req, res } = createMocks({
        method: 'DELETE',
        body: {
          id: 'role123'
        }
      });

      await rolesHandler(req, res);

      expect(res._getStatusCode()).toBe(409);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.message).toBe('Cannot delete role. 5 users are assigned to this role.');
    });
  });
});