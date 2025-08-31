import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import mongoose from 'mongoose';
import MediaFolder from '../../models/MediaFolder.js';
import User from '../../models/User.js';
import dbConnect from '../../lib/dbConnect.js';

describe('MediaFolder Model', () => {
  let testUser;

  beforeEach(async () => {
    const connection = await dbConnect();
    
    // Wait for connection to be ready
    if (connection.connection.readyState !== 1) {
      await new Promise(resolve => {
        connection.connection.once('connected', resolve);
      });
    }
    
    // Clean up first to ensure fresh state
    await MediaFolder.deleteMany({});
    await User.deleteMany({});
    
    // Create test user with unique email
    testUser = new User({
      email: `test-${Date.now()}@example.com`,
      password: 'password123',
      name: 'Test User',
      status: 'approved'
    });
    await testUser.save();
  });

  afterEach(async () => {
    // Clean up test data
    await MediaFolder.deleteMany({});
    await User.deleteMany({});
  });

  describe('Model Creation', () => {
    it('should create a folder with required fields', async () => {
      const folderData = {
        name: 'Test Folder',
        createdBy: testUser._id
      };

      const folder = new MediaFolder(folderData);
      const savedFolder = await folder.save();

      expect(savedFolder._id).toBeDefined();
      expect(savedFolder.name).toBe('Test Folder');
      expect(savedFolder.slug).toBe('test-folder');
      expect(savedFolder.path).toBe('/test-folder');
      expect(savedFolder.level).toBe(0);
      expect(savedFolder.createdBy.toString()).toBe(testUser._id.toString());
    });

    it('should create nested folder with parent', async () => {
      // Create parent folder
      const parentFolder = new MediaFolder({
        name: 'Parent Folder',
        createdBy: testUser._id
      });
      await parentFolder.save();

      // Create child folder
      const childFolder = new MediaFolder({
        name: 'Child Folder',
        parent: parentFolder._id,
        createdBy: testUser._id
      });
      await childFolder.save();

      expect(childFolder.path).toBe('/parent-folder/child-folder');
      expect(childFolder.level).toBe(1);
      expect(childFolder.parent.toString()).toBe(parentFolder._id.toString());
    });

    it('should validate required fields', async () => {
      const folder = new MediaFolder({});
      
      await expect(folder.save()).rejects.toThrow();
    });

    it('should prevent excessive nesting', async () => {
      let currentFolder = null;
      
      // Create 10 levels of nesting (should work)
      for (let i = 0; i < 10; i++) {
        const folder = new MediaFolder({
          name: `Level ${i}`,
          parent: currentFolder?._id,
          createdBy: testUser._id
        });
        await folder.save();
        currentFolder = folder;
      }

      // Try to create 11th level (should fail)
      const deepFolder = new MediaFolder({
        name: 'Too Deep',
        parent: currentFolder._id,
        createdBy: testUser._id
      });

      await expect(deepFolder.save()).rejects.toThrow('Maximum folder nesting level');
    });
  });

  describe('Virtuals', () => {
    it('should return correct breadcrumbs', async () => {
      const parentFolder = new MediaFolder({
        name: 'Parent',
        createdBy: testUser._id
      });
      await parentFolder.save();

      const childFolder = new MediaFolder({
        name: 'Child',
        parent: parentFolder._id,
        createdBy: testUser._id
      });
      await childFolder.save();

      const breadcrumbs = childFolder.breadcrumbs;
      expect(breadcrumbs).toHaveLength(2);
      expect(breadcrumbs[0]).toEqual({ name: 'parent', path: '/parent' });
      expect(breadcrumbs[1]).toEqual({ name: 'child', path: '/parent/child' });
    });
  });

  describe('Instance Methods', () => {
    let parentFolder, childFolder, grandchildFolder;

    beforeEach(async () => {
      parentFolder = new MediaFolder({
        name: 'Parent',
        createdBy: testUser._id
      });
      await parentFolder.save();

      childFolder = new MediaFolder({
        name: 'Child',
        parent: parentFolder._id,
        createdBy: testUser._id
      });
      await childFolder.save();

      grandchildFolder = new MediaFolder({
        name: 'Grandchild',
        parent: childFolder._id,
        createdBy: testUser._id
      });
      await grandchildFolder.save();
    });

    it('should get ancestors correctly', async () => {
      const ancestors = await grandchildFolder.getAncestors();
      
      expect(ancestors).toHaveLength(2);
      expect(ancestors[0]._id.toString()).toBe(parentFolder._id.toString());
      expect(ancestors[1]._id.toString()).toBe(childFolder._id.toString());
    });

    it('should get descendants correctly', async () => {
      const descendants = await parentFolder.getDescendants();
      
      expect(descendants).toHaveLength(2);
      expect(descendants.some(d => d._id.toString() === childFolder._id.toString())).toBe(true);
      expect(descendants.some(d => d._id.toString() === grandchildFolder._id.toString())).toBe(true);
    });

    it('should check user access permissions', () => {
      // Creator should have access
      expect(parentFolder.canUserAccess(testUser._id, 'read')).toBe(true);
      expect(parentFolder.canUserAccess(testUser._id, 'write')).toBe(true);
      expect(parentFolder.canUserAccess(testUser._id, 'admin')).toBe(true);

      // Other user should not have access
      const otherUserId = new mongoose.Types.ObjectId();
      expect(parentFolder.canUserAccess(otherUserId, 'read')).toBe(false);

      // Public folder should allow read access
      parentFolder.isPublic = true;
      expect(parentFolder.canUserAccess(otherUserId, 'read')).toBe(true);
      expect(parentFolder.canUserAccess(otherUserId, 'write')).toBe(false);
    });

    it('should add and remove permissions', async () => {
      const otherUserId = new mongoose.Types.ObjectId();
      
      // Add read permission
      await parentFolder.addPermission(otherUserId, 'read');
      expect(parentFolder.canUserAccess(otherUserId, 'read')).toBe(true);
      expect(parentFolder.canUserAccess(otherUserId, 'write')).toBe(false);

      // Add write permission
      await parentFolder.addPermission(otherUserId, 'write');
      expect(parentFolder.canUserAccess(otherUserId, 'write')).toBe(true);

      // Remove read permission
      await parentFolder.removePermission(otherUserId, 'read');
      expect(parentFolder.canUserAccess(otherUserId, 'read')).toBe(true); // Still has write, which includes read
      expect(parentFolder.canUserAccess(otherUserId, 'write')).toBe(true);

      // Remove write permission
      await parentFolder.removePermission(otherUserId, 'write');
      expect(parentFolder.canUserAccess(otherUserId, 'read')).toBe(false);
      expect(parentFolder.canUserAccess(otherUserId, 'write')).toBe(false);
    });
  });

  describe('Static Methods', () => {
    let publicFolder, privateFolder;

    beforeEach(async () => {
      publicFolder = new MediaFolder({
        name: 'Public Folder',
        isPublic: true,
        createdBy: testUser._id
      });
      await publicFolder.save();

      privateFolder = new MediaFolder({
        name: 'Private Folder',
        isPublic: false,
        createdBy: testUser._id
      });
      await privateFolder.save();
    });

    it('should find folder by path', async () => {
      const found = await MediaFolder.findByPath('/public-folder');
      expect(found).toBeDefined();
      expect(found._id.toString()).toBe(publicFolder._id.toString());
    });

    it('should get root folders', async () => {
      const rootFolders = await MediaFolder.getRootFolders();
      expect(rootFolders).toHaveLength(2);

      // Test with user filter
      const userRootFolders = await MediaFolder.getRootFolders(testUser._id);
      expect(userRootFolders).toHaveLength(2);

      // Test with different user (should only see public)
      const otherUserId = new mongoose.Types.ObjectId();
      const otherUserRootFolders = await MediaFolder.getRootFolders(otherUserId);
      expect(otherUserRootFolders).toHaveLength(1);
      expect(otherUserRootFolders[0].isPublic).toBe(true);
    });

    it('should build folder tree', async () => {
      // Create nested structure
      const childFolder = new MediaFolder({
        name: 'Child of Public',
        parent: publicFolder._id,
        createdBy: testUser._id
      });
      await childFolder.save();

      const tree = await MediaFolder.buildFolderTree();
      expect(tree).toHaveLength(2);
      
      const publicFolderInTree = tree.find(f => f._id.toString() === publicFolder._id.toString());
      expect(publicFolderInTree.children).toHaveLength(1);
      expect(publicFolderInTree.children[0]._id.toString()).toBe(childFolder._id.toString());
    });

    it('should validate path format', () => {
      expect(MediaFolder.validatePath('/valid-path')).toEqual({ valid: true });
      expect(MediaFolder.validatePath('invalid-path')).toEqual({ 
        valid: false, 
        error: 'Path must start with /' 
      });
      expect(MediaFolder.validatePath('/invalid path with spaces')).toEqual({ 
        valid: false, 
        error: 'Path contains invalid characters' 
      });
      expect(MediaFolder.validatePath('')).toEqual({ 
        valid: false, 
        error: 'Path is required and must be a string' 
      });
    });
  });

  describe('Pre-save Middleware', () => {
    it('should generate slug from name', async () => {
      const folder = new MediaFolder({
        name: 'Test Folder With Spaces',
        createdBy: testUser._id
      });
      await folder.save();

      expect(folder.slug).toBe('test-folder-with-spaces');
      expect(folder.path).toBe('/test-folder-with-spaces');
    });

    it('should use provided slug', async () => {
      const folder = new MediaFolder({
        name: 'Test Folder',
        slug: 'custom-slug',
        createdBy: testUser._id
      });
      await folder.save();

      expect(folder.slug).toBe('custom-slug');
      expect(folder.path).toBe('/custom-slug');
    });
  });

  describe('Pre-remove Middleware', () => {
    it('should prevent deletion of folder with children', async () => {
      const parentFolder = new MediaFolder({
        name: 'Parent',
        createdBy: testUser._id
      });
      await parentFolder.save();

      const childFolder = new MediaFolder({
        name: 'Child',
        parent: parentFolder._id,
        createdBy: testUser._id
      });
      await childFolder.save();

      await expect(parentFolder.remove()).rejects.toThrow('Cannot delete folder that contains subfolders');
    });
  });
});