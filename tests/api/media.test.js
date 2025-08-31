import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import dbConnect from '../../lib/dbConnect';
import Media from '../../models/Media';
import MediaFolder from '../../models/MediaFolder';
import User from '../../models/User';
import { createMocks } from 'node-mocks-http';
import uploadHandler from '../../pages/api/media/upload';
import mediaHandler from '../../pages/api/media/index';
import mediaItemHandler from '../../pages/api/media/[id]';
import bulkMediaHandler from '../../pages/api/media/bulk';

// Mock dependencies
vi.mock('../../lib/dbConnect');
vi.mock('../../middleware/auth');
vi.mock('../../lib/storage');
vi.mock('../../lib/fileProcessor');

describe('Media API Endpoints', () => {
  let mockUser;
  let mockMedia;
  let mockFolder;

  beforeEach(async () => {
    // Setup mock user
    mockUser = {
      _id: '507f1f77bcf86cd799439011',
      username: 'testuser',
      email: 'test@example.com',
      role: 'user',
      status: 'approved'
    };

    // Setup mock folder
    mockFolder = {
      _id: '507f1f77bcf86cd799439012',
      name: 'Test Folder',
      slug: 'test-folder',
      path: '/test-folder',
      level: 0,
      parent: null,
      isPublic: false,
      createdBy: mockUser._id,
      canUserAccess: vi.fn().mockReturnValue(true)
    };

    // Setup mock media
    mockMedia = {
      _id: '507f1f77bcf86cd799439013',
      filename: 'test-image.jpg',
      originalName: 'test-image.jpg',
      mimeType: 'image/jpeg',
      size: 1024000,
      path: 'media/2024/1/test-image.jpg',
      url: 'http://localhost:3000/uploads/media/2024/1/test-image.jpg',
      storageProvider: 'local',
      thumbnails: [],
      metadata: { width: 800, height: 600 },
      folder: mockFolder._id,
      tags: ['test'],
      alt: 'Test image',
      caption: 'A test image',
      description: 'This is a test image',
      usage: [],
      isPublic: false,
      uploadedBy: mockUser._id,
      processingStatus: 'completed',
      createdAt: new Date(),
      updatedAt: new Date()
    };

    // Mock database connection
    dbConnect.mockResolvedValue();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/media', () => {
    it('should return media files with pagination', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        query: { page: '1', limit: '20' }
      });

      // Mock authentication
      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      // Mock database queries
      Media.find = vi.fn().mockReturnValue({
        sort: vi.fn().mockReturnValue({
          skip: vi.fn().mockReturnValue({
            limit: vi.fn().mockReturnValue({
              populate: vi.fn().mockReturnValue({
                populate: vi.fn().mockReturnValue({
                  lean: vi.fn().mockResolvedValue([mockMedia])
                })
              })
            })
          })
        })
      });

      Media.countDocuments = vi.fn().mockResolvedValue(1);

      await mediaHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.mediaFiles).toHaveLength(1);
      expect(data.data.pagination.totalCount).toBe(1);
    });

    it('should filter media files by folder', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        query: { folder: mockFolder._id.toString() }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      MediaFolder.findById = vi.fn().mockResolvedValue(mockFolder);
      
      Media.find = vi.fn().mockReturnValue({
        sort: vi.fn().mockReturnValue({
          skip: vi.fn().mockReturnValue({
            limit: vi.fn().mockReturnValue({
              populate: vi.fn().mockReturnValue({
                populate: vi.fn().mockReturnValue({
                  lean: vi.fn().mockResolvedValue([mockMedia])
                })
              })
            })
          })
        })
      });

      Media.countDocuments = vi.fn().mockResolvedValue(1);

      await mediaHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      expect(Media.find).toHaveBeenCalledWith(
        expect.objectContaining({ folder: mockFolder._id.toString() }),
        null,
        expect.any(Object)
      );
    });

    it('should filter media files by type', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        query: { type: 'image' }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.find = vi.fn().mockReturnValue({
        sort: vi.fn().mockReturnValue({
          skip: vi.fn().mockReturnValue({
            limit: vi.fn().mockReturnValue({
              populate: vi.fn().mockReturnValue({
                populate: vi.fn().mockReturnValue({
                  lean: vi.fn().mockResolvedValue([mockMedia])
                })
              })
            })
          })
        })
      });

      Media.countDocuments = vi.fn().mockResolvedValue(1);

      await mediaHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      expect(Media.find).toHaveBeenCalledWith(
        expect.objectContaining({ 
          mimeType: expect.objectContaining({ $regex: /^image\// })
        }),
        null,
        expect.any(Object)
      );
    });
  });

  describe('GET /api/media/[id]', () => {
    it('should return a specific media file', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        query: { id: mockMedia._id.toString() }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.findById = vi.fn().mockReturnValue({
        populate: vi.fn().mockReturnValue({
          populate: vi.fn().mockReturnValue({
            populate: vi.fn().mockResolvedValue({
              ...mockMedia,
              uploadedBy: { _id: mockUser._id }
            })
          })
        })
      });

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.id).toBe(mockMedia._id);
    });

    it('should return 404 for non-existent media file', async () => {
      const { req, res } = createMocks({
        method: 'GET',
        query: { id: '507f1f77bcf86cd799439999' }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.findById = vi.fn().mockReturnValue({
        populate: vi.fn().mockReturnValue({
          populate: vi.fn().mockReturnValue({
            populate: vi.fn().mockResolvedValue(null)
          })
        })
      });

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(404);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.error.code).toBe('NOT_FOUND');
    });
  });

  describe('PUT /api/media/[id]', () => {
    it('should update media file metadata', async () => {
      const updateData = {
        alt: 'Updated alt text',
        caption: 'Updated caption',
        tags: ['updated', 'test']
      };

      const { req, res } = createMocks({
        method: 'PUT',
        query: { id: mockMedia._id.toString() },
        body: updateData
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.findById = vi.fn().mockResolvedValue({
        ...mockMedia,
        uploadedBy: mockUser._id
      });

      Media.findByIdAndUpdate = vi.fn().mockReturnValue({
        populate: vi.fn().mockReturnValue({
          populate: vi.fn().mockResolvedValue({
            ...mockMedia,
            ...updateData,
            uploadedBy: mockUser
          })
        })
      });

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.alt).toBe(updateData.alt);
      expect(data.data.caption).toBe(updateData.caption);
    });

    it('should return 403 for unauthorized update', async () => {
      const { req, res } = createMocks({
        method: 'PUT',
        query: { id: mockMedia._id.toString() },
        body: { alt: 'Updated alt' }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue({
        ...mockUser,
        _id: '507f1f77bcf86cd799439999' // Different user
      });

      Media.findById = vi.fn().mockResolvedValue({
        ...mockMedia,
        uploadedBy: mockUser._id
      });

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.error.code).toBe('VALIDATION_ERROR');
    });
  });

  describe('DELETE /api/media/[id]', () => {
    it('should delete media file', async () => {
      const { req, res } = createMocks({
        method: 'DELETE',
        query: { id: mockMedia._id.toString() }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.findById = vi.fn().mockResolvedValue({
        ...mockMedia,
        uploadedBy: mockUser._id,
        usage: []
      });

      Media.findByIdAndDelete = vi.fn().mockResolvedValue(mockMedia);

      // Mock storage provider
      const { StorageFactory } = await import('../../lib/storage');
      const mockStorageProvider = {
        delete: vi.fn().mockResolvedValue()
      };
      StorageFactory.createFromEnv = vi.fn().mockReturnValue(mockStorageProvider);

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(Media.findByIdAndDelete).toHaveBeenCalledWith(mockMedia._id.toString());
    });

    it('should return conflict error for media file in use', async () => {
      const { req, res } = createMocks({
        method: 'DELETE',
        query: { id: mockMedia._id.toString() }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.findById = vi.fn().mockResolvedValue({
        ...mockMedia,
        uploadedBy: mockUser._id,
        usage: [{ modelName: 'Post', documentId: '507f1f77bcf86cd799439014' }]
      });

      await mediaItemHandler(req, res);

      expect(res._getStatusCode()).toBe(409);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.error.code).toBe('CONFLICT');
    });
  });

  describe('POST /api/media/bulk', () => {
    it('should perform bulk delete operation', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          action: 'delete',
          mediaIds: [mockMedia._id.toString()],
          data: { force: false }
        }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.find = vi.fn().mockResolvedValue([{
        ...mockMedia,
        uploadedBy: mockUser._id,
        usage: []
      }]);

      Media.findByIdAndDelete = vi.fn().mockResolvedValue(mockMedia);

      // Mock storage provider
      const { StorageFactory } = await import('../../lib/storage');
      const mockStorageProvider = {
        delete: vi.fn().mockResolvedValue()
      };
      StorageFactory.createFromEnv = vi.fn().mockReturnValue(mockStorageProvider);

      await bulkMediaHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.deleted).toHaveLength(1);
      expect(data.data.failed).toHaveLength(0);
    });

    it('should perform bulk move operation', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          action: 'move',
          mediaIds: [mockMedia._id.toString()],
          data: { folderId: mockFolder._id.toString() }
        }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      Media.find = vi.fn().mockResolvedValue([{
        ...mockMedia,
        uploadedBy: mockUser._id
      }]);

      MediaFolder.findById = vi.fn().mockResolvedValue(mockFolder);
      Media.findByIdAndUpdate = vi.fn().mockResolvedValue();

      await bulkMediaHandler(req, res);

      expect(res._getStatusCode()).toBe(200);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(true);
      expect(data.data.moved).toHaveLength(1);
      expect(data.data.failed).toHaveLength(0);
    });

    it('should return validation error for invalid action', async () => {
      const { req, res } = createMocks({
        method: 'POST',
        body: {
          action: 'invalid',
          mediaIds: [mockMedia._id.toString()]
        }
      });

      const { requireApprovedUser } = await import('../../middleware/auth');
      requireApprovedUser.mockResolvedValue(mockUser);

      await bulkMediaHandler(req, res);

      expect(res._getStatusCode()).toBe(400);
      const data = JSON.parse(res._getData());
      expect(data.success).toBe(false);
      expect(data.error.code).toBe('VALIDATION_ERROR');
    });
  });
});