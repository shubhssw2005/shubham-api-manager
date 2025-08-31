import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import mongoose from 'mongoose';
import Media from '../../models/Media.js';
import MediaFolder from '../../models/MediaFolder.js';
import User from '../../models/User.js';
import dbConnect from '../../lib/dbConnect.js';

describe('Media Model', () => {
  let testUser;
  let testFolder;

  beforeEach(async () => {
    const connection = await dbConnect();
    
    // Wait for connection to be ready
    if (connection.connection.readyState !== 1) {
      await new Promise(resolve => {
        connection.connection.once('connected', resolve);
      });
    }
    
    // Clean up first to ensure fresh state
    await Media.deleteMany({});
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

    // Create test folder
    testFolder = new MediaFolder({
      name: 'Test Folder',
      createdBy: testUser._id
    });
    await testFolder.save();
  });

  afterEach(async () => {
    // Clean up test data
    await Media.deleteMany({});
    await MediaFolder.deleteMany({});
    await User.deleteMany({});
  });

  describe('Model Creation', () => {
    it('should create a media file with required fields', async () => {
      const mediaData = {
        filename: 'test-image.jpg',
        originalName: 'test image.jpg',
        mimeType: 'image/jpeg',
        size: 1024000,
        path: '/uploads/test-image.jpg',
        url: 'http://localhost:3000/uploads/test-image.jpg',
        uploadedBy: testUser._id
      };

      const media = new Media(mediaData);
      const savedMedia = await media.save();

      expect(savedMedia._id).toBeDefined();
      expect(savedMedia.filename).toBe('test-image.jpg');
      expect(savedMedia.originalName).toBe('test image.jpg');
      expect(savedMedia.mimeType).toBe('image/jpeg');
      expect(savedMedia.size).toBe(1024000);
      expect(savedMedia.uploadedBy.toString()).toBe(testUser._id.toString());
      expect(savedMedia.processingStatus).toBe('pending');
    });

    it('should create media with folder reference', async () => {
      const media = new Media({
        filename: 'folder-test.jpg',
        originalName: 'folder test.jpg',
        mimeType: 'image/jpeg',
        size: 500000,
        path: '/uploads/folder-test.jpg',
        url: 'http://localhost:3000/uploads/folder-test.jpg',
        folder: testFolder._id,
        uploadedBy: testUser._id
      });

      const savedMedia = await media.save();
      expect(savedMedia.folder.toString()).toBe(testFolder._id.toString());
    });

    it('should validate required fields', async () => {
      const media = new Media({});
      
      await expect(media.save()).rejects.toThrow();
    });
  });

  describe('Virtuals', () => {
    let media;

    beforeEach(async () => {
      media = new Media({
        filename: 'test-virtual.jpg',
        originalName: 'test virtual.jpg',
        mimeType: 'image/jpeg',
        size: 2048000,
        path: '/uploads/test-virtual.jpg',
        url: 'http://localhost:3000/uploads/test-virtual.jpg',
        uploadedBy: testUser._id
      });
      await media.save();
    });

    it('should return correct file type for images', () => {
      expect(media.fileType).toBe('image');
    });

    it('should return correct file type for videos', async () => {
      media.mimeType = 'video/mp4';
      expect(media.fileType).toBe('video');
    });

    it('should return correct file type for documents', async () => {
      media.mimeType = 'application/pdf';
      expect(media.fileType).toBe('document');
    });

    it('should format file size correctly', () => {
      expect(media.formattedSize).toBe('2 MB');
    });

    it('should return usage count', () => {
      expect(media.usageCount).toBe(0);
      
      media.usage.push({
        modelName: 'Article',
        documentId: new mongoose.Types.ObjectId(),
        fieldName: 'featuredImage'
      });
      
      expect(media.usageCount).toBe(1);
    });
  });

  describe('Instance Methods', () => {
    let media;

    beforeEach(async () => {
      media = new Media({
        filename: 'test-methods.jpg',
        originalName: 'test methods.jpg',
        mimeType: 'image/jpeg',
        size: 1024000,
        path: '/uploads/test-methods.jpg',
        url: 'http://localhost:3000/uploads/test-methods.jpg',
        uploadedBy: testUser._id,
        thumbnails: [
          {
            size: 'small',
            url: '/uploads/thumbs/small-test-methods.jpg',
            path: '/uploads/thumbs/small-test-methods.jpg',
            width: 150,
            height: 150,
            fileSize: 5000
          },
          {
            size: 'medium',
            url: '/uploads/thumbs/medium-test-methods.jpg',
            path: '/uploads/thumbs/medium-test-methods.jpg',
            width: 300,
            height: 300,
            fileSize: 15000
          }
        ]
      });
      await media.save();
    });

    it('should add usage correctly', async () => {
      const documentId = new mongoose.Types.ObjectId();
      
      await media.addUsage('Article', documentId, 'featuredImage');
      
      expect(media.usage).toHaveLength(1);
      expect(media.usage[0].modelName).toBe('Article');
      expect(media.usage[0].documentId.toString()).toBe(documentId.toString());
      expect(media.usage[0].fieldName).toBe('featuredImage');
    });

    it('should not add duplicate usage', async () => {
      const documentId = new mongoose.Types.ObjectId();
      
      await media.addUsage('Article', documentId, 'featuredImage');
      await media.addUsage('Article', documentId, 'featuredImage');
      
      expect(media.usage).toHaveLength(1);
    });

    it('should remove usage correctly', async () => {
      const documentId = new mongoose.Types.ObjectId();
      
      await media.addUsage('Article', documentId, 'featuredImage');
      expect(media.usage).toHaveLength(1);
      
      await media.removeUsage('Article', documentId, 'featuredImage');
      expect(media.usage).toHaveLength(0);
    });

    it('should check if media is used', async () => {
      expect(media.isUsed()).toBe(false);
      
      const documentId = new mongoose.Types.ObjectId();
      await media.addUsage('Article', documentId, 'featuredImage');
      
      expect(media.isUsed()).toBe(true);
    });

    it('should check if media can be deleted', async () => {
      expect(media.canDelete()).toBe(true);
      
      const documentId = new mongoose.Types.ObjectId();
      await media.addUsage('Article', documentId, 'featuredImage');
      
      expect(media.canDelete()).toBe(false);
    });

    it('should get thumbnail by size', () => {
      const smallThumb = media.getThumbnail('small');
      expect(smallThumb).toBeDefined();
      expect(smallThumb.size).toBe('small');
      expect(smallThumb.width).toBe(150);
      
      const mediumThumb = media.getThumbnail('medium');
      expect(mediumThumb).toBeDefined();
      expect(mediumThumb.size).toBe('medium');
      expect(mediumThumb.width).toBe(300);
      
      const largeThumb = media.getThumbnail('large');
      expect(largeThumb).toBeUndefined();
    });
  });

  describe('Static Methods', () => {
    let imageMedia, videoMedia, documentMedia;

    beforeEach(async () => {
      imageMedia = new Media({
        filename: 'image.jpg',
        originalName: 'image.jpg',
        mimeType: 'image/jpeg',
        size: 1000000,
        path: '/uploads/image.jpg',
        url: 'http://localhost:3000/uploads/image.jpg',
        folder: testFolder._id,
        uploadedBy: testUser._id
      });
      await imageMedia.save();

      videoMedia = new Media({
        filename: 'video.mp4',
        originalName: 'video.mp4',
        mimeType: 'video/mp4',
        size: 5000000,
        path: '/uploads/video.mp4',
        url: 'http://localhost:3000/uploads/video.mp4',
        uploadedBy: testUser._id
      });
      await videoMedia.save();

      documentMedia = new Media({
        filename: 'document.pdf',
        originalName: 'document.pdf',
        mimeType: 'application/pdf',
        size: 2000000,
        path: '/uploads/document.pdf',
        url: 'http://localhost:3000/uploads/document.pdf',
        uploadedBy: testUser._id
      });
      await documentMedia.save();
    });

    it('should find media by folder', async () => {
      const folderMedia = await Media.findByFolder(testFolder._id);
      expect(folderMedia).toHaveLength(1);
      expect(folderMedia[0]._id.toString()).toBe(imageMedia._id.toString());

      const rootMedia = await Media.findByFolder(null);
      expect(rootMedia).toHaveLength(2);
    });

    it('should find media by type', async () => {
      const images = await Media.findByType('image');
      expect(images).toHaveLength(1);
      expect(images[0].mimeType).toBe('image/jpeg');

      const videos = await Media.findByType('video');
      expect(videos).toHaveLength(1);
      expect(videos[0].mimeType).toBe('video/mp4');

      const documents = await Media.findByType('document');
      expect(documents).toHaveLength(1);
      expect(documents[0].mimeType).toBe('application/pdf');
    });

    it('should find unused media', async () => {
      const unused = await Media.findUnused();
      expect(unused).toHaveLength(3);

      // Add usage to one media
      const documentId = new mongoose.Types.ObjectId();
      await imageMedia.addUsage('Article', documentId, 'featuredImage');

      const unusedAfter = await Media.findUnused();
      expect(unusedAfter).toHaveLength(2);
    });
  });

  describe('Pre-save Middleware', () => {
    it('should ensure tags are unique and lowercase', async () => {
      const media = new Media({
        filename: 'tags-test.jpg',
        originalName: 'tags test.jpg',
        mimeType: 'image/jpeg',
        size: 1000000,
        path: '/uploads/tags-test.jpg',
        url: 'http://localhost:3000/uploads/tags-test.jpg',
        tags: ['Nature', 'LANDSCAPE', 'nature', 'Photography'],
        uploadedBy: testUser._id
      });

      await media.save();
      
      expect(media.tags).toEqual(['nature', 'landscape', 'photography']);
    });
  });
});