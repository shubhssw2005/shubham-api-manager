import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs/promises';
import path from 'path';
import { LocalStorageProvider } from '../../lib/storage/index.js';

describe('LocalStorageProvider', () => {
  let provider;
  const testDir = path.join(process.cwd(), 'test-uploads');
  
  beforeEach(async () => {
    provider = new LocalStorageProvider({
      basePath: testDir,
      baseUrl: '/test-uploads'
    });
    
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch (error) {
      // Directory might not exist
    }
  });

  afterEach(async () => {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch (error) {
      // Directory might not exist
    }
  });

  describe('upload', () => {
    it('should upload a buffer successfully', async () => {
      const testData = Buffer.from('test file content');
      const filePath = 'test/file.txt';
      
      const result = await provider.upload(testData, filePath, {
        contentType: 'text/plain'
      });

      expect(result).toMatchObject({
        path: filePath,
        url: '/test-uploads/test/file.txt',
        contentType: 'text/plain'
      });
      expect(result.size).toBe(testData.length);
      expect(result.lastModified).toBeInstanceOf(Date);

      // Verify file exists
      const exists = await provider.exists(filePath);
      expect(exists).toBe(true);
    });

    it('should create directories automatically', async () => {
      const testData = Buffer.from('test content');
      const filePath = 'deep/nested/directory/file.txt';
      
      const result = await provider.upload(testData, filePath);
      
      expect(result.path).toBe(filePath);
      
      // Verify file exists
      const exists = await provider.exists(filePath);
      expect(exists).toBe(true);
    });
  });

  describe('delete', () => {
    it('should delete an existing file', async () => {
      const testData = Buffer.from('test content');
      const filePath = 'test-delete.txt';
      
      // Upload file first
      await provider.upload(testData, filePath);
      
      // Verify file exists
      let exists = await provider.exists(filePath);
      expect(exists).toBe(true);
      
      // Delete file
      const result = await provider.delete(filePath);
      expect(result).toBe(true);
      
      // Verify file no longer exists
      exists = await provider.exists(filePath);
      expect(exists).toBe(false);
    });

    it('should return true when deleting non-existent file', async () => {
      const result = await provider.delete('non-existent.txt');
      expect(result).toBe(true);
    });
  });

  describe('getUrl', () => {
    it('should return correct URL for file', async () => {
      const filePath = 'test/image.jpg';
      const url = await provider.getUrl(filePath);
      
      expect(url).toBe('/test-uploads/test/image.jpg');
    });
  });

  describe('copy', () => {
    it('should copy file successfully', async () => {
      const testData = Buffer.from('test content for copy');
      const sourcePath = 'copy-test/source-copy.txt';
      const destPath = 'copy-test/destination-copy.txt';
      
      // Upload source file
      await provider.upload(testData, sourcePath);
      
      // Verify source exists before copy
      let sourceExists = await provider.exists(sourcePath);
      expect(sourceExists).toBe(true);
      
      // Copy file
      const result = await provider.copy(sourcePath, destPath);
      
      expect(result).toMatchObject({
        path: destPath,
        url: '/test-uploads/copy-test/destination-copy.txt'
      });
      expect(result.size).toBe(testData.length);
      
      // Verify both files exist after copy
      sourceExists = await provider.exists(sourcePath);
      const destExists = await provider.exists(destPath);
      expect(sourceExists).toBe(true);
      expect(destExists).toBe(true);
    });
  });

  describe('move', () => {
    it('should move file successfully', async () => {
      const testData = Buffer.from('test content for move');
      const sourcePath = 'move-test/source-move.txt';
      const destPath = 'move-test/moved/destination.txt';
      
      // Upload source file
      await provider.upload(testData, sourcePath);
      
      // Move file
      const result = await provider.move(sourcePath, destPath);
      
      expect(result).toMatchObject({
        path: destPath,
        url: '/test-uploads/move-test/moved/destination.txt'
      });
      expect(result.size).toBe(testData.length);
      
      // Verify source no longer exists and destination exists
      const sourceExists = await provider.exists(sourcePath);
      const destExists = await provider.exists(destPath);
      expect(sourceExists).toBe(false);
      expect(destExists).toBe(true);
    });
  });

  describe('getMetadata', () => {
    it('should return file metadata', async () => {
      const testData = Buffer.from('test metadata content');
      const filePath = 'metadata-test.txt';
      
      // Upload file
      await provider.upload(testData, filePath);
      
      // Get metadata
      const metadata = await provider.getMetadata(filePath);
      
      expect(metadata).toMatchObject({
        size: testData.length,
        isDirectory: false,
        isFile: true
      });
      expect(metadata.lastModified).toBeInstanceOf(Date);
      expect(metadata.created).toBeInstanceOf(Date);
    });

    it('should throw error for non-existent file', async () => {
      await expect(provider.getMetadata('non-existent.txt'))
        .rejects.toThrow('Failed to get metadata');
    });
  });

  describe('list', () => {
    it('should list files in directory', async () => {
      const files = [
        { path: 'list-test/file1.txt', content: 'content 1' },
        { path: 'list-test/file2.txt', content: 'content 2' },
        { path: 'list-test/subdir/file3.txt', content: 'content 3' }
      ];
      
      // Upload test files
      for (const file of files) {
        await provider.upload(Buffer.from(file.content), file.path);
      }
      
      // List files in directory
      const result = await provider.list('list-test');
      
      expect(result).toHaveLength(3);
      expect(result.map(f => f.name)).toContain('file1.txt');
      expect(result.map(f => f.name)).toContain('file2.txt');
      expect(result.map(f => f.name)).toContain('subdir');
      
      // Check file properties
      const file1 = result.find(f => f.name === 'file1.txt');
      expect(file1).toMatchObject({
        path: 'list-test/file1.txt',
        name: 'file1.txt',
        size: 9,
        isDirectory: false,
        url: '/test-uploads/list-test/file1.txt'
      });
    });

    it('should return empty array for non-existent directory', async () => {
      const result = await provider.list('non-existent-dir');
      expect(result).toEqual([]);
    });
  });
});