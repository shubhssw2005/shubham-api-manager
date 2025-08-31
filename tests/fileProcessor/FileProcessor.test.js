import { describe, it, expect, beforeEach } from 'vitest';
import { FileProcessor } from '../../lib/fileProcessor/index.js';
import fs from 'fs/promises';
import path from 'path';

describe('FileProcessor', () => {
  let processor;

  beforeEach(() => {
    processor = new FileProcessor({
      thumbnailSizes: [
        { name: 'small', width: 100, height: 100 },
        { name: 'medium', width: 200, height: 200 }
      ],
      imageQuality: 80,
      maxFileSize: 10 * 1024 * 1024 // 10MB for tests
    });
  });

  describe('validateFile', () => {
    it('should validate file size', async () => {
      const largeBuffer = Buffer.alloc(15 * 1024 * 1024); // 15MB
      
      await expect(processor.validateFile(largeBuffer, {
        mimeType: 'image/jpeg',
        originalName: 'large.jpg'
      })).rejects.toThrow('exceeds maximum allowed size');
    });

    it('should validate allowed file types', async () => {
      const buffer = Buffer.from('test content');
      
      await expect(processor.validateFile(buffer, {
        mimeType: 'application/x-executable',
        originalName: 'malicious.exe'
      })).rejects.toThrow('File type application/x-executable is not allowed');
    });

    it('should block dangerous file extensions', async () => {
      const buffer = Buffer.from('test content');
      
      await expect(processor.validateFile(buffer, {
        mimeType: 'image/jpeg',
        originalName: 'malicious.exe'
      })).rejects.toThrow('File extension .exe is not allowed for security reasons');
    });
  });

  describe('scanForSecurity', () => {
    it('should detect script tags in non-HTML files', async () => {
      const maliciousBuffer = Buffer.from('<script>alert("xss")</script>');
      
      await expect(processor.scanForSecurity(maliciousBuffer, {
        mimeType: 'image/jpeg'
      })).rejects.toThrow('Potentially malicious content detected');
    });

    it('should allow script tags in HTML files', async () => {
      const htmlBuffer = Buffer.from('<html><script>console.log("ok")</script></html>');
      
      // This should not throw since we're not checking HTML files in the current implementation
      await expect(processor.scanForSecurity(htmlBuffer, {
        mimeType: 'text/html'
      })).resolves.not.toThrow();
    });
  });

  describe('processImage', () => {
    it('should process a simple image buffer', async () => {
      // Create a simple 1x1 pixel PNG using sharp
      const sharp = (await import('sharp')).default;
      const simpleImageBuffer = await sharp({
        create: {
          width: 10,
          height: 10,
          channels: 3,
          background: { r: 255, g: 0, b: 0 }
        }
      }).png().toBuffer();

      const result = await processor.processImage(simpleImageBuffer);

      expect(result.type).toBe('image');
      expect(result.metadata).toHaveProperty('width');
      expect(result.metadata).toHaveProperty('height');
      expect(result.metadata).toHaveProperty('format');
      expect(result.thumbnails).toHaveLength(2);
      expect(result.optimized).toHaveProperty('buffer');
      expect(result.optimized).toHaveProperty('size');
      
      // Check thumbnail properties
      result.thumbnails.forEach(thumbnail => {
        expect(thumbnail).toHaveProperty('name');
        expect(thumbnail).toHaveProperty('width');
        expect(thumbnail).toHaveProperty('height');
        expect(thumbnail).toHaveProperty('buffer');
        expect(thumbnail).toHaveProperty('size');
        expect(Buffer.isBuffer(thumbnail.buffer)).toBe(true);
      });
    });
  });

  describe('processDocument', () => {
    it('should process text document', async () => {
      const textContent = 'This is a test document with some content.\nIt has multiple lines.\nAnd some words.';
      const textBuffer = Buffer.from(textContent);

      const result = await processor.processDocument(textBuffer, {
        mimeType: 'text/plain'
      });

      expect(result.type).toBe('document');
      expect(result.metadata.type).toBe('text');
      expect(result.metadata.lineCount).toBe(3);
      // Count words more accurately - the text has 15 words
      expect(result.metadata.wordCount).toBe(15);
      expect(result.metadata.charCount).toBe(textContent.length);
      expect(result.originalSize).toBe(textBuffer.length);
    });

    it('should process PDF document', async () => {
      const pdfBuffer = Buffer.from('fake pdf content');

      const result = await processor.processDocument(pdfBuffer, {
        mimeType: 'application/pdf'
      });

      expect(result.type).toBe('document');
      expect(result.metadata.type).toBe('pdf');
      expect(result.metadata.size).toBe(pdfBuffer.length);
      expect(result.originalSize).toBe(pdfBuffer.length);
    });
  });

  describe('processGenericFile', () => {
    it('should process generic file', async () => {
      const buffer = Buffer.from('generic file content');

      const result = await processor.processGenericFile(buffer);

      expect(result.type).toBe('generic');
      expect(result.metadata.size).toBe(buffer.length);
      expect(result.originalSize).toBe(buffer.length);
    });
  });

  describe('file type detection', () => {
    it('should correctly identify image types', () => {
      expect(processor.isImage('image/jpeg')).toBe(true);
      expect(processor.isImage('image/png')).toBe(true);
      expect(processor.isImage('text/plain')).toBe(false);
    });

    it('should correctly identify video types', () => {
      expect(processor.isVideo('video/mp4')).toBe(true);
      expect(processor.isVideo('video/avi')).toBe(true);
      expect(processor.isVideo('image/jpeg')).toBe(false);
    });

    it('should correctly identify document types', () => {
      expect(processor.isDocument('application/pdf')).toBe(true);
      expect(processor.isDocument('text/plain')).toBe(true);
      expect(processor.isDocument('image/jpeg')).toBe(false);
    });

    it('should correctly identify allowed types', () => {
      expect(processor.isAllowedType('image/jpeg')).toBe(true);
      expect(processor.isAllowedType('video/mp4')).toBe(true);
      expect(processor.isAllowedType('application/pdf')).toBe(true);
      expect(processor.isAllowedType('application/x-executable')).toBe(false);
    });
  });

  describe('processFile', () => {
    it('should route to correct processor based on MIME type', async () => {
      const textBuffer = Buffer.from('test content');

      const result = await processor.processFile(textBuffer, {
        mimeType: 'text/plain',
        originalName: 'test.txt'
      });

      expect(result.type).toBe('document');
      expect(result.metadata.type).toBe('text');
    });

    it('should process generic files for unknown types', async () => {
      // Mock a file type that's allowed but not specifically handled
      processor.config.allowedDocumentTypes.push('application/custom');
      
      const buffer = Buffer.from('custom content');

      const result = await processor.processFile(buffer, {
        mimeType: 'application/custom',
        originalName: 'test.custom'
      });

      expect(result.type).toBe('document');
    });
  });
});