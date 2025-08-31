import { describe, it, expect, beforeEach } from 'vitest';
import { FileValidator } from '../../lib/fileProcessor/index.js';

describe('FileValidator', () => {
  let validator;

  beforeEach(() => {
    validator = new FileValidator({
      maxFileSize: 5 * 1024 * 1024, // 5MB for tests
      maxFilenameLength: 100
    });
  });

  describe('validateFile', () => {
    it('should validate a good file', async () => {
      const file = {
        buffer: Buffer.from('test content'),
        originalName: 'test.jpg',
        mimeType: 'image/jpeg'
      };

      const result = await validator.validateFile(file);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.fileInfo).toMatchObject({
        originalName: 'test.jpg',
        mimeType: 'image/jpeg',
        size: 12,
        extension: '.jpg'
      });
    });

    it('should reject files that are too large', async () => {
      const file = {
        buffer: Buffer.alloc(6 * 1024 * 1024), // 6MB
        originalName: 'large.jpg',
        mimeType: 'image/jpeg'
      };

      const result = await validator.validateFile(file);

      expect(result.isValid).toBe(false);
      expect(result.errors.some(error => error.includes('exceeds maximum allowed size'))).toBe(true);
    });

    it('should reject empty files', async () => {
      const file = {
        buffer: Buffer.alloc(0),
        originalName: 'empty.jpg',
        mimeType: 'image/jpeg'
      };

      const result = await validator.validateFile(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('File is empty');
    });

    it('should reject files with no name', async () => {
      const file = {
        buffer: Buffer.from('test'),
        originalName: null,
        mimeType: 'image/jpeg'
      };

      const result = await validator.validateFile(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Filename is required');
    });
  });

  describe('validateFileSize', () => {
    it('should accept files within size limit', () => {
      const file = { buffer: Buffer.from('small content') };
      const result = validator.validateFileSize(file);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject files exceeding size limit', () => {
      const file = { buffer: Buffer.alloc(6 * 1024 * 1024) };
      const result = validator.validateFileSize(file);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('exceeds maximum allowed size');
    });
  });

  describe('validateFilename', () => {
    it('should accept valid filenames', () => {
      const result = validator.validateFilename('document.pdf');

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject filenames that are too long', () => {
      const longName = 'a'.repeat(150) + '.txt';
      const result = validator.validateFilename(longName);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('Filename is too long');
    });

    it('should reject filenames with dangerous characters', () => {
      const result = validator.validateFilename('file<script>.txt');

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('contains invalid characters');
    });

    it('should reject reserved system names', () => {
      const result = validator.validateFilename('CON.txt');

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('reserved system name');
    });

    it('should reject blocked extensions', () => {
      const result = validator.validateFilename('malicious.exe');

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('not allowed for security reasons');
    });

    it('should warn about hidden files', () => {
      const result = validator.validateFilename('.hidden');

      expect(result.isValid).toBe(true);
      expect(result.warnings[0]).toContain('Hidden files');
    });

    it('should warn about filenames with spaces', () => {
      const result = validator.validateFilename('file with spaces.txt');

      expect(result.isValid).toBe(true);
      expect(result.warnings[0]).toContain('spaces may cause issues');
    });
  });

  describe('validateFileType', () => {
    it('should accept allowed MIME types', () => {
      const file = {
        mimeType: 'image/jpeg',
        originalName: 'photo.jpg'
      };

      const result = validator.validateFileType(file);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject disallowed MIME types', () => {
      const file = {
        mimeType: 'application/x-malware',
        originalName: 'malware.exe'
      };

      const result = validator.validateFileType(file);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('is not allowed');
    });

    it('should detect MIME type from filename if not provided', () => {
      const file = {
        originalName: 'document.pdf'
      };

      const result = validator.validateFileType(file);

      expect(result.isValid).toBe(true);
    });

    it('should warn about MIME type mismatch', () => {
      const file = {
        mimeType: 'image/jpeg',
        originalName: 'document.pdf'
      };

      const result = validator.validateFileType(file);

      expect(result.warnings[0]).toContain("doesn't match detected MIME type");
    });
  });

  describe('validateSecurity', () => {
    it('should detect script tags in files', async () => {
      const file = {
        buffer: Buffer.from('<script>alert("xss")</script>'),
        mimeType: 'text/plain'
      };

      const result = await validator.validateSecurity(file);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('potentially malicious content');
    });

    it('should detect javascript: URLs', async () => {
      const file = {
        buffer: Buffer.from('javascript:alert("xss")'),
        mimeType: 'text/plain'
      };

      const result = await validator.validateSecurity(file);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('potentially malicious content');
    });

    it('should detect executable signatures', async () => {
      const file = {
        buffer: Buffer.from('MZ\x90\x00'), // Windows executable signature
        mimeType: 'application/octet-stream'
      };

      const result = await validator.validateSecurity(file);

      expect(result.isValid).toBe(false);
      expect(result.errors[0]).toContain('Windows executable');
    });

    it('should warn about ZIP archives', async () => {
      const file = {
        buffer: Buffer.from([0x50, 0x4B]), // ZIP signature 'PK' as bytes
        mimeType: 'application/zip'
      };

      const result = await validator.validateSecurity(file);

      expect(result.isValid).toBe(true);
      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.warnings.some(warning => warning.includes('ZIP archive'))).toBe(true);
    });
  });

  describe('utility methods', () => {
    it('should format file sizes correctly', () => {
      expect(validator.formatFileSize(0)).toBe('0 Bytes');
      expect(validator.formatFileSize(1024)).toBe('1 KB');
      expect(validator.formatFileSize(1024 * 1024)).toBe('1 MB');
      expect(validator.formatFileSize(1536)).toBe('1.5 KB');
    });

    it('should return allowed types', () => {
      const types = validator.getAllowedTypes();
      expect(types).toContain('image/jpeg');
      expect(types).toContain('application/pdf');
    });

    it('should return blocked extensions', () => {
      const extensions = validator.getBlockedExtensions();
      expect(extensions).toContain('.exe');
      expect(extensions).toContain('.bat');
    });

    it('should check if type is allowed', () => {
      expect(validator.isTypeAllowed('image/jpeg')).toBe(true);
      expect(validator.isTypeAllowed('application/x-malware')).toBe(false);
    });

    it('should check if extension is blocked', () => {
      expect(validator.isExtensionBlocked('.exe')).toBe(true);
      expect(validator.isExtensionBlocked('exe')).toBe(true);
      expect(validator.isExtensionBlocked('.jpg')).toBe(false);
    });
  });
});