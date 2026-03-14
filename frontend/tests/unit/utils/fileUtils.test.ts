import { describe, it, expect } from 'vitest';
import { fileUtils } from '@/lib/fileUtils';

describe('fileUtils', () => {
  describe('getCategory', () => {
    describe('by MIME type', () => {
      it('should identify image/jpeg as image', () => {
        const file = new File([''], 'test.jpg', { type: 'image/jpeg' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify image/png as image', () => {
        const file = new File([''], 'test.png', { type: 'image/png' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify image/gif as image', () => {
        const file = new File([''], 'test.gif', { type: 'image/gif' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify image/svg+xml as image', () => {
        const file = new File([''], 'test.svg', { type: 'image/svg+xml' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify application/pdf as document', () => {
        const file = new File([''], 'test.pdf', { type: 'application/pdf' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify application/msword as document', () => {
        const file = new File([''], 'test.doc', { type: 'application/msword' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify text/plain as document', () => {
        const file = new File([''], 'test.txt', { type: 'text/plain' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify text/javascript as code', () => {
        const file = new File([''], 'test.js', { type: 'text/javascript' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify text/typescript as code', () => {
        const file = new File([''], 'test.ts', { type: 'text/typescript' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify text/csv as data', () => {
        const file = new File([''], 'test.csv', { type: 'text/csv' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify application/json as data', () => {
        const file = new File([''], 'test.json', { type: 'application/json' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify application/vnd.ms-excel as data', () => {
        const file = new File([''], 'test.xls', { type: 'application/vnd.ms-excel' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });
    });

    describe('by extension', () => {
      it('should identify .jpg as image', () => {
        const file = new File([''], 'test.jpg', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify .jpeg as image', () => {
        const file = new File([''], 'test.jpeg', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify .png as image', () => {
        const file = new File([''], 'test.png', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify .gif as image', () => {
        const file = new File([''], 'test.gif', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify .svg as image', () => {
        const file = new File([''], 'test.svg', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('image');
      });

      it('should identify .pdf as document', () => {
        const file = new File([''], 'test.pdf', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify .doc as document', () => {
        const file = new File([''], 'test.doc', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify .docx as document', () => {
        const file = new File([''], 'test.docx', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify .txt as document', () => {
        const file = new File([''], 'test.txt', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify .md as document', () => {
        const file = new File([''], 'test.md', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('document');
      });

      it('should identify .js as code', () => {
        const file = new File([''], 'test.js', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .jsx as code', () => {
        const file = new File([''], 'test.jsx', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .ts as code', () => {
        const file = new File([''], 'test.ts', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .tsx as code', () => {
        const file = new File([''], 'test.tsx', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .py as code', () => {
        const file = new File([''], 'test.py', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .java as code', () => {
        const file = new File([''], 'test.java', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .c as code', () => {
        const file = new File([''], 'test.c', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .cpp as code', () => {
        const file = new File([''], 'test.cpp', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('code');
      });

      it('should identify .csv as data', () => {
        const file = new File([''], 'test.csv', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify .json as data', () => {
        const file = new File([''], 'test.json', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify .xls as data', () => {
        const file = new File([''], 'test.xls', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify .xlsx as data', () => {
        const file = new File([''], 'test.xlsx', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });

      it('should identify .xml as data', () => {
        const file = new File([''], 'test.xml', { type: 'application/octet-stream' });
        expect(fileUtils.getCategory(file)).toBe('data');
      });
    });

    it('should return other for unknown types', () => {
      const file = new File([''], 'test.xyz', { type: 'application/octet-stream' });
      expect(fileUtils.getCategory(file)).toBe('other');
    });

    it('should prioritize MIME type over extension', () => {
      const file = new File([''], 'test.txt', { type: 'image/jpeg' });
      expect(fileUtils.getCategory(file)).toBe('image');
    });
  });

  describe('getFileIcon', () => {
    it('should return 🖼️ for image', () => {
      expect(fileUtils.getFileIcon('image')).toBe('🖼️');
    });

    it('should return 📄 for document', () => {
      expect(fileUtils.getFileIcon('document')).toBe('📄');
    });

    it('should return 💻 for code', () => {
      expect(fileUtils.getFileIcon('code')).toBe('💻');
    });

    it('should return 📊 for data', () => {
      expect(fileUtils.getFileIcon('data')).toBe('📊');
    });

    it('should return 📎 for other', () => {
      expect(fileUtils.getFileIcon('other')).toBe('📎');
    });
  });

  describe('getFileColor', () => {
    it('should return cyan for image', () => {
      expect(fileUtils.getFileColor('image')).toBe('var(--accent-cyan)');
    });

    it('should return purple for document', () => {
      expect(fileUtils.getFileColor('document')).toBe('var(--accent-purple)');
    });

    it('should return lime for code', () => {
      expect(fileUtils.getFileColor('code')).toBe('var(--accent-lime)');
    });

    it('should return gold for data', () => {
      expect(fileUtils.getFileColor('data')).toBe('var(--accent-gold)');
    });

    it('should return muted for other', () => {
      expect(fileUtils.getFileColor('other')).toBe('var(--text-muted)');
    });
  });

  describe('validateFile', () => {
    it('should reject oversized image (>5MB)', () => {
      const largeFile = new File(['x'.repeat(6 * 1024 * 1024)], 'large.jpg', { 
        type: 'image/jpeg' 
      });
      const result = fileUtils.validateFile(largeFile);
      
      expect(result.valid).toBe(false);
      expect(result.error).toContain('文件大小超出限制');
    });

    it('should reject oversized document (>10MB)', () => {
      const largeFile = new File(['x'.repeat(11 * 1024 * 1024)], 'large.pdf', { 
        type: 'application/pdf' 
      });
      const result = fileUtils.validateFile(largeFile);
      
      expect(result.valid).toBe(false);
      expect(result.error).toContain('文件大小超出限制');
    });

    it('should reject oversized code file (>2MB)', () => {
      const largeFile = new File(['x'.repeat(3 * 1024 * 1024)], 'large.ts', { 
        type: 'text/typescript' 
      });
      const result = fileUtils.validateFile(largeFile);
      
      expect(result.valid).toBe(false);
      expect(result.error).toContain('文件大小超出限制');
    });

    it('should reject oversized data file (>10MB)', () => {
      const largeFile = new File(['x'.repeat(11 * 1024 * 1024)], 'large.json', { 
        type: 'application/json' 
      });
      const result = fileUtils.validateFile(largeFile);
      
      expect(result.valid).toBe(false);
      expect(result.error).toContain('文件大小超出限制');
    });

    it('should reject unsupported extension', () => {
      const file = new File(['test'], 'test.xyz', { type: 'application/octet-stream' });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(false);
      expect(result.error).toContain('不支持的文件类型');
    });

    it('should accept valid image', () => {
      const file = new File(['x'.repeat(1 * 1024 * 1024)], 'test.jpg', { 
        type: 'image/jpeg' 
      });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept valid document', () => {
      const file = new File(['x'.repeat(5 * 1024 * 1024)], 'test.pdf', { 
        type: 'application/pdf' 
      });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept valid code file', () => {
      const file = new File(['x'.repeat(1 * 1024 * 1024)], 'test.ts', { 
        type: 'text/typescript' 
      });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept valid data file', () => {
      const file = new File(['x'.repeat(5 * 1024 * 1024)], 'test.json', { 
        type: 'application/json' 
      });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept file at exact size limit', () => {
      const file = new File(['x'.repeat(5 * 1024 * 1024)], 'test.jpg', { 
        type: 'image/jpeg' 
      });
      const result = fileUtils.validateFile(file);
      
      expect(result.valid).toBe(true);
    });
  });

  describe('createFileAttachment', () => {
    it('should create with all required fields', () => {
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const sessionId = 'test-session';
      const url = 'blob:test';
      
      const attachment = fileUtils.createFileAttachment(file, sessionId, url);
      
      expect(attachment.id).toBeDefined();
      expect(attachment.name).toBe('test.txt');
      expect(attachment.size).toBe(4);
      expect(attachment.type).toBe('text/plain');
      expect(attachment.url).toBe(url);
      expect(attachment.sessionId).toBe(sessionId);
      expect(attachment.uploadTime).toBeDefined();
      expect(attachment.category).toBe('document');
    });

    it('should generate unique ID', () => {
      const file1 = new File(['test'], 'test.txt', { type: 'text/plain' });
      const file2 = new File(['test'], 'test.txt', { type: 'text/plain' });
      
      const attachment1 = fileUtils.createFileAttachment(file1, 'session1', 'url1');
      const attachment2 = fileUtils.createFileAttachment(file2, 'session1', 'url1');
      
      expect(attachment1.id).not.toBe(attachment2.id);
    });

    it('should set correct category', () => {
      const imageFile = new File([''], 'test.jpg', { type: 'image/jpeg' });
      const attachment = fileUtils.createFileAttachment(imageFile, 'session', 'url');
      
      expect(attachment.category).toBe('image');
    });

    it('should set uploadTime', () => {
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const beforeTime = Date.now();
      const attachment = fileUtils.createFileAttachment(file, 'session', 'url');
      const afterTime = Date.now();
      
      expect(attachment.uploadTime).toBeGreaterThanOrEqual(beforeTime);
      expect(attachment.uploadTime).toBeLessThanOrEqual(afterTime);
    });
  });

  describe('formatFileSize', () => {
    it('should format 0 bytes', () => {
      expect(fileUtils.formatFileSize(0)).toBe('0 B');
    });

    it('should format bytes', () => {
      expect(fileUtils.formatFileSize(512)).toBe('512 B');
    });

    it('should format kilobytes', () => {
      expect(fileUtils.formatFileSize(1024)).toBe('1 KB');
      expect(fileUtils.formatFileSize(1536)).toBe('1.5 KB');
    });

    it('should format megabytes', () => {
      expect(fileUtils.formatFileSize(1048576)).toBe('1 MB');
      expect(fileUtils.formatFileSize(1572864)).toBe('1.5 MB');
    });

    it('should format gigabytes', () => {
      expect(fileUtils.formatFileSize(1073741824)).toBe('1 GB');
    });
  });

  describe('truncateFileName', () => {
    it('should not truncate short names', () => {
      expect(fileUtils.truncateFileName('test.txt')).toBe('test.txt');
    });

    it('should truncate long names', () => {
      const longName = 'very_long_file_name_that_exceeds_limit.txt';
      const truncated = fileUtils.truncateFileName(longName);
      
      expect(truncated.length).toBeLessThanOrEqual(20);
      expect(truncated).toContain('...');
      expect(truncated).toMatch(/\.txt$/);
    });

    it('should preserve extension', () => {
      const longName = 'very_long_file_name_that_exceeds_limit.pdf';
      const truncated = fileUtils.truncateFileName(longName);
      
      expect(truncated).toMatch(/\.pdf$/);
    });

    it('should use default max length 20', () => {
      const longName = 'very_long_file_name_that_exceeds_limit.txt';
      const truncated = fileUtils.truncateFileName(longName);
      
      expect(truncated.length).toBeLessThanOrEqual(20);
      expect(truncated).toContain('...');
    });

    it('should use custom max length', () => {
      const longName = 'very_long_file_name_that_exceeds_limit.txt';
      const truncated = fileUtils.truncateFileName(longName, 30);
      
      expect(truncated.length).toBeLessThanOrEqual(30);
      expect(truncated).toContain('...');
    });
  });
});
