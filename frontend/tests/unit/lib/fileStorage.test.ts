import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { fileStorage } from '@/lib/fileStorage';
import { createMockFile } from '../../utils/mockFactories';

describe('FileStorage', () => {
  beforeEach(async () => {
    await fileStorage.init();
  });

  describe('init', () => {
    it('should initialize IndexedDB', async () => {
      await expect(fileStorage.init()).resolves.not.toThrow();
    });

    it('should be idempotent', async () => {
      await fileStorage.init();
      await fileStorage.init();
    });
  });

  describe('saveFile', () => {
    it('should save file to IndexedDB', async () => {
      const file = createMockFile();
      const blob = new Blob(['test content']);
      
      await expect(fileStorage.saveFile(file, blob)).resolves.not.toThrow();
    });
  });

  describe('getFile', () => {
    it('should retrieve existing file', async () => {
      const file = createMockFile({ id: 'test-file-id' });
      const content = 'test content';
      const blob = new Blob([content]);
      
      await fileStorage.saveFile(file, blob);
      const retrieved = await fileStorage.getFile('test-file-id');
      
      expect(retrieved).toBeDefined();
      expect(retrieved).toBeInstanceOf(Blob);
    });

    it('should return null for non-existent file', async () => {
      const retrieved = await fileStorage.getFile('non-existent-id');
      expect(retrieved).toBeNull();
    });
  });

  describe('deleteFile', () => {
    it('should delete file from IndexedDB', async () => {
      const file = createMockFile({ id: 'test-file-id' });
      const blob = new Blob(['test']);
      
      await fileStorage.saveFile(file, blob);
      await fileStorage.deleteFile('test-file-id');
      
      const retrieved = await fileStorage.getFile('test-file-id');
      expect(retrieved).toBeNull();
    });
  });

  describe('formatFileSize', () => {
    it('should format 0 bytes', () => {
      expect(fileStorage.formatFileSize(0)).toBe('0 B');
    });

    it('should format bytes', () => {
      expect(fileStorage.formatFileSize(512)).toBe('512 B');
    });

    it('should format kilobytes', () => {
      expect(fileStorage.formatFileSize(1024)).toBe('1 KB');
      expect(fileStorage.formatFileSize(1536)).toBe('1.5 KB');
    });

    it('should format megabytes', () => {
      expect(fileStorage.formatFileSize(1048576)).toBe('1 MB');
      expect(fileStorage.formatFileSize(1572864)).toBe('1.5 MB');
    });

    it('should format gigabytes', () => {
      expect(fileStorage.formatFileSize(1073741824)).toBe('1 GB');
    });
  });

  describe('metadata operations', () => {
    it('should have saveFileMeta method', () => {
      expect(fileStorage.saveFileMeta).toBeDefined();
      expect(typeof fileStorage.saveFileMeta).toBe('function');
    });

    it('should have getAllFilesMeta method', () => {
      expect(fileStorage.getAllFilesMeta).toBeDefined();
      expect(typeof fileStorage.getAllFilesMeta).toBe('function');
    });

    it('should have getSessionFilesMeta method', () => {
      expect(fileStorage.getSessionFilesMeta).toBeDefined();
      expect(typeof fileStorage.getSessionFilesMeta).toBe('function');
    });

    it('should have getSessionFiles method', () => {
      expect(fileStorage.getSessionFiles).toBeDefined();
      expect(typeof fileStorage.getSessionFiles).toBe('function');
    });
  });
});
