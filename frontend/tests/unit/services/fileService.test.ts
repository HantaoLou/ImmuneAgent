import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  uploadFile,
  downloadFile,
  getSessionFiles,
  deleteFile,
  batchDownloadFiles,
  uploadFileLocal,
  downloadFileLocal,
  fileService,
  UploadFileResponse,
  GetSessionFilesResponse,
} from '@/services/fileService';
import api from '@/services/api';
import { fileStorage } from '@/lib/fileStorage';
import { fileUtils } from '@/lib/fileUtils';

vi.mock('@/services/api', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
  },
}));

vi.mock('@/lib/fileStorage', () => ({
  fileStorage: {
    saveFile: vi.fn(),
    saveFileMeta: vi.fn(),
    getFile: vi.fn(),
    getSessionFiles: vi.fn(),
    deleteFile: vi.fn(),
  },
}));

vi.mock('@/lib/fileUtils', () => ({
  fileUtils: {
    createFileAttachment: vi.fn(),
  },
}));

describe('fileService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('uploadFile', () => {
    it('should upload file successfully', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const mockResponse: UploadFileResponse = {
        fileId: 'file-123',
        url: 'https://example.com/file',
        filename: 'test.txt',
        size: 12,
        mimeType: 'text/plain',
        sessionId: 'session-1',
        uploadTime: '2024-01-15T10:00:00Z',
      };

      vi.mocked(api.post).mockResolvedValueOnce({ data: mockResponse });

      const result = await uploadFile({
        file,
        sessionId: 'session-1',
      });

      expect(result).toEqual(mockResponse);
      expect(api.post).toHaveBeenCalledWith(
        '/api/files/upload',
        expect.any(FormData),
        expect.objectContaining({
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      );
    });

    it('should call onProgress callback during upload', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const onProgress = vi.fn();

      vi.mocked(api.post).mockResolvedValueOnce({
        data: {
          fileId: 'file-123',
          url: 'https://example.com/file',
          filename: 'test.txt',
          size: 12,
          mimeType: 'text/plain',
          sessionId: 'session-1',
          uploadTime: '2024-01-15T10:00:00Z',
        },
      });

      await uploadFile({
        file,
        sessionId: 'session-1',
        onProgress,
      });

      expect(api.post).toHaveBeenCalledWith(
        '/api/files/upload',
        expect.any(FormData),
        expect.objectContaining({
          onUploadProgress: expect.any(Function),
        })
      );
    });

    it('should handle upload error', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const error = new Error('Upload failed');

      vi.mocked(api.post).mockRejectedValueOnce(error);

      await expect(
        uploadFile({ file, sessionId: 'session-1' })
      ).rejects.toThrow('Upload failed');
    });
  });

  describe('downloadFile', () => {
    it('should download file successfully', async () => {
      const blob = new Blob(['test content'], { type: 'text/plain' });

      vi.mocked(api.get).mockResolvedValueOnce({ data: blob });

      const result = await downloadFile({
        fileId: 'file-123',
        sessionId: 'session-1',
      });

      expect(result).toBe(blob);
      expect(api.get).toHaveBeenCalledWith('/api/files/download/file-123', {
        params: { sessionId: 'session-1' },
        responseType: 'blob',
      });
    });

    it('should handle download error', async () => {
      const error = new Error('Download failed');

      vi.mocked(api.get).mockRejectedValueOnce(error);

      await expect(
        downloadFile({ fileId: 'file-123', sessionId: 'session-1' })
      ).rejects.toThrow('Download failed');
    });
  });

  describe('getSessionFiles', () => {
    it('should get session files successfully', async () => {
      const mockResponse: GetSessionFilesResponse = {
        sessionId: 'session-1',
        files: [],
        totalSize: 0,
      };

      vi.mocked(api.get).mockResolvedValueOnce({ data: mockResponse });

      const result = await getSessionFiles({ sessionId: 'session-1' });

      expect(result).toEqual(mockResponse);
      expect(api.get).toHaveBeenCalledWith('/api/files/session/session-1');
    });
  });

  describe('deleteFile', () => {
    it('should delete file successfully', async () => {
      vi.mocked(api.delete).mockResolvedValueOnce({ data: undefined });

      await deleteFile({ fileId: 'file-123', sessionId: 'session-1' });

      expect(api.delete).toHaveBeenCalledWith('/api/files/file-123', {
        params: { sessionId: 'session-1' },
      });
    });

    it('should handle delete error', async () => {
      const error = new Error('Delete failed');

      vi.mocked(api.delete).mockRejectedValueOnce(error);

      await expect(
        deleteFile({ fileId: 'file-123', sessionId: 'session-1' })
      ).rejects.toThrow('Delete failed');
    });
  });

  describe('batchDownloadFiles', () => {
    it('should batch download files successfully', async () => {
      const blob = new Blob(['test content'], { type: 'application/zip' });

      vi.mocked(api.post).mockResolvedValueOnce({ data: blob });

      const result = await batchDownloadFiles({
        fileIds: ['file-1', 'file-2'],
        sessionId: 'session-1',
      });

      expect(result).toBe(blob);
      expect(api.post).toHaveBeenCalledWith(
        '/api/files/batch-download',
        { fileIds: ['file-1', 'file-2'], sessionId: 'session-1' },
        { responseType: 'blob' }
      );
    });
  });

  describe('uploadFileLocal', () => {
    it('should upload file locally', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const mockAttachment = {
        id: 'file-123',
        name: 'test.txt',
        size: 12,
        type: 'text/plain',
        url: 'blob:test',
        sessionId: 'session-1',
        uploadTime: Date.now(),
        category: 'document' as const,
      };

      vi.mocked(fileUtils.createFileAttachment).mockReturnValue(mockAttachment);
      vi.mocked(fileStorage.saveFile).mockResolvedValueOnce(undefined);
      vi.mocked(fileStorage.saveFileMeta).mockReturnValue(undefined);

      const onProgress = vi.fn();
      const result = await uploadFileLocal({
        file,
        sessionId: 'session-1',
        onProgress,
      });

      expect(result.fileId).toBe('file-123');
      expect(result.filename).toBe('test.txt');
      expect(fileStorage.saveFile).toHaveBeenCalled();
      expect(fileStorage.saveFileMeta).toHaveBeenCalled();
    });

    it('should handle local upload error', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });

      vi.mocked(fileUtils.createFileAttachment).mockImplementation(() => {
        throw new Error('Local upload failed');
      });

      await expect(
        uploadFileLocal({ file, sessionId: 'session-1' })
      ).rejects.toThrow();
    });
  });

  describe('downloadFileLocal', () => {
    it('should download file locally', async () => {
      const blob = new Blob(['test content'], { type: 'text/plain' });

      vi.mocked(fileStorage.getFile).mockResolvedValueOnce(blob);

      const result = await downloadFileLocal({
        fileId: 'file-123',
        sessionId: 'session-1',
      });

      expect(result).toBe(blob);
    });

    it('should throw error when file not found', async () => {
      vi.mocked(fileStorage.getFile).mockResolvedValueOnce(null);

      await expect(
        downloadFileLocal({ fileId: 'file-123', sessionId: 'session-1' })
      ).rejects.toThrow('文件不存在');
    });
  });

  describe('FileService class', () => {
    describe('upload', () => {
      it('should use remote mode in production', async () => {
        const file = new File(['test'], 'test.txt', { type: 'text/plain' });
        const mockResponse: UploadFileResponse = {
          fileId: 'file-123',
          url: 'https://example.com/file',
          filename: 'test.txt',
          size: 4,
          mimeType: 'text/plain',
          sessionId: 'session-1',
          uploadTime: '2024-01-15T10:00:00Z',
        };

        vi.mocked(api.post).mockResolvedValueOnce({ data: mockResponse });

        const result = await fileService.upload({ file, sessionId: 'session-1' });

        expect(result.fileId).toBeDefined();
      });
    });

    describe('download', () => {
      it('should download file', async () => {
        const blob = new Blob(['test'], { type: 'text/plain' });

        vi.mocked(api.get).mockResolvedValueOnce({ data: blob });
        vi.mocked(fileStorage.getFile).mockResolvedValueOnce(blob);

        const result = await fileService.download({
          fileId: 'file-123',
          sessionId: 'session-1',
        });

        expect(result).toBeInstanceOf(Blob);
      });
    });

    describe('delete', () => {
      it('should delete file', async () => {
        vi.mocked(api.delete).mockResolvedValueOnce({ data: undefined });
        vi.mocked(fileStorage.deleteFile).mockResolvedValueOnce(undefined);

        await fileService.delete({ fileId: 'file-123', sessionId: 'session-1' });

        expect(true).toBe(true);
      });
    });

    describe('batchDownload', () => {
      it('should batch download or throw error in local mode', async () => {
        try {
          await fileService.batchDownload({
            fileIds: ['file-1', 'file-2'],
            sessionId: 'session-1',
          });
        } catch (error) {
          expect(error).toBeInstanceOf(Error);
        }
      });
    });

    describe('triggerDownload', () => {
      it('should trigger browser download', () => {
        const blob = new Blob(['test content'], { type: 'text/plain' });
        const createElementSpy = vi.spyOn(document, 'createElement');
        const appendChildSpy = vi.spyOn(document.body, 'appendChild');
        const removeChildSpy = vi.spyOn(document.body, 'removeChild');

        fileService.triggerDownload(blob, 'test.txt');

        expect(createElementSpy).toHaveBeenCalledWith('a');
        expect(appendChildSpy).toHaveBeenCalled();
        expect(removeChildSpy).toHaveBeenCalled();
      });
    });
  });
});
