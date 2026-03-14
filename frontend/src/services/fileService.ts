import api from './api';
import { FileAttachment } from '@/types';
import { fileStorage } from '@/lib/fileStorage';
import { fileUtils } from '@/lib/fileUtils';

/**
 * 文件上传 API
 * POST /api/files/upload
 */
export interface UploadFileRequest {
  file: File;
  sessionId: string;
  onProgress?: (progress: number) => void;
}

export interface UploadFileResponse {
  fileId: string;
  url: string;
  filename: string;
  size: number;
  mimeType: string;
  sessionId: string;
  uploadTime: string;
}

export const uploadFile = async (params: UploadFileRequest): Promise<UploadFileResponse> => {
  const formData = new FormData();
  formData.append('file', params.file);
  formData.append('sessionId', params.sessionId);

  const response = await api.post<UploadFileResponse>('/api/files/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && params.onProgress) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        params.onProgress(progress);
      }
    },
  });

  return response.data;
};

/**
 * 文件下载 API
 * GET /api/files/download/:fileId
 */
export interface DownloadFileRequest {
  fileId: string;
  sessionId: string;
}

export const downloadFile = async (params: DownloadFileRequest): Promise<Blob> => {
  const response = await api.get<Blob>(`/api/files/download/${params.fileId}`, {
    params: { sessionId: params.sessionId },
    responseType: 'blob',
  });

  return response.data;
};

/**
 * 获取会话文件列表 API
 * GET /api/files/session/:sessionId
 */
export interface GetSessionFilesRequest {
  sessionId: string;
}

export interface GetSessionFilesResponse {
  sessionId: string;
  files: FileAttachment[];
  totalSize: number;
}

export const getSessionFiles = async (params: GetSessionFilesRequest): Promise<GetSessionFilesResponse> => {
  const response = await api.get<GetSessionFilesResponse>(`/api/files/session/${params.sessionId}`);
  return response.data;
};

/**
 * 删除文件 API
 * DELETE /api/files/:fileId
 */
export interface DeleteFileRequest {
  fileId: string;
  sessionId: string;
}

export const deleteFile = async (params: DeleteFileRequest): Promise<void> => {
  await api.delete(`/api/files/${params.fileId}`, {
    params: { sessionId: params.sessionId },
  });
};

/**
 * 批量下载文件 API
 * POST /api/files/batch-download
 */
export interface BatchDownloadRequest {
  fileIds: string[];
  sessionId: string;
}

export const batchDownloadFiles = async (params: BatchDownloadRequest): Promise<Blob> => {
  const response = await api.post<Blob>('/api/files/batch-download', params, {
    responseType: 'blob',
  });

  return response.data;
};

/**
 * 本地文件上传（开发模式，不调用后端）
 */
export const uploadFileLocal = async (params: UploadFileRequest): Promise<UploadFileResponse> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = async () => {
      try {
        const blob = new Blob([reader.result as ArrayBuffer], { type: params.file.type });
        const url = URL.createObjectURL(blob);
        
        const fileAttachment = fileUtils.createFileAttachment(
          params.file,
          params.sessionId,
          url
        );

        await fileStorage.saveFile(fileAttachment, blob);
        fileStorage.saveFileMeta(fileAttachment);

        if (params.onProgress) {
          params.onProgress(100);
        }

        resolve({
          fileId: fileAttachment.id,
          url: fileAttachment.url,
          filename: fileAttachment.name,
          size: fileAttachment.size,
          mimeType: fileAttachment.type,
          sessionId: fileAttachment.sessionId,
          uploadTime: new Date(fileAttachment.uploadTime).toISOString(),
        });
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => reject(reader.error);

    // 模拟进度
    if (params.onProgress) {
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        params.onProgress!(Math.min(progress, 90));
        if (progress >= 90) clearInterval(interval);
      }, 50);
    }

    reader.readAsArrayBuffer(params.file);
  });
};

/**
 * 本地文件下载（开发模式）
 */
export const downloadFileLocal = async (params: DownloadFileRequest): Promise<Blob> => {
  const blob = await fileStorage.getFile(params.fileId);
  if (!blob) {
    throw new Error('文件不存在');
  }
  return blob;
};

/**
 * 文件服务类（统一管理本地/远程模式）
 */
class FileService {
  private useLocalMode: boolean;

  constructor() {
    this.useLocalMode = process.env.NODE_ENV === 'development' || 
                        process.env.NEXT_PUBLIC_USE_LOCAL_FILES === 'true';
  }

  async upload(params: UploadFileRequest): Promise<UploadFileResponse> {
    if (this.useLocalMode) {
      return uploadFileLocal(params);
    }
    return uploadFile(params);
  }

  async download(params: DownloadFileRequest): Promise<Blob> {
    if (this.useLocalMode) {
      return downloadFileLocal(params);
    }
    return downloadFile(params);
  }

  async getSessionFiles(params: GetSessionFilesRequest): Promise<GetSessionFilesResponse> {
    if (this.useLocalMode) {
      const sessionFiles = fileStorage.getSessionFiles(params.sessionId);
      return sessionFiles;
    }
    return getSessionFiles(params);
  }

  async delete(params: DeleteFileRequest): Promise<void> {
    if (this.useLocalMode) {
      await fileStorage.deleteFile(params.fileId);
      return;
    }
    return deleteFile(params);
  }

  async batchDownload(params: BatchDownloadRequest): Promise<Blob> {
    if (this.useLocalMode) {
      throw new Error('本地模式暂不支持批量下载');
    }
    return batchDownloadFiles(params);
  }

  // 触发浏览器下载
  triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

export const fileService = new FileService();
