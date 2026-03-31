import api from './api';
import { FileAttachment, SandboxFile, SandboxFilesResponse } from '@/types';
import { fileStorage } from '@/lib/fileStorage';
import { fileUtils } from '@/lib/fileUtils';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Files larger than this threshold will be uploaded in chunks */
const CHUNKED_UPLOAD_THRESHOLD = 5 * 1024 * 1024; // 5 MB
/** Size of each chunk */
const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extract a human-readable error message from an axios error.
 * FastAPI validation errors arrive as arrays of objects in `response.data.detail`,
 * so we need to handle objects/arrays carefully — String() alone gives "[object Object]".
 */
function extractErrorMessage(err: any): string {
  const raw = err?.response?.data?.detail ?? err?.message ?? err?.code;
  if (raw === undefined || raw === null) return String(err);
  if (typeof raw === 'string') return raw;
  try {
    return JSON.stringify(raw);
  } catch {
    return String(raw);
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * File Upload API
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
  sandboxPath?: string;
  sandboxId?: string;
  chunks?: number;
}

// ---------------------------------------------------------------------------
// Direct (single-request) upload — used for files ≤ threshold
// ---------------------------------------------------------------------------

export const uploadFile = async (params: UploadFileRequest): Promise<UploadFileResponse> => {
  const formData = new FormData();
  formData.append('file', params.file);
  formData.append('sessionId', params.sessionId);

  // NOTE: Do NOT set Content-Type manually — axios auto-detects FormData and
  // sets the correct multipart/form-data header WITH the boundary string.
  // A manual 'multipart/form-data' without boundary causes the server to fail
  // to parse the body, resulting in empty file data.
  const response = await api.post<UploadFileResponse>('/api/files/upload', formData, {
    timeout: 0,
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && params.onProgress) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        params.onProgress(progress);
      }
    },
  });

  return response.data;
};

// ---------------------------------------------------------------------------
// Chunked upload — used for files > threshold
// POST /api/files/upload-chunk  (per chunk)
// POST /api/files/upload-complete  (finalise)
// ---------------------------------------------------------------------------

export const uploadFileChunked = async (
  params: UploadFileRequest,
): Promise<UploadFileResponse> => {
  const { file, sessionId, onProgress } = params;
  const uploadId = uuidv4();
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

  console.info(
    `[uploadChunked] Starting chunked upload: "${file.name}" (${(file.size / 1024 / 1024).toFixed(1)} MB, ${totalChunks} chunks, uploadId=${uploadId})`,
  );

  // 1. Upload each chunk sequentially
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    const start = chunkIndex * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunkBlob = file.slice(start, end);

    const formData = new FormData();
    formData.append('chunk_index', String(chunkIndex));
    formData.append('total_chunks', String(totalChunks));
    formData.append('upload_id', uploadId);
    formData.append('sessionId', sessionId);
    formData.append('filename', file.name);
    formData.append('file', chunkBlob, file.name);

    try {
    await api.post('/api/files/upload-chunk', formData, {
      timeout: 0, // no timeout for chunk uploads
    });
    } catch (err: any) {
      const detail = extractErrorMessage(err);
      console.error(
        `[uploadChunked] Chunk ${chunkIndex + 1}/${totalChunks} failed for "${file.name}" (uploadId=${uploadId}):`,
        detail,
      );
      throw new Error(
        `Chunk ${chunkIndex + 1}/${totalChunks} upload failed: ${detail}`,
      );
    }

    // Report progress based on completed chunks
    if (onProgress) {
      const progress = Math.round(((chunkIndex + 1) / totalChunks) * 100);
      onProgress(progress);
    }
  }

  // 2. Notify backend to assemble chunks
  console.info(`[uploadChunked] All ${totalChunks} chunks uploaded, requesting assembly for "${file.name}"...`);

  try {
    const completeResponse = await api.post<UploadFileResponse>(
      '/api/files/upload-complete',
      {
        upload_id: uploadId,
        session_id: sessionId,
        filename: file.name,
      },
      {
        timeout: 300000, // 5 min — assembly of large files takes time
      },
    );

    console.info(
      `[uploadChunked] Assembly complete: "${file.name}" -> ${completeResponse.data.sandboxPath}`,
    );
    return completeResponse.data;
  } catch (err: any) {
    const detail = extractErrorMessage(err);
    console.error(
      `[uploadChunked] Assembly failed for "${file.name}" (uploadId=${uploadId}, ${totalChunks} chunks uploaded):`,
      detail,
    );
    throw new Error(`File assembly failed: ${detail}`);
  }
};

/**
 * File Download API
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
 * Get Session Files API
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
 * Delete File API
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
 * Batch Download Files API
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
 * Local File Upload (development mode, no backend call)
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

    // Simulate progress
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
 * Local File Download (development mode)
 */
export const downloadFileLocal = async (params: DownloadFileRequest): Promise<Blob> => {
  const blob = await fileStorage.getFile(params.fileId);
  if (!blob) {
    throw new Error('File not found');
  }
  return blob;
};

/**
 * File Service Class (unified management of local/remote modes)
 */
class FileService {
  private useLocalMode: boolean;

  constructor() {
    this.useLocalMode = process.env.NEXT_PUBLIC_USE_LOCAL_FILES === 'true';
  }

  async upload(params: UploadFileRequest): Promise<UploadFileResponse> {
    if (this.useLocalMode) {
      return uploadFileLocal(params);
    }
    // Route to chunked upload for files larger than threshold
    if (params.file.size > CHUNKED_UPLOAD_THRESHOLD) {
      return uploadFileChunked(params);
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
      throw new Error('Batch download not supported in local mode');
    }
    return batchDownloadFiles(params);
  }

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

  async getSandboxFiles(sessionId: string): Promise<SandboxFilesResponse> {
    const response = await api.get<SandboxFilesResponse>(`/api/sessions/${sessionId}/files`, {
      timeout: 0,
    });
    return response.data;
  }

  getSandboxDownloadUrl(sessionId: string, filePath: string): string {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    return `${baseUrl}/api/download/${sessionId}/${filePath}`;
  }

  async downloadSandboxFile(sessionId: string, filePath: string): Promise<Blob> {
    const response = await api.get(`/api/download/${sessionId}/${filePath}`, {
      responseType: 'blob',
      timeout: 120000,
    });
    return response.data;
  }
}

export const fileService = new FileService();
