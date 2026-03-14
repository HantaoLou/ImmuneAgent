import { FileAttachment } from '@/types';

export const fixtures = {
  imageFile: {
    id: 'file-image-1',
    name: 'test-image.jpg',
    size: 2 * 1024 * 1024,
    type: 'image/jpeg',
    url: 'blob:test-image',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'image' as const,
    uploadProgress: 100,
  },
  
  documentFile: {
    id: 'file-document-1',
    name: 'test-document.pdf',
    size: 5 * 1024 * 1024,
    type: 'application/pdf',
    url: 'blob:test-document',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'document' as const,
    uploadProgress: 100,
  },
  
  codeFile: {
    id: 'file-code-1',
    name: 'test-code.ts',
    size: 1 * 1024 * 1024,
    type: 'text/typescript',
    url: 'blob:test-code',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'code' as const,
    uploadProgress: 100,
  },
  
  dataFile: {
    id: 'file-data-1',
    name: 'test-data.json',
    size: 3 * 1024 * 1024,
    type: 'application/json',
    url: 'blob:test-data',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'data' as const,
    uploadProgress: 100,
  },
  
  uploadingFile: {
    id: 'file-uploading-1',
    name: 'uploading.txt',
    size: 1 * 1024 * 1024,
    type: 'text/plain',
    url: 'blob:uploading',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'document' as const,
    uploadProgress: 45,
  },
  
  largeFile: {
    id: 'file-large-1',
    name: 'large-file.pdf',
    size: 15 * 1024 * 1024,
    type: 'application/pdf',
    url: 'blob:large',
    sessionId: 'session-active',
    uploadTime: Date.now(),
    category: 'document' as const,
    uploadProgress: 100,
  },
};

export const fileList: FileAttachment[] = [
  fixtures.imageFile,
  fixtures.documentFile,
  fixtures.codeFile,
  fixtures.dataFile,
];
