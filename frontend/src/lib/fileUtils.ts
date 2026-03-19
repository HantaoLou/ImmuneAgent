import { FileCategory, FileAttachment } from '@/types';
import { v4 as uuidv4 } from 'uuid';

const FILE_TYPE_MAP: Record<string, FileCategory> = {
  // Image
  'image/jpeg': 'image',
  'image/jpg': 'image',
  'image/png': 'image',
  'image/gif': 'image',
  'image/svg+xml': 'image',
  'image/webp': 'image',
  
  // Document
  'application/pdf': 'document',
  'application/msword': 'document',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
  'text/plain': 'document',
  'text/markdown': 'document',
  
  // Code
  'text/javascript': 'code',
  'text/typescript': 'code',
  'application/javascript': 'code',
  'text/x-python': 'code',
  'text/x-java': 'code',
  'text/x-c': 'code',
  'text/x-cpp': 'code',
  
  // Data
  'text/csv': 'data',
  'application/json': 'data',
  'application/vnd.ms-excel': 'data',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'data',
  'application/xml': 'data',
  'text/xml': 'data',
};

const FILE_EXTENSION_MAP: Record<string, FileCategory> = {
  // Image
  '.jpg': 'image',
  '.jpeg': 'image',
  '.png': 'image',
  '.gif': 'image',
  '.svg': 'image',
  '.webp': 'image',
  
  // Document
  '.pdf': 'document',
  '.doc': 'document',
  '.docx': 'document',
  '.txt': 'document',
  '.md': 'document',
  
  // Code
  '.js': 'code',
  '.jsx': 'code',
  '.ts': 'code',
  '.tsx': 'code',
  '.py': 'code',
  '.java': 'code',
  '.c': 'code',
  '.cpp': 'code',
  
  // Data
  '.csv': 'data',
  '.json': 'data',
  '.xls': 'data',
  '.xlsx': 'data',
  '.xml': 'data',
};

export const fileUtils = {
  getCategory(file: File): FileCategory {
    const mimeType = file.type.toLowerCase();
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    return FILE_TYPE_MAP[mimeType] || FILE_EXTENSION_MAP[extension] || 'other';
  },

  getFileIcon(category: FileCategory): string {
    const iconMap: Record<FileCategory, string> = {
      image: '🖼️',
      document: '📄',
      code: '💻',
      data: '📊',
      other: '📎',
    };
    return iconMap[category];
  },

  getFileColor(category: FileCategory): string {
    const colorMap: Record<FileCategory, string> = {
      image: 'var(--accent-cyan)',
      document: 'var(--accent-purple)',
      code: 'var(--accent-lime)',
      data: 'var(--accent-gold)',
      other: 'var(--text-muted)',
    };
    return colorMap[category];
  },

  validateFile(file: File): { valid: boolean; error?: string } {
    return { valid: true };
  },

  createFileAttachment(
    file: File,
    sessionId: string,
    url: string
  ): FileAttachment {
    return {
      id: uuidv4(),
      name: file.name,
      size: file.size,
      type: file.type,
      url,
      sessionId,
      uploadTime: Date.now(),
      category: this.getCategory(file),
    };
  },

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  },

  truncateFileName(name: string, maxLength: number = 20): string {
    if (name.length <= maxLength) return name;
    const extension = name.split('.').pop();
    const baseName = name.substring(0, name.lastIndexOf('.'));
    const truncatedBase = baseName.substring(0, maxLength - extension!.length - 4);
    return `${truncatedBase}...${extension}`;
  },
};
