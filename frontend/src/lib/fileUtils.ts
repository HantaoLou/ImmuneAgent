import { FileCategory, FileAttachment } from '@/types';
import { v4 as uuidv4 } from 'uuid';

const FILE_TYPE_MAP: Record<string, FileCategory> = {
  // 图片
  'image/jpeg': 'image',
  'image/jpg': 'image',
  'image/png': 'image',
  'image/gif': 'image',
  'image/svg+xml': 'image',
  'image/webp': 'image',
  
  // 文档
  'application/pdf': 'document',
  'application/msword': 'document',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
  'text/plain': 'document',
  'text/markdown': 'document',
  
  // 代码
  'text/javascript': 'code',
  'text/typescript': 'code',
  'application/javascript': 'code',
  'text/x-python': 'code',
  'text/x-java': 'code',
  'text/x-c': 'code',
  'text/x-cpp': 'code',
  
  // 数据
  'text/csv': 'data',
  'application/json': 'data',
  'application/vnd.ms-excel': 'data',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'data',
  'application/xml': 'data',
  'text/xml': 'data',
};

const FILE_EXTENSION_MAP: Record<string, FileCategory> = {
  // 图片
  '.jpg': 'image',
  '.jpeg': 'image',
  '.png': 'image',
  '.gif': 'image',
  '.svg': 'image',
  '.webp': 'image',
  
  // 文档
  '.pdf': 'document',
  '.doc': 'document',
  '.docx': 'document',
  '.txt': 'document',
  '.md': 'document',
  
  // 代码
  '.js': 'code',
  '.jsx': 'code',
  '.ts': 'code',
  '.tsx': 'code',
  '.py': 'code',
  '.java': 'code',
  '.c': 'code',
  '.cpp': 'code',
  
  // 数据
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
    const maxSizeMap: Record<FileCategory, number> = {
      image: 5 * 1024 * 1024, // 5MB
      document: 10 * 1024 * 1024, // 10MB
      code: 2 * 1024 * 1024, // 2MB
      data: 10 * 1024 * 1024, // 10MB
      other: 5 * 1024 * 1024, // 5MB
    };

    const category = this.getCategory(file);
    const maxSize = maxSizeMap[category];

    if (file.size > maxSize) {
      return {
        valid: false,
        error: `文件大小超出限制（最大 ${this.formatFileSize(maxSize)}）`,
      };
    }

    const allowedExtensions = Object.keys(FILE_EXTENSION_MAP);
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedExtensions.includes(extension)) {
      return {
        valid: false,
        error: '不支持的文件类型',
      };
    }

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
