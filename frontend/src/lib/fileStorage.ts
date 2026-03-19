import { FileAttachment, SessionFiles } from '@/types';

const DB_NAME = 'agent-chat-files';
const DB_VERSION = 1;
const STORE_NAME = 'files';

class FileStorageManager {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    
    this.initPromise = new Promise((resolve, reject) => {
      if (typeof window === 'undefined') {
        resolve();
        return;
      }

      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
    });

    return this.initPromise;
  }

  async saveFile(file: FileAttachment, blob: Blob): Promise<void> {
    await this.init();
    
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      
      const record = {
        ...file,
        blob,
      };

      const request = store.put(record);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getFile(fileId: string): Promise<Blob | null> {
    await this.init();
    
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(fileId);

      request.onsuccess = () => {
        const record = request.result;
        resolve(record?.blob || null);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async deleteFile(fileId: string): Promise<void> {
    await this.init();
    
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(fileId);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async deleteSessionFiles(sessionId: string): Promise<void> {
    await this.init();
    
    const files = this.getSessionFilesMeta(sessionId);
    const deletePromises = (await files).map(f => this.deleteFile(f.id));
    await Promise.all(deletePromises);
    
    // Clean up localStorage
    const meta = this.getAllFilesMeta();
    const filtered = meta.filter(f => f.sessionId !== sessionId);
    localStorage.setItem('file-metadata', JSON.stringify(filtered));
  }

  saveFileMeta(file: FileAttachment): void {
    const meta = this.getAllFilesMeta();
    meta.push(file);
    localStorage.setItem('file-metadata', JSON.stringify(meta));
  }

  getAllFilesMeta(): FileAttachment[] {
    if (typeof window === 'undefined') return [];
    const data = localStorage.getItem('file-metadata');
    return data ? JSON.parse(data) : [];
  }

  async getSessionFilesMeta(sessionId: string): Promise<FileAttachment[]> {
    return this.getAllFilesMeta().filter(f => f.sessionId === sessionId);
  }

  getSessionFiles(sessionId: string): SessionFiles {
    const files = this.getAllFilesMeta().filter(f => f.sessionId === sessionId);
    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    
    return {
      sessionId,
      files,
      totalSize,
    };
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }
}

export const fileStorage = new FileStorageManager();
