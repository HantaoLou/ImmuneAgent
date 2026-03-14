import '@testing-library/jest-dom/vitest';
import { afterEach, vi, beforeAll } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

class MockDataTransferItemList {
  private _items: { kind: string; type: string; _file: File | null }[] = [];

  get length() {
    return this._items.length;
  }

  add(file: File): void;
  add(data: string, type: string): void;
  add(data: File | string, type?: string): void {
    if (data instanceof File) {
      this._items.push({ kind: 'file', type: data.type, _file: data });
    } else if (typeof data === 'string' && type) {
      this._items.push({ kind: 'string', type, _file: null });
    }
  }

  remove(_index: number): void {
    this._items.splice(_index, 1);
  }

  clear(): void {
    this._items = [];
  }
}

class MockFileList extends Array<File> {
  item(index: number): File | null {
    return this[index] || null;
  }
}

class MockDataTransfer {
  items: MockDataTransferItemList;
  files: MockFileList;
  private _data: Record<string, string> = {};

  constructor() {
    this.items = new MockDataTransferItemList();
    this.files = new MockFileList() as MockFileList;
  }

  add(file: File): void;
  add(data: string, type: string): void;
  add(data: File | string, type?: string): void {
    if (data instanceof File) {
      this.items.add(data);
      this.files.push(data);
    } else if (typeof data === 'string' && type) {
      this.items.add(data, type);
    }
  }

  clearData(): void {
    this._data = {};
    this.files = new MockFileList() as MockFileList;
    this.items = new MockDataTransferItemList();
  }

  getData(format: string): string {
    return this._data[format] || '';
  }

  setData(format: string, data: string): void {
    this._data[format] = data;
  }

  setDragImage(): void {}
}

global.DataTransfer = MockDataTransfer as any;

const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
global.localStorage = localStorageMock as any;

global.URL.createObjectURL = vi.fn(() => 'blob:test');
global.URL.revokeObjectURL = vi.fn();

global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

beforeAll(async () => {
  await import('fake-indexeddb/auto');
});
