import { vi } from 'vitest';

export const waitFor = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const mockNetworkError = () => {
  const error = new Error('Network Error');
  (error as any).code = 'ERR_NETWORK';
  throw error;
};

export const mockTimeoutError = () => {
  const error = new Error('timeout of 10000ms exceeded');
  (error as any).code = 'ECONNABORTED';
  throw error;
};

export const mockServerError = (status: number = 500) => {
  const error = new Error(`Request failed with status code ${status}`);
  (error as any).response = { status, data: { message: 'Server Error' } };
  throw error;
};

export const createMockFile = (
  content: string = 'test content',
  name: string = 'test.txt',
  type: string = 'text/plain'
): File => {
  return new File([content], name, { type });
};

export const flushPromises = () => new Promise(resolve => setTimeout(resolve, 0));

export const mockLocalStorage = () => {
  const store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      Object.keys(store).forEach(key => delete store[key]);
    }),
  };
};
