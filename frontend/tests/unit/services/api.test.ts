import { describe, it, expect, vi } from 'vitest';
import api from '@/services/api';

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    })),
  },
}));

describe('API Client', () => {
  describe('configuration', () => {
    it('should create axios instance', () => {
      expect(api).toBeDefined();
      expect(api.get).toBeDefined();
      expect(api.post).toBeDefined();
      expect(api.put).toBeDefined();
      expect(api.delete).toBeDefined();
    });

    it('should have interceptors', () => {
      expect(api.interceptors).toBeDefined();
      expect(api.interceptors.request).toBeDefined();
      expect(api.interceptors.response).toBeDefined();
    });
  });

  describe('HTTP methods', () => {
    it('should have GET method', () => {
      expect(typeof api.get).toBe('function');
    });

    it('should have POST method', () => {
      expect(typeof api.post).toBe('function');
    });

    it('should have PUT method', () => {
      expect(typeof api.put).toBe('function');
    });

    it('should have DELETE method', () => {
      expect(typeof api.delete).toBe('function');
    });
  });

  describe('Interceptors', () => {
    it('should have request interceptor', () => {
      expect(api.interceptors.request.use).toBeDefined();
      expect(typeof api.interceptors.request.use).toBe('function');
    });

    it('should have response interceptor', () => {
      expect(api.interceptors.response.use).toBeDefined();
      expect(typeof api.interceptors.response.use).toBe('function');
    });
  });
});
