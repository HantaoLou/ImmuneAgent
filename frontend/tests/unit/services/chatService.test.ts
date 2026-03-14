import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('@/services/api', () => ({
  default: {
    post: vi.fn(),
  },
}));

import api from '@/services/api';
import { useChatService } from '@/services/chatService';

const mockedApi = vi.mocked(api);

describe('useChatService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('sendChatMessage', () => {
    it('should send message successfully', async () => {
      const mockResponse = {
        data: {
          content: 'Agent response',
          sessionId: 'test-session-id',
        },
      };
      mockedApi.post.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useChatService());

      let response: any;
      await act(async () => {
        response = await result.current.sendChatMessage({
          sessionId: 'test-session-id',
          messages: [{ role: 'user', content: 'Hello' }],
        });
      });

      expect(response).toEqual(mockResponse.data);
      expect(mockedApi.post).toHaveBeenCalledWith('/api/chat', expect.any(Object), expect.any(Object));
    });

    it('should include sessionId in request', async () => {
      mockedApi.post.mockResolvedValueOnce({ data: {} });

      const { result } = renderHook(() => useChatService());

      await act(async () => {
        await result.current.sendChatMessage({
          sessionId: 'my-session-id',
          messages: [],
        });
      });

      expect(mockedApi.post).toHaveBeenCalledWith(
        '/api/chat',
        expect.objectContaining({ sessionId: 'my-session-id' }),
        expect.any(Object)
      );
    });

    it('should include messages in request', async () => {
      mockedApi.post.mockResolvedValueOnce({ data: {} });

      const { result } = renderHook(() => useChatService());

      const messages = [
        { role: 'user' as const, content: 'Hello' },
        { role: 'agent' as const, content: 'Hi' },
      ];

      await act(async () => {
        await result.current.sendChatMessage({
          sessionId: 'test',
          messages,
        });
      });

      expect(mockedApi.post).toHaveBeenCalledWith(
        '/api/chat',
        expect.objectContaining({ messages }),
        expect.any(Object)
      );
    });

    it('should handle network error', async () => {
      const error = new Error('Network Error');
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow('Network Error');
    });

    it('should handle timeout error', async () => {
      const error = new Error('timeout of 10000ms exceeded');
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow('timeout');
    });

    it('should handle 401 error', async () => {
      const error: any = new Error('Request failed with status code 401');
      error.response = { status: 401, data: { message: 'Unauthorized' } };
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow();
    });

    it('should handle 403 error', async () => {
      const error: any = new Error('Request failed with status code 403');
      error.response = { status: 403, data: { message: 'Forbidden' } };
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow();
    });

    it('should handle 404 error', async () => {
      const error: any = new Error('Request failed with status code 404');
      error.response = { status: 404, data: { message: 'Not Found' } };
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow();
    });

    it('should handle 500 error', async () => {
      const error: any = new Error('Request failed with status code 500');
      error.response = { status: 500, data: { message: 'Internal Server Error' } };
      mockedApi.post.mockRejectedValueOnce(error);

      const { result } = renderHook(() => useChatService());

      await expect(
        act(async () => {
          await result.current.sendChatMessage({
            sessionId: 'test',
            messages: [],
          });
        })
      ).rejects.toThrow();
    });
  });

  describe('cancelChatRequest', () => {
    it('should have cancelChatRequest method', () => {
      const { result } = renderHook(() => useChatService());
      expect(result.current.cancelChatRequest).toBeDefined();
      expect(typeof result.current.cancelChatRequest).toBe('function');
    });

    it('should cancel pending request', async () => {
      mockedApi.post.mockImplementation(() => new Promise(() => {}));

      const { result } = renderHook(() => useChatService());

      act(() => {
        result.current.sendChatMessage({
          sessionId: 'test',
          messages: [],
        }).catch(() => {});
      });

      act(() => {
        result.current.cancelChatRequest();
      });

      expect(mockedApi.post).toHaveBeenCalled();
    });
  });

  describe('cleanup', () => {
    it('should have cleanup on unmount', () => {
      const { unmount } = renderHook(() => useChatService());
      unmount();
    });
  });
});
