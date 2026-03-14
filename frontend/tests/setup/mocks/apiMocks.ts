import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';

export const createMockApi = () => {
  const mock = new MockAdapter(axios);
  
  mock.onPost('/api/chat').reply(200, {
    content: 'Mock agent response',
    sessionId: 'test-session-id',
  });
  
  mock.onPost('/api/files/upload').reply(200, {
    fileId: 'test-file-id',
    url: 'blob:test',
    filename: 'test.txt',
    size: 1024,
    mimeType: 'text/plain',
    sessionId: 'test-session-id',
    uploadTime: new Date().toISOString(),
  });
  
  mock.onGet(/\/api\/files\/download\/.+/).reply(200, new Blob(['test content']));
  
  mock.onGet(/\/api\/files\/session\/.+/).reply(200, {
    sessionId: 'test-session-id',
    files: [],
    totalSize: 0,
  });
  
  mock.onDelete(/\/api\/files\/.+/).reply(200);
  
  mock.onPost('/api/files/batch-download').reply(200, new Blob(['batch content']));
  
  return mock;
};

export const mockChatSuccess = (mock: MockAdapter, response?: any) => {
  mock.onPost('/api/chat').reply(200, response || {
    content: 'Success response',
    sessionId: 'test-session-id',
  });
};

export const mockChatError = (mock: MockAdapter, status: number = 500, message?: string) => {
  mock.onPost('/api/chat').reply(status, {
    message: message || 'Error message',
  });
};

export const mockChatTimeout = (mock: MockAdapter) => {
  mock.onPost('/api/chat').timeout();
};

export const mockNetworkError = (mock: MockAdapter) => {
  mock.onPost('/api/chat').networkError();
};
