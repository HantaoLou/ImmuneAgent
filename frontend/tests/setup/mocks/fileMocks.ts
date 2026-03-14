import { createMockFile as createFile } from '../../utils/mockFactories';

export const mockImageFile = () => createFile({
  name: 'test-image.jpg',
  type: 'image/jpeg',
  size: 2 * 1024 * 1024,
  category: 'image',
});

export const mockDocumentFile = () => createFile({
  name: 'test-document.pdf',
  type: 'application/pdf',
  size: 5 * 1024 * 1024,
  category: 'document',
});

export const mockCodeFile = () => createFile({
  name: 'test-code.ts',
  type: 'text/typescript',
  size: 1 * 1024 * 1024,
  category: 'code',
});

export const mockDataFile = () => createFile({
  name: 'test-data.json',
  type: 'application/json',
  size: 3 * 1024 * 1024,
  category: 'data',
});

export const mockOversizedImageFile = () => createFile({
  name: 'large-image.jpg',
  type: 'image/jpeg',
  size: 6 * 1024 * 1024,
  category: 'image',
});

export const mockInvalidExtensionFile = () => createFile({
  name: 'test.xyz',
  type: 'application/octet-stream',
  size: 1024,
  category: 'other',
});

export const mockFiles = {
  image: mockImageFile(),
  document: mockDocumentFile(),
  code: mockCodeFile(),
  data: mockDataFile(),
  oversized: mockOversizedImageFile(),
  invalidExtension: mockInvalidExtensionFile(),
};
