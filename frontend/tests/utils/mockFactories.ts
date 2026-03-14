import { faker } from '@faker-js/faker';
import { Session, Message, FileAttachment } from '@/types';

faker.seed(123);

export const createMockSession = (overrides?: Partial<Session>): Session => ({
  id: faker.string.uuid(),
  title: faker.lorem.sentence(),
  messages: [],
  createTime: faker.date.recent().getTime(),
  updateTime: faker.date.recent().getTime(),
  ...overrides,
});

export const createMockMessage = (overrides?: Partial<Message>): Message => ({
  id: faker.string.uuid(),
  role: faker.helpers.arrayElement(['user', 'agent']),
  content: faker.lorem.paragraph(),
  timestamp: faker.date.recent().getTime(),
  status: faker.helpers.arrayElement(['success', 'loading', 'error']),
  ...overrides,
});

export const createMockFile = (overrides?: Partial<FileAttachment>): FileAttachment => ({
  id: faker.string.uuid(),
  name: faker.system.fileName(),
  size: faker.number.int({ min: 1024, max: 10 * 1024 * 1024 }),
  type: faker.system.mimeType(),
  url: faker.internet.url(),
  sessionId: faker.string.uuid(),
  uploadTime: faker.date.recent().getTime(),
  category: faker.helpers.arrayElement(['image', 'document', 'code', 'data', 'other']),
  ...overrides,
});

export const createMockSessions = (count: number = 3): Session[] => {
  return Array.from({ length: count }, () => createMockSession());
};

export const createMockMessages = (count: number = 5): Message[] => {
  return Array.from({ length: count }, () => createMockMessage());
};

export const createMockFiles = (count: number = 5): FileAttachment[] => {
  return Array.from({ length: count }, () => createMockFile());
};
