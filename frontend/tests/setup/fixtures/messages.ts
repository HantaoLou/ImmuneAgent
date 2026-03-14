import { Message } from '@/types';

export const fixtures = {
  userMessage: {
    id: 'msg-user-1',
    role: 'user' as const,
    content: 'Hello, this is a test message',
    timestamp: Date.now(),
    status: 'success' as const,
  },
  
  agentMessage: {
    id: 'msg-agent-1',
    role: 'agent' as const,
    content: 'Hello! I am the agent response',
    timestamp: Date.now(),
    status: 'success' as const,
  },
  
  loadingMessage: {
    id: 'msg-loading-1',
    role: 'agent' as const,
    content: '',
    timestamp: Date.now(),
    status: 'loading' as const,
  },
  
  errorMessage: {
    id: 'msg-error-1',
    role: 'agent' as const,
    content: 'An error occurred',
    timestamp: Date.now(),
    status: 'error' as const,
  },
  
  longMessage: {
    id: 'msg-long-1',
    role: 'user' as const,
    content: 'This is a very long message that exceeds the normal length limit. '.repeat(100),
    timestamp: Date.now(),
    status: 'success' as const,
  },
  
  messageWithEmoji: {
    id: 'msg-emoji-1',
    role: 'user' as const,
    content: 'Hello! 👋 This message has emojis 🎉 and special characters <>&"\'',
    timestamp: Date.now(),
    status: 'success' as const,
  },
};

export const messageList: Message[] = [
  fixtures.userMessage,
  fixtures.agentMessage,
];
