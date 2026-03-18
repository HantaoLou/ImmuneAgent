import { HITLRequest } from '@/types';

export interface SSEEventData {
  event_type: string;
  message: string;
  timestamp: string;
  node_name?: string;
  details?: Record<string, any>;
  session_id?: string;
}

export interface ParsedSSEResult {
  type: 'hitl_request' | 'task_complete' | 'progress' | 'unknown';
  data: SSEEventData;
  hitlRequest?: HITLRequest;
  result?: any;
}

export function parseSSEEventData(rawData: string): ParsedSSEResult {
  try {
    const data: SSEEventData = JSON.parse(rawData);
    
    if (data.event_type === 'hitl_request') {
      return {
        type: 'hitl_request',
        data,
        hitlRequest: data.details?.hitl_request as HITLRequest | undefined,
      };
    }
    
    if (data.event_type === 'task_complete') {
      return {
        type: 'task_complete',
        data,
        result: data.details?.result,
      };
    }
    
    return {
      type: 'progress',
      data,
    };
  } catch (e) {
    return {
      type: 'unknown',
      data: {} as SSEEventData,
    };
  }
}

export interface SSEMessage {
  event: string;
  data: string;
}

export function parseSSELines(text: string, isFinal: boolean = false): { messages: SSEMessage[]; remaining: string } {
  const messages: SSEMessage[] = [];
  
  const lastBoundary = text.lastIndexOf('\n\n');
  
  if (lastBoundary === -1) {
    if (isFinal && text.trim()) {
      let event = '';
      let data = '';
      const lines = text.split('\n');
      for (const line of lines) {
        if (line.startsWith('event:')) {
          event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          data = line.slice(5).trim();
        }
      }
      if (data && data !== '[DONE]') {
        messages.push({ event: event || 'message', data });
      }
      return { messages, remaining: '' };
    }
    return { messages, remaining: text };
  }
  
  const completeText = text.substring(0, lastBoundary);
  let remaining = text.substring(lastBoundary + 2);
  
  if (isFinal && remaining.trim()) {
    let event = '';
    let data = '';
    const lines = remaining.split('\n');
    for (const line of lines) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        data = line.slice(5).trim();
      }
    }
    if (data && data !== '[DONE]') {
      messages.push({ event: event || 'message', data });
    }
    remaining = '';
  }
  
  const messageBlocks = completeText.split('\n\n');
  
  for (const block of messageBlocks) {
    if (!block.trim()) continue;
    
    let event = '';
    let data = '';
    
    const lines = block.split('\n');
    for (const line of lines) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        data = line.slice(5).trim();
      }
    }
    
    if (data && data !== '[DONE]') {
      messages.push({ event: event || 'message', data });
    }
  }
  
  return { messages, remaining };
}
