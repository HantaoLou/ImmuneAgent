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
  
  // Normalize line endings - handle both \n\n and \r\n\r\n
  const normalizedText = text.replace(/\r\n/g, '\n');
  
  // Try standard SSE format first (messages separated by \n\n)
  const lastBoundary = normalizedText.lastIndexOf('\n\n');
  
  if (lastBoundary !== -1) {
    // Standard SSE format with \n\n separators
    const completeText = normalizedText.substring(0, lastBoundary);
    let remaining = text.substring(lastBoundary + 4); // Keep original text for remaining, but skip \n\n (or \r\n\r\n which is 4 chars)
    
    // Find actual position in original text
    const actualLastBoundary = text.lastIndexOf('\n\n') !== -1 
      ? text.lastIndexOf('\n\n') 
      : text.lastIndexOf('\r\n\r\n');
    if (actualLastBoundary !== -1) {
      remaining = text.substring(actualLastBoundary + (text.includes('\r\n\r\n') ? 4 : 2));
    }
    
    if (isFinal && remaining.trim()) {
      let event = '';
      let data = '';
      const lines = remaining.replace(/\r\n/g, '\n').split('\n');
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
  
  // Fallback: Non-standard format where messages are separated by new event: lines
  // Only process if isFinal is true, otherwise wait for more data
  if (!isFinal) {
    return { messages, remaining: text };
  }
  
  // Parse line by line and group by event blocks
  const normalizedLines = normalizedText.split('\n');
  let currentEvent = '';
  let currentData = '';
  
  for (const line of normalizedLines) {
    if (line.startsWith('event:')) {
      // New event block starts, save previous if exists
      if (currentData && currentData !== '[DONE]') {
        messages.push({ event: currentEvent || 'message', data: currentData });
      }
      currentEvent = line.slice(6).trim();
      currentData = '';
    } else if (line.startsWith('data:')) {
      currentData = line.slice(5).trim();
    }
  }
  
  // Don't forget the last message
  if (currentData && currentData !== '[DONE]') {
    messages.push({ event: currentEvent || 'message', data: currentData });
  }
  
  return { messages, remaining: '' };
}
