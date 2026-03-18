import { HITLRequest } from '@/types';
import { parseSSEEventData, parseSSELines } from '@/utils/sseParser';

export interface HITLResumeRequest {
  session_id: string;
  hitl_id: string;
  confirmed: boolean;
  feedback?: string;
  parameters?: Record<string, any>;
}

export interface HITLResumeResponse {
  success: boolean;
  message?: string;
  hitlRequest?: HITLRequest;
}

export const resumeHITL = async (
  params: HITLResumeRequest,
  onProgress?: (event: MessageEvent) => void,
  onHITLRequest?: (hitlRequest: HITLRequest) => void
): Promise<HITLResumeResponse> => {
  try {
    console.log('[hitlService] Calling resume API:', params);
    const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${baseURL}/api/chat/resume`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let result: HITLResumeResponse = { success: false };
    let buffer = '';

    const processMessage = (msg: { event: string; data: string }) => {
      console.log('[hitlService] Processing message, event:', msg.event);
      console.log('[hitlService] Message data (first 200):', msg.data.substring(0, 200));
      
      if (msg.data === '[DONE]') {
        console.log('[hitlService] Got [DONE], skipping');
        return;
      }

      try {
        if (msg.event === 'error') {
          const errorData = JSON.parse(msg.data);
          console.error('[hitlService] Error event:', errorData);
          throw new Error(errorData.error || 'Resume failed');
        }
        
        if (msg.event === 'done') {
          console.log('[hitlService] Done event');
          result = { success: true };
          return;
        }

        console.log('[hitlService] Parsing event data...');
        const parsed = parseSSEEventData(msg.data);
        console.log('[hitlService] Parsed event type:', parsed.type);
        console.log('[hitlService] Parsed event data.event_type:', parsed.data?.event_type);
        console.log('[hitlService] Parsed hitlRequest id:', parsed.hitlRequest?.hitl_id);

        if (parsed.type === 'hitl_request' && parsed.hitlRequest) {
          console.log('[hitlService] *** HITL request found! ***');
          result = { success: true, hitlRequest: parsed.hitlRequest };
          if (onHITLRequest) {
            console.log('[hitlService] Calling onHITLRequest callback');
            onHITLRequest(parsed.hitlRequest);
          }
        } else if (parsed.type === 'task_complete') {
          console.log('[hitlService] Task complete event');
          result = { success: true };
        }

        if (onProgress && parsed.data.event_type) {
          onProgress(new MessageEvent('message', { data: msg.data }));
        }
      } catch (e) {
        console.error('[hitlService] Error processing message:', e);
        if (e instanceof SyntaxError) {
          console.warn('[hitlService] Failed to parse JSON:', msg.data.substring(0, 100));
        } else {
          throw e;
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        console.log('[hitlService] Stream done, processing remaining buffer:', buffer.length);
        if (buffer.trim()) {
          const { messages } = parseSSELines(buffer, true);
          console.log('[hitlService] Final messages:', messages.length);
          for (const msg of messages) {
            processMessage(msg);
          }
        }
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      console.log('[hitlService] Received chunk:', chunk.length, 'bytes');
      buffer += chunk;
      
      const { messages, remaining } = parseSSELines(buffer, false);
      console.log('[hitlService] Parsed messages:', messages.length, 'remaining:', remaining.length);
      buffer = remaining;
      
      for (const msg of messages) {
        processMessage(msg);
      }
    }

    console.log('[hitlService] Resume completed, result:', result);
    return result;
  } catch (error: any) {
    console.error('[hitlService] Resume API error:', error);
    throw error;
  }
};
