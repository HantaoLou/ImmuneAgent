'use client';

import { useState } from 'react';

export default function DebugProgress() {
  const [events, setEvents] = useState<any[]>([]);
  const [status, setStatus] = useState<string>('');

  const testSSE = async () => {
    setEvents([]);
    setStatus('Connecting...');

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          message: 'hello',
          session_id: 'debug_' + Date.now(),
        }),
      });

      setStatus('Connected, reading stream...');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        setStatus('Error: No reader available');
        return;
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data:')) {
            const data = line.substring(5).trim();
            try {
              const parsed = JSON.parse(data);
              setEvents(prev => [...prev, {
                ...parsed,
                receivedAt: new Date().toISOString()
              }]);
            } catch (e) {
              console.error('Failed to parse:', data);
            }
          }
        }
      }

      setStatus('Stream completed');
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">SSE Debug Page</h1>
      
      <button
        onClick={testSSE}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Test SSE Connection
      </button>

      <div className="mt-4">
        <p className="font-semibold">Status: {status}</p>
      </div>

      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">
          Events Received: {events.length}
        </h2>
        
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {events.map((event, index) => (
            <div
              key={index}
              className="p-2 border rounded text-sm"
            >
              <div className="font-semibold text-blue-600">
                {event.event_type || 'status'}
              </div>
              <div className="text-gray-700">
                {event.message || event.stage || JSON.stringify(event).substring(0, 100)}
              </div>
              {event.node_name && (
                <div className="text-gray-500 text-xs">
                  Node: {event.node_name}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
