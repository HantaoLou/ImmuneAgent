import React, { useState, KeyboardEvent } from 'react';
import { Send, FolderOpen, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  sessionId?: string;
  outputFilesCount?: number;
  onFetchFiles?: () => void;
  isLoadingFiles?: boolean;
}

export function MessageInput({
  onSend,
  disabled = false,
  placeholder = 'Type your message...',
  sessionId,
  outputFilesCount = 0,
  onFetchFiles,
  isLoadingFiles = false
}: MessageInputProps) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-gray-200 p-4">

      {/* 输入框和发送按钮 */}
      <div className="flex items-end space-x-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder}
          rows={5}
          className="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-400"
        />
        <div className="flex flex-col">
          <Button
            onClick={onFetchFiles}
            disabled={isLoadingFiles}
            className="h-10 px-4 mb-2"
          >
            <FolderOpen className="h-4 w-4" />
          </Button>

          <Button
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            className="h-10 px-4"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
