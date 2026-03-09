'use client';

import React from 'react';
import { Download, FileText, Eye } from 'lucide-react';
import { OutputFile } from '@/lib/types';
import { getFileDownloadUrl } from '@/lib/api';

interface OutputFilesDisplayProps {
  files: OutputFile[];
  sessionId: string;
}

export function OutputFilesDisplay({ files, sessionId }: OutputFilesDisplayProps) {
  if (!files || files.length === 0) return null;

  const getFileIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'csv data':
      case 'tsv data':
        return '📊';
      case 'json data':
        return '📋';
      case 'markdown':
      case 'text file':
        return '📄';
      case 'pdf report':
        return '📕';
      case 'image':
        return '🖼️';
      case 'fasta sequence':
        return '🧬';
      default:
        return '📁';
    }
  };

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 my-2">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">📤</span>
        <h3 className="font-semibold text-blue-800">
          输出文件 ({files.length})
        </h3>
      </div>
      
      <div className="space-y-2">
        {files.map((file, index) => (
          <div
            key={index}
            className="flex items-center justify-between bg-white rounded-lg p-3 border border-blue-100 hover:border-blue-300 transition-colors"
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <span className="text-2xl flex-shrink-0">
                {getFileIcon(file.type)}
              </span>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-800 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-gray-500">
                  {file.type} • {file.size_formatted}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => {
                  const url = getFileDownloadUrl(sessionId, file.relative_path);
                  window.open(url, '_blank');
                }}
                className="flex items-center gap-1 px-3 py-1.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
              >
                <Download className="w-4 h-4" />
                下载
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
