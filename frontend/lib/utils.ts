export function formatDate(date: Date | string): string {
  return new Date(date).toLocaleString();
}

export function formatRelativeTime(date: Date | string): string {
  const now = new Date();
  const diff = now.getTime() - new Date(date).getTime();
  
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}h ago`;
  } else if (minutes > 0) {
    return `${minutes}m ago`;
  } else {
    return `${seconds}s ago`;
  }
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return text.substring(0, maxLength) + '...';
}

export function getProgressIcon(type: string): string {
  const iconMap: Record<string, string> = {
    'node_start': '▶️',
    'node_progress': '🔄',
    'node_complete': '✅',
    'task_start': '🎯',
    'task_progress': '⚙️',
    'task_complete': '✨',
    'code_generation': '💻',
    'code_execution': '🚀',
    'tool_call': '🔧',
    'error': '❌',
    'info': 'ℹ️',
    'llm_thinking': '🧠',
    'llm_reasoning': '💭',
    'llm_streaming': '✍️',
    'tool_result': '📤',
    'subgraph_step': '🔷',
    'knowledge_retrieval': '🔍',
    'analysis_progress': '📊',
  };
  
  return iconMap[type] || '📍';
}

export function getProgressColor(type: string): string {
  const colorMap: Record<string, string> = {
    'node_start': 'text-blue-600 bg-blue-50 border-blue-200',
    'node_progress': 'text-yellow-700 bg-yellow-50 border-yellow-200',
    'node_complete': 'text-green-600 bg-green-50 border-green-200',
    'task_start': 'text-blue-600 bg-blue-50 border-blue-200',
    'task_progress': 'text-yellow-700 bg-yellow-50 border-yellow-200',
    'task_complete': 'text-green-600 bg-green-50 border-green-200',
    'code_generation': 'text-purple-600 bg-purple-50 border-purple-200',
    'code_execution': 'text-purple-600 bg-purple-50 border-purple-200',
    'tool_call': 'text-indigo-600 bg-indigo-50 border-indigo-200',
    'error': 'text-red-600 bg-red-50 border-red-200',
    'info': 'text-gray-600 bg-gray-50 border-gray-200',
    'llm_thinking': 'text-violet-600 bg-violet-50 border-violet-200',
    'llm_reasoning': 'text-violet-600 bg-violet-50 border-violet-200',
    'llm_streaming': 'text-blue-600 bg-blue-50 border-blue-200',
    'tool_result': 'text-emerald-600 bg-emerald-50 border-emerald-200',
    'subgraph_step': 'text-cyan-600 bg-cyan-50 border-cyan-200',
    'knowledge_retrieval': 'text-amber-600 bg-amber-50 border-amber-200',
    'analysis_progress': 'text-teal-600 bg-teal-50 border-teal-200',
  };
  
  return colorMap[type] || 'text-gray-600 bg-gray-50 border-gray-200';
}
