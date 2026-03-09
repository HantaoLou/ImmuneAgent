import { Session, Template, Message } from './types';

const SESSIONS_KEY = 'bio_agent_sessions';
const TEMPLATES_KEY = 'bio_agent_templates';

export class SessionStorage {
  static getSessions(): Session[] {
    if (typeof window === 'undefined') return [];
    const data = localStorage.getItem(SESSIONS_KEY);
    return data ? JSON.parse(data) : [];
  }

  static saveSession(session: Session): void {
    if (typeof window === 'undefined') return;
    const sessions = this.getSessions();
    const index = sessions.findIndex(s => s.id === session.id);
    if (index >= 0) {
      sessions[index] = session;
    } else {
      sessions.unshift(session);
    }
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  }

  static deleteSession(sessionId: string): void {
    if (typeof window === 'undefined') return;
    const sessions = this.getSessions().filter(s => s.id !== sessionId);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  }

  static getTemplates(): Template[] {
    if (typeof window === 'undefined') return [];
    const data = localStorage.getItem(TEMPLATES_KEY);
    return data ? JSON.parse(data) : this.getDefaultTemplates();
  }

  static saveTemplate(template: Template): void {
    if (typeof window === 'undefined') return;
    const templates = this.getTemplates();
    const index = templates.findIndex(t => t.id === template.id);
    if (index >= 0) {
      templates[index] = template;
    } else {
      templates.push(template);
    }
    localStorage.setItem(TEMPLATES_KEY, JSON.stringify(templates));
  }

  static deleteTemplate(templateId: string): void {
    if (typeof window === 'undefined') return;
    const templates = this.getTemplates().filter(t => t.id !== templateId);
    localStorage.setItem(TEMPLATES_KEY, JSON.stringify(templates));
  }

  static getDefaultTemplates(): Template[] {
    return [
      {
        id: '1',
        name: 'Antibody Analysis',
        content: 'Analyze the antibody data in {file_path} and predict neutralization potential.',
        category: 'Immunology',
      },
      {
        id: '2',
        name: 'Sequence Alignment',
        content: 'Perform sequence alignment for {sequences} using BLAST.',
        category: 'Bioinformatics',
      },
      {
        id: '3',
        name: 'Data Visualization',
        content: 'Create visualizations for the dataset in {file_path}.',
        category: 'Analysis',
      },
      {
        id: '4',
        name: 'General Question',
        content: 'Explain the concept of {concept} in simple terms.',
        category: 'General',
      },
    ];
  }
}

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function createSession(title: string = 'New Chat'): Session {
  return {
    id: generateId(),
    title,
    createdAt: new Date().toISOString(),
    messages: [],
  };
}

export function createMessage(role: 'user' | 'assistant', content: string): Message {
  return {
    id: generateId(),
    role,
    content,
    timestamp: new Date().toISOString(),
  };
}
