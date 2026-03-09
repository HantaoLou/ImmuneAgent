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
        id: 'Q01',
        name: 'H1N1 Michigan 结合抗体预测',
        content: 'Given the single-cell RNA-seq data and antibody sequences for 100 flu-specific monoclonal antibodies, predict which antibodies bind to H1N1 A/Michigan/45/2015 hemagglutinin. For each antibody (identified by `mAb`), output a binary prediction: 1 = binds, 0 = does not bind.\nExpected output format:\n- CSV with columns: `mAb`, `prediction` (0 or 1), optionally `probability` (0.0-1.0)\n- One row per antibody (up to 100 rows)\n\nGround truth summary:\n- 98 antibodies tested (2 are NaN)\n- 82 positive (83.7%), 16 negative\n\nPrimary metric: F1\n"meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",\n"antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",\n"meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"',
        category: 'FLU',
      },
      {
        id: 'Q13',
        name: 'MART-1癌症表位TCR结合预测',
        content: 'Given 2080 T cell receptors with paired CDR3 alpha (CDR3a) and CDR3 beta (CDR3b) sequences, predict which TCRs bind the MART-1 cancer epitope (peptide: ELAGIGILTV, presented by HLA-A*02:01). MART-1 is a melanoma-associated antigen. For each TCR (identified by `main_name`), output a binary prediction: True = binder, False = non-binder.\nWhat to use:\n- CDR3a and CDR3b sequences for each TCR\n- Target peptide: ELAGIGILTV\n- HLA restriction: A*02:01\n- TCR V/J gene usage annotations\n- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)\n\nExpected output format:\n- CSV with columns: `main_name`, `prediction` (True or False), optionally `probability` (0.0-1.0)\n- 2080 rows (one per TCR)\n\nGround truth summary:\n- 2080 TCRs tested\n- 60 positive (2.9%), 2020 negative\n- Highly imbalanced -- only ~3% are MART-1 binders\n\nPrimary metric: F1\n"rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds",\n"meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv"',
        category: 'TCR',
      }
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
