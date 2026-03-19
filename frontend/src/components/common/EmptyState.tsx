'use client';

import React from 'react';
import styles from './EmptyState.module.css';

export const DEFAULT_QUESTION_TEMPLATES = [
  { label: 'Which T cells bind MART-1 cancer epitope?', question: 'Which T cells bind MART-1 cancer epitope?\n- rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds\n- csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv' },
  { label: 'Which T cells bind influenza Flu-MP epitope?', question: 'Which T cells bind influenza Flu-MP epitope?\n- rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds\n- csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv' },
  { label: 'Which T cells bind EBV epitopes?', question: 'Which T cells bind EBV epitopes?\n- rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds\n- csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv' },
  { label: 'Which TCRs bind SARS-CoV-2 YLQPRTFLL?', question: 'Which TCRs bind SARS-CoV-2 YLQPRTFLL?\n- rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds\n- csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv' },
];

interface QuestionTemplate {
  label: string;
  question: string;
}

interface EmptyStateProps {
  tip?: string;
  questionTemplates?: QuestionTemplate[];
  onQuestionClick?: (question: string) => void;
  showTemplates?: boolean;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  tip = 'no data',
  questionTemplates = DEFAULT_QUESTION_TEMPLATES,
  onQuestionClick,
  showTemplates = true,
}) => {
  return (
    <div className={styles.emptyState}>
      <div className={styles.icon}>💬</div>
      <div className={styles.content}>
        <div className={styles.title}>Start Conversation</div>
        <div className={styles.tip}>{tip}</div>
        {showTemplates && questionTemplates.length > 0 && (
          <div className={styles.questionTemplates}>
            {questionTemplates.map((template, index) => (
              <button
                key={index}
                className={styles.questionTag}
                onClick={() => onQuestionClick?.(template.question)}
              >
                {template.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className={styles.glow}></div>
    </div>
  );
};
