import React from 'react'
import { Card, Steps, Typography, Space, Button, Tag } from 'antd'

export interface PlanStepDraft {
  id: string
  title: string
  objective: string
  notes?: string
  tools: string[]
}

interface PlanSummary {
  planId?: string | null
  originalQuestion?: string
  planText?: string
  steps: PlanStepDraft[]
}

interface PlanSummaryPanelProps {
  summary: PlanSummary
  onEdit?: () => void
}

export const PlanSummaryPanel: React.FC<PlanSummaryPanelProps> = ({ summary, onEdit }) => {
  const { steps = [], originalQuestion, planId } = summary

  return (
    <Card
      title="Execution Plan Overview"
      extra={
        onEdit ? (
          <Button type="link" onClick={onEdit}>
            Modify Plan
          </Button>
        ) : null
      }
      style={{ marginBottom: 16 }}
      bodyStyle={{ paddingBottom: 12 }}
    >
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        {originalQuestion && (
          <Typography.Paragraph style={{ marginBottom: 0 }}>
            <Typography.Text strong>Objective:&nbsp;</Typography.Text>
            {originalQuestion}
          </Typography.Paragraph>
        )}
        {planId && (
          <Typography.Text type="secondary">Plan ID: {planId}</Typography.Text>
        )}
        <Steps direction="vertical" size="small" current={-1}>
          {steps.map((step, idx) => (
            <Steps.Step
              key={step.id || idx}
              title={
                <Space direction="vertical" size={4} style={{ width: '100%' }}>
                  <Typography.Text strong>
                    {step.title || `Step ${idx + 1}`}
                  </Typography.Text>
                  {step.objective && (
                    <Typography.Text>{step.objective}</Typography.Text>
                  )}
                  {!!step.tools?.length && (
                    <Space size={[4, 4]} wrap>
                      {step.tools.map((tool) => (
                        <Tag key={`${step.id}-${tool}`} color="blue">
                          {tool}
                        </Tag>
                      ))}
                    </Space>
                  )}
                  {step.notes && (
                    <Typography.Text type="secondary">
                      {step.notes}
                    </Typography.Text>
                  )}
                </Space>
              }
            />
          ))}
        </Steps>
      </Space>
    </Card>
  )
}


