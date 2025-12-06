import React, { useEffect, useMemo, useState } from 'react'
import { Modal, Collapse, Input, Select, Space, Typography, Button, Divider } from 'antd'
import type { PlanStepDraft } from './plan-summary-panel'

const { Panel } = Collapse
const { TextArea } = Input

export interface PlanReviewModalProps {
  visible: boolean
  planId?: string | null
  originalQuestion?: string
  initialSteps: PlanStepDraft[]
  onConfirm: (steps: PlanStepDraft[], planText: string) => void
  onReject: () => void
  onCancel: () => void
}

const normalizeToolsInput = (value: string[] | undefined) =>
  (value ?? []).map((item) => item.trim()).filter(Boolean)

export const buildPlanTextFromSteps = (steps: PlanStepDraft[]): string => {
  return steps
    .map((step, idx) => {
      const parts: string[] = []
      const title = step.title || `Step ${idx + 1}`
      parts.push(`${idx + 1}. ${title}`)
      if (step.objective) {
        parts.push(`Objective: ${step.objective}`)
      }
      if (step.tools?.length) {
        parts.push(`Tools: ${step.tools.join(', ')}`)
      }
      if (step.notes) {
        parts.push(`Notes: ${step.notes}`)
      }
      return parts.join(' | ')
    })
    .join('\n')
}

export const PlanReviewModal: React.FC<PlanReviewModalProps> = ({
  visible,
  planId,
  originalQuestion,
  initialSteps,
  onConfirm,
  onReject,
  onCancel,
}) => {
  const [steps, setSteps] = useState<PlanStepDraft[]>(initialSteps)

  useEffect(() => {
    if (visible) {
      setSteps(initialSteps)
    }
  }, [initialSteps, visible])

  const planText = useMemo(() => buildPlanTextFromSteps(steps), [steps])

  const updateStep = (index: number, field: keyof PlanStepDraft, value: any) => {
    setSteps((prev) => {
      const next = [...prev]
      next[index] = {
        ...next[index],
        [field]: field === 'tools' ? normalizeToolsInput(value) : value,
      }
      return next
    })
  }

  return (
    <Modal
      open={visible}
      title="Confirm or Adjust Execution Plan"
      onCancel={onCancel}
      width={840}
      destroyOnClose={false}
      footer={
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Button onClick={onCancel}>Close</Button>
          <Space>
            <Button danger onClick={onReject}>
              Reject Plan
            </Button>
            <Button
              type="primary"
              onClick={() => onConfirm(steps, planText)}
            >
              Confirm Plan
            </Button>
          </Space>
        </Space>
      }
    >
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        {planId && (
          <Typography.Text type="secondary">Plan ID: {planId}</Typography.Text>
        )}
        {originalQuestion && (
          <Typography.Paragraph style={{ marginBottom: 0 }}>
            <Typography.Text strong>Objective:&nbsp;</Typography.Text>
            {originalQuestion}
          </Typography.Paragraph>
        )}
        <Divider style={{ margin: '12px 0' }} />
        <Collapse accordion defaultActiveKey={steps[0]?.id}>
          {steps.map((step, idx) => (
            <Panel
              header={step.title || `Step ${idx + 1}`}
              key={step.id || idx}
            >
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Input
                  value={step.title}
                  onChange={(e) => updateStep(idx, 'title', e.target.value)}
                  placeholder="Step title"
                />
                <TextArea
                  value={step.objective}
                  onChange={(e) => updateStep(idx, 'objective', e.target.value)}
                  placeholder="Objective / Description"
                  autoSize={{ minRows: 2, maxRows: 4 }}
                />
                <Select
                  value={step.tools}
                  onChange={(value) => updateStep(idx, 'tools', value)}
                  mode="tags"
                  placeholder="Tools to use (press Enter to add)"
                  style={{ width: '100%' }}
                />
                <TextArea
                  value={step.notes}
                  onChange={(e) => updateStep(idx, 'notes', e.target.value)}
                  placeholder="Additional notes"
                  autoSize={{ minRows: 2, maxRows: 4 }}
                />
              </Space>
            </Panel>
          ))}
        </Collapse>
        <Divider style={{ margin: '12px 0' }} />
        <Typography.Text type="secondary">
          Final Plan Preview:
        </Typography.Text>
        <Typography.Paragraph
          style={{
            background: '#fafafa',
            padding: 12,
            borderRadius: 6,
            whiteSpace: 'pre-wrap',
            marginBottom: 0,
          }}
        >
          {planText}
        </Typography.Paragraph>
      </Space>
    </Modal>
  )
}


