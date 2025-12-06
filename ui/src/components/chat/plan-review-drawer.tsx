import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react'
import {
  Button,
  Card,
  Drawer,
  Form,
  Input,
  Radio,
  Select,
  Space,
  Tag,
  Timeline,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { listServices, listTools } from '../../services/tools-service'

const { TextArea } = Input
const { Paragraph, Title, Text } = Typography

export interface PlanStepFormValues {
  step_id?: string
  title?: string
  description?: string
  objective?: string
  tools?: string[]
  toolchain?: string[]
  recommended_tools?: string[]
  notes?: string
  inputs?: string[]
  outputs?: string[]
  status?: string
}

export interface PlanSummaryData {
  planId?: string | null
  totalSteps?: number
  originalQuestion?: string
  planText?: string
  steps: PlanStepFormValues[]
}

export interface ExecutionStepState {
  status?: string
  toolsCalled?: any[]
  toolResults?: any[]
  plannedTools?: any[]
  output?: string
  message?: string
}

export type ExecutionStateMap = Record<string, ExecutionStepState>

interface PlanReviewDrawerProps {
  open: boolean
  mode: 'view' | 'confirm'
  planSummary: PlanSummaryData | null
  executionState?: ExecutionStateMap
  onClose: () => void
  onConfirm?: (payload: {
    planId?: string | null
    planText?: string
    steps: PlanStepFormValues[]
    actionId?: string | number
  }) => Promise<void> | void
  onReject?: () => void
  confirmActionId?: string | number
}

const statusColorMap: Record<string, string> = {
  pending: 'default',
  running: 'processing',
  completed: 'success',
  failed: 'error',
  needs_review: 'warning',
  timeout: 'warning',
}

const statusLabelMap: Record<string, string> = {
  pending: 'Pending',
  running: 'In Progress',
  completed: 'Completed',
  failed: 'Failed',
  needs_review: 'Needs Review',
  timeout: 'Timeout',
}

const toolStatusColorMap: Record<string, string> = {
  running: 'processing',
  completed: 'success',
  failed: 'error',
  rejected: 'default',
  skipped: 'default',
}

const toolStatusLabelMap: Record<string, string> = {
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  rejected: 'Rejected',
  skipped: 'Skipped',
}

export const PlanReviewDrawer: React.FC<PlanReviewDrawerProps> = ({
  open,
  mode,
  planSummary,
  executionState = {},
  onClose,
  onConfirm,
  onReject,
  confirmActionId,
}) => {
  const [form] = Form.useForm()
  const isConfirmMode = mode === 'confirm'
  const readOnly = !isConfirmMode
  const [selectedStepIndex, setSelectedStepIndex] = useState(0)
  const [serviceOptions, setServiceOptions] = useState<Array<{ label: string; value: string }>>([])
  const [groupedToolOptions, setGroupedToolOptions] = useState<Record<string, Array<{ label: string; value: string }>>>({})
  const toolDescriptionMapRef = useRef<Map<string, string>>(new Map())
  const [serviceFilter, setServiceFilter] = useState<string>('')
  const selectedStepIdRef = useRef<string | null>(null)
  const userPinnedSelectionRef = useRef(false)
  const toolFetchAbortersRef = useRef<AbortController[]>([])
  const [confirming, setConfirming] = useState(false)
  const toolsLoadingRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (isConfirmMode) {
      console.log('[PlanReviewDrawer] props update', {
        confirmActionId,
        open,
        mode,
        hasPlanSummary: !!planSummary,
      })
    }
  }, [confirmActionId, open, mode, isConfirmMode, planSummary])

  const initialSteps = useMemo(() => {
    if (!planSummary?.steps) return []
    return planSummary.steps.map((step, index) => ({
      step_id: step.step_id || `step-${index + 1}`,
      title: step.title ?? '',
      description: step.description ?? '',
      objective: step.objective ?? '',
      tools: step.tools ?? step.toolchain ?? [],
      recommended_tools: step.recommended_tools ?? [],
      notes: step.notes ?? '',
      inputs: step.inputs ?? [],
      outputs: step.outputs ?? [],
      status: step.status ?? 'pending',
    }))
  }, [planSummary])

  const updateSelection = useCallback(
    (nextIndex: number, stepsSnapshot: PlanStepFormValues[] = initialSteps) => {
      if (!stepsSnapshot.length) {
        selectedStepIdRef.current = null
        setSelectedStepIndex(0)
        return
      }
      const boundedIndex = Math.max(0, Math.min(nextIndex, stepsSnapshot.length - 1))
      const targetStep = stepsSnapshot[boundedIndex]
      const targetId = targetStep?.step_id || `step-${boundedIndex + 1}`
      selectedStepIdRef.current = targetId
      setSelectedStepIndex(prev => (prev === boundedIndex ? prev : boundedIndex))
    },
    [initialSteps],
  )

  useEffect(() => {
    if (!open) return
    if (planSummary) {
      form.setFieldsValue({
        planText: planSummary.planText ?? '',
        steps: initialSteps,
      })
      userPinnedSelectionRef.current = false
    } else {
      form.resetFields()
      userPinnedSelectionRef.current = false
      selectedStepIdRef.current = null
      setSelectedStepIndex(0)
      setLocalToolSelection([])
      setLocalServiceSelection('all')
    }
  }, [open, planSummary, form, initialSteps])

  useEffect(() => {
    if (!open) return
    const steps = initialSteps
    if (!steps.length) {
      selectedStepIdRef.current = null
      setSelectedStepIndex(0)
      return
    }

    if (userPinnedSelectionRef.current) {
      const pinnedId = selectedStepIdRef.current
      if (pinnedId) {
        const pinnedIndex = steps.findIndex((step, idx) => {
          const stepId = step.step_id || `step-${idx + 1}`
          return stepId === pinnedId
        })
        if (pinnedIndex !== -1) {
          updateSelection(pinnedIndex, steps)
          return
        }
        userPinnedSelectionRef.current = false
      }
    }

    const runningIndex = steps.findIndex((step, idx) => {
      const stepId = step.step_id || `step-${idx + 1}`
      const status = executionState[stepId]?.status || step.status
      return status === 'running'
    })
    const fallbackIndex =
      runningIndex !== -1
        ? runningIndex
        : Math.max(
            0,
            Math.min(selectedStepIndex, steps.length - 1),
          )

    updateSelection(fallbackIndex, steps)
  }, [open, initialSteps, executionState, selectedStepIndex, updateSelection])

  const handleConfirm = async () => {
    if (!isConfirmMode || !planSummary) {
      onClose()
      return
    }
    if (confirmActionId === undefined || confirmActionId === null || confirmActionId === '') {
      message.warning('正在准备确认上下文，请稍后再试。')
      return
    }
    setConfirming(true)
    try {
      toolFetchAbortersRef.current.forEach(controller => controller.abort())
      toolFetchAbortersRef.current = []
      const values = await form.validateFields()
      const payloadSteps: PlanStepFormValues[] = (values.steps || []).map((step: PlanStepFormValues, index: number) => ({
        ...step,
        step_id: step.step_id || `step-${index + 1}`,
      }))

      console.log('[PlanReviewDrawer] Submit confirm click', {
        planId: planSummary.planId,
        actionId: confirmActionId,
      })

      await onConfirm?.({
        planId: planSummary.planId ?? null,
        planText: values.planText ?? '',
        steps: payloadSteps,
        actionId: confirmActionId,
      })
    } catch (err) {
      // validation errors handled by form
    } finally {
      setConfirming(false)
    }
  }

  const handleReject = () => {
    if (onReject) {
      onReject()
    }
  }

  const renderStatusTag = (step: PlanStepFormValues, index: number) => {
    const stepId = step.step_id || `step-${index + 1}`
    const executionInfo = executionState[stepId]
    const status = executionInfo?.status || step.status || 'pending'
    const color = statusColorMap[status] || 'default'
    const label = statusLabelMap[status] || status
    return <Tag color={color}>{label}</Tag>
  }

  const renderToolCallStatusTag = (status?: string) => {
    if (!status) return null
    const normalized = status.toLowerCase()
    const color = toolStatusColorMap[normalized] || 'default'
    const label = toolStatusLabelMap[normalized] || status
    return <Tag color={color}>{label}</Tag>
  }

  const formatArgsPreview = (args: any) => {
    if (args === undefined || args === null) return '—'
    if (typeof args === 'string') return args
    try {
      return JSON.stringify(args, null, 2)
    } catch (err) {
      return String(args)
    }
  }

  const historicalToolOptions = useMemo(() => {
    const collected = new Set<string>()

    ;(planSummary?.steps ?? []).forEach(step => {
      ;(step.tools ?? step.toolchain ?? []).forEach((tool: string) => {
        if (tool) collected.add(tool)
      })
    })

    Object.values(executionState).forEach(stateItem => {
      const planned = stateItem?.plannedTools ?? []
      if (Array.isArray(planned)) {
        planned.forEach((tool: any) => {
          const value = typeof tool === 'string' ? tool : tool?.tool_name || tool?.name
          if (value) collected.add(value)
        })
      }
      const calls = stateItem?.toolsCalled ?? []
      if (Array.isArray(calls)) {
        calls.forEach((call: any) => {
          const value = call?.tool_name || call?.name
          if (value) collected.add(value)
        })
      }
    })

    return Array.from(collected)
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b))
      .map(tool => ({ label: tool, value: tool }))
  }, [planSummary, executionState])

  const toolOptions = useMemo(() => {
    if (serviceFilter === 'all') {
      const merged = [...historicalToolOptions]
      Object.values(groupedToolOptions).forEach(list => {
        list.forEach(item => {
          if (!merged.find(existing => existing.value === item.value)) {
            merged.push(item)
          }
        })
      })
      return merged.sort((a, b) => a.label.localeCompare(b.label))
    }

    const options = groupedToolOptions[serviceFilter] || []
    if (!options.length) {
      return historicalToolOptions
    }
    return options
  }, [historicalToolOptions, groupedToolOptions, serviceFilter])

  const toolDescriptionMap = useMemo(() => {
    const map = new Map<string, string>(toolDescriptionMapRef.current)
    Object.values(groupedToolOptions).forEach(list => {
      list.forEach(item => {
        if (item.value) {
          map.set(item.value, item.label)
        }
      })
    })
    historicalToolOptions.forEach(item => {
      if (item.value && !map.has(item.value)) {
        map.set(item.value, item.label)
      }
    })
    return map
  }, [groupedToolOptions])

  const [localServiceSelection, setLocalServiceSelection] = useState<string>('')
  const [localToolSelection, setLocalToolSelection] = useState<string[]>([])
  const handleServiceChange = useCallback((value: string) => {
    setLocalServiceSelection(prev => (prev === value ? prev : value))
    setServiceFilter(prev => (prev === value ? prev : value))
  }, [])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    toolFetchAbortersRef.current.forEach(controller => controller.abort())
    toolFetchAbortersRef.current = []

    const fetchTools = async () => {
      try {
        const services = await listServices()
        if (cancelled) return

        const serviceOpts = services
          .map(service => ({ label: service.id, value: service.id }))
          .sort((a, b) => a.label.localeCompare(b.label))

        if (cancelled) return

        setServiceOptions(serviceOpts)

        setGroupedToolOptions({})
        const firstService = serviceOpts[0]?.value || ''
        if (firstService) {
          handleServiceChange(firstService)
        } else {
          handleServiceChange('')
        }
      } catch (err) {
        console.error('Failed to load tool catalog:', err)
      }
    }

    fetchTools()

    return () => {
      cancelled = true
      toolFetchAbortersRef.current.forEach(controller => controller.abort())
      toolFetchAbortersRef.current = []
    }
  }, [open, handleServiceChange])

  useEffect(() => {
    if (!open) return
    const stepsValue = form.getFieldValue('steps') || initialSteps
    const currentTools = stepsValue[selectedStepIndex]?.tools || []
    setLocalToolSelection(currentTools)

    const currentStep = initialSteps[selectedStepIndex]
    const currentId = currentStep ? currentStep.step_id || `step-${selectedStepIndex + 1}` : null
    if (currentId) {
      selectedStepIdRef.current = currentId
    }
  }, [selectedStepIndex, form, open, initialSteps])

  useEffect(() => {
    if (!open) return
    const currentStep = initialSteps[selectedStepIndex]
    const detectedService =
      (currentStep as any)?.service_id ||
      (currentStep as any)?.serviceId ||
      ''

    if (detectedService && serviceOptions.find(opt => opt.value === detectedService)) {
      handleServiceChange(detectedService)
      return
    }

    const fallbackService = serviceOptions[0]?.value || ''
    if (fallbackService) {
      handleServiceChange(fallbackService)
    }
  }, [open, selectedStepIndex, initialSteps, serviceOptions, handleServiceChange])

  useEffect(() => {
    if (!open || !serviceFilter) {
      return
    }

    if (groupedToolOptions[serviceFilter]?.length) {
      return
    }

    if (toolsLoadingRef.current.has(serviceFilter)) {
      return
    }

    const controller = new AbortController()
    toolFetchAbortersRef.current.push(controller)
    toolsLoadingRef.current.add(serviceFilter)

    listTools(serviceFilter, controller.signal)
      .then(tools => {
        if (controller.signal.aborted) return
        const items: Array<{ label: string; value: string }> = []
        tools.forEach(tool => {
          const value = tool.name
          if (!value) return
          items.push({ label: value, value })
        })
        setGroupedToolOptions(prev => ({
          ...prev,
          [serviceFilter]: items,
        }))
        if (items.length) {
          const nextDescriptionMap = new Map<string, string>(toolDescriptionMapRef.current)
          tools.forEach(tool => {
            const value = tool.name
            if (!value) return
            if (tool.description) {
              nextDescriptionMap.set(value, tool.description)
            } else if (!nextDescriptionMap.has(value)) {
              nextDescriptionMap.set(value, value)
            }
          })
          toolDescriptionMapRef.current = nextDescriptionMap
        }
      })
      .catch(err => {
        if (!controller.signal.aborted) {
          console.error(`Failed to load tools for service ${serviceFilter}:`, err)
        }
      })
      .finally(() => {
        toolFetchAbortersRef.current = toolFetchAbortersRef.current.filter(item => item !== controller)
        toolsLoadingRef.current.delete(serviceFilter)
      })

    return () => {
      controller.abort()
      toolsLoadingRef.current.delete(serviceFilter)
    }
  }, [serviceFilter, open, groupedToolOptions])

  const handleTimelineStepClick = useCallback(
    (index: number) => {
      userPinnedSelectionRef.current = true
      updateSelection(index)
    },
    [updateSelection],
  )

  const timelineItems = useMemo(() => {
    return initialSteps.map((step, index) => {
      const stepId = step.step_id || `step-${index + 1}`
      const executionInfo = executionState[stepId]
      const status = executionInfo?.status || step.status || 'pending'
      const color = statusColorMap[status] || 'default'
      const isSelected = index === selectedStepIndex
      return {
        color,
        key: stepId,
        children: (
          <Space direction="vertical" size={4}>
            <Tooltip title={step.title || `Step ${index + 1}` } placement="right">
              <Button
                type={isSelected ? 'primary' : 'link'}
                onClick={() => handleTimelineStepClick(index)}
                style={{ padding: 0, maxWidth: 220 }}
              >
                <Text ellipsis style={{ maxWidth: 200, display: 'inline-block', textAlign: 'left' }}>
                  {step.title || `Step ${index + 1}`}
                </Text>
              </Button>
            </Tooltip>
            {renderStatusTag(step, index)}
          </Space>
        ),
      }
    })
  }, [initialSteps, executionState, selectedStepIndex])

  const selectedFieldName = initialSteps.length ? selectedStepIndex : -1

  return (
    <Drawer
      title={isConfirmMode ? 'Confirm Execution Plan' : 'Execution Plan'}
      width={880}
      open={open}
      onClose={onClose}
      destroyOnClose
      extra={
        mode === 'view' && planSummary ? (
          <Text type="secondary">Total {planSummary.steps?.length ?? 0} steps</Text>
        ) : null
      }
      footer={
        <Space style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button onClick={onClose}>Close</Button>
          {isConfirmMode ? (
            <>
              <Button danger onClick={handleReject}>
                Reject Plan
              </Button>
              <Button
                type="primary"
                loading={confirming}
                disabled={
                  confirming ||
                  confirmActionId === undefined ||
                  confirmActionId === null ||
                  confirmActionId === ''
                }
                onClick={handleConfirm}
              >
                Confirm Plan
              </Button>
            </>
          ) : null}
        </Space>
      }
    >
      {planSummary ? (
        <Form layout="vertical" form={form}>
          <Card size="small" style={{ marginBottom: 16 }}>
            <Title level={5} style={{ marginTop: 0 }}>
              Original Question
            </Title>
            <Paragraph>{planSummary.originalQuestion || 'N/A'}</Paragraph>
          </Card>

          <Form.Item name="planText" label="Plan Overview">
            <TextArea
              rows={3}
              placeholder="Provide a high-level description of the plan"
              disabled={readOnly}
            />
          </Form.Item>

          <Form.List name="steps">
            {(fields) => {
              if (!fields.length) {
                return <Paragraph type="secondary">No steps available.</Paragraph>
              }

              const clampedIndex = Math.min(Math.max(selectedFieldName, 0), fields.length - 1)
              const activeField = fields[clampedIndex]
              const stepValue = initialSteps[clampedIndex] || {}
              const stepId = stepValue.step_id || `step-${clampedIndex + 1}`
              const executionInfo = executionState[stepId]

              return (
                <Space align="start" style={{ width: '100%' }} size="large">
                  <Timeline style={{ minWidth: 240, marginTop: 8 }} items={timelineItems} />

                  <Card title={stepValue.title || `Step ${clampedIndex + 1}`} style={{ flex: 1 }}>
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                      <Form.Item name={[activeField.name, 'step_id']} hidden>
                        <Input />
                      </Form.Item>
                      <Form.Item
                         name={[activeField.name, 'title']}
                         label="Title"
                         rules={[{ required: true, message: 'Please enter a step title' }]}
                       >
                         <Input placeholder="Step name" disabled={readOnly} />
                       </Form.Item>
                       <Form.Item name={[activeField.name, 'description']} label="Details">
                         <TextArea
                           rows={3}
                           placeholder="Provide additional execution details"
                           disabled={readOnly}
                         />
                       </Form.Item>
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        <Text type="secondary">Service</Text>
                        <Radio.Group
                          value={localServiceSelection}
                          onChange={(event) => handleServiceChange(event.target.value)}
                          disabled={readOnly}
                          style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}
                        >
                          {serviceOptions.map(option => (
                            <Radio.Button
                              key={option.value}
                              value={option.value}
                              style={{ marginBottom: 8 }}
                            >
                              {option.label}
                            </Radio.Button>
                          ))}
                        </Radio.Group>
                        <Form.Item name={[activeField.name, 'tools']} label="Tool Chain" style={{ marginBottom: 0 }}>
                          <Select
                            mode="multiple"
                            showSearch
                            maxTagCount="responsive"
                            placeholder="Select tools to be used in this step"
                            options={toolOptions}
                            optionLabelProp="value"
                            dropdownMatchSelectWidth={false}
                            style={{ width: '100%' }}
                            value={localToolSelection}
                            onChange={(values) => {
                              const next = Array.from(new Set(values as string[]))
                              setLocalToolSelection(next)
                              form.setFieldValue(['steps', activeField.name, 'tools'], next)
                            }}
                            disabled={readOnly}
                            dropdownRender={(menu) => (
                              <div>
                                {React.Children.map(menu, child => {
                                  if (!React.isValidElement(child)) return child
                                  const optionProps: any = child.props
                                  const toolValue = optionProps?.value
                                  const displayLabel = optionProps?.children
                                  const description = toolDescriptionMap.get(String(toolValue))
                                  const tooltipTitle = description && description !== displayLabel ? description : displayLabel
                                  return (
                                    <Tooltip key={toolValue} title={tooltipTitle} placement="right">
                                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {child}
                                      </div>
                                    </Tooltip>
                                  )
                                })}
                              </div>
                            )}
                            tagRender={(tagProps) => {
                              const { label, value, closable, onClose } = tagProps
                              const resolvedLabel = typeof label === 'string' ? label : String(label)
                              const tooltipTitle = toolDescriptionMap.get(String(value)) || resolvedLabel
                              return (
                                <Tooltip title={tooltipTitle} key={value as string}>
                                  <Tag closable={closable} onClose={onClose} style={{ marginRight: 4 }}>
                                    {label}
                                  </Tag>
                                </Tooltip>
                              )
                            }}
                          />
                        </Form.Item>
                      </Space>

                      {executionInfo?.toolsCalled?.length ? (
                        <Card type="inner" title="Tool Calls">
                          <Space direction="vertical" style={{ width: '100%' }} size={8}>
                            {executionInfo.toolsCalled.map((call: any, callIndex: number) => {
                              const toolName = call?.tool_name || call?.name || `Tool ${callIndex + 1}`
                              const toolKey = call?.call_id || call?.tool_call_id || `${toolName}_${callIndex}`
                              const matchingResult = (executionInfo.toolResults || []).find((result: any, resultIndex: number) => {
                                if (result?.call_id && call?.call_id) {
                                  return result.call_id === call.call_id
                                }
                                if (result?.tool_call_id && call?.tool_call_id) {
                                  return result.tool_call_id === call.tool_call_id
                                }
                                if (result?.tool_name && call?.tool_name) {
                                  return result.tool_name === call.tool_name
                                }
                                return resultIndex === callIndex
                              })
                              const resultContent = matchingResult?.result || matchingResult?.content
                              const downloadInfo = matchingResult?.download_info || call?.download_info
                              const reasoning = matchingResult?.reasoning
                              // 推断评估进度Tag（基于step级别message关键字）
                              const stepMessage = executionInfo?.message || ''
                              let reasoningPhase: string | null = null
                              if (stepMessage.includes('进入结果评估')) reasoningPhase = 'reasoning_started'
                              else if (stepMessage.includes('评估中')) reasoningPhase = 'reasoning_running'
                              else if (stepMessage.includes('评估完成')) reasoningPhase = 'reasoning_completed'
                              else if (stepMessage.includes('评估失败')) reasoningPhase = 'reasoning_failed'
                              const reasoningPhaseColor =
                                reasoningPhase === 'reasoning_started' ? 'default' :
                                reasoningPhase === 'reasoning_running' ? 'processing' :
                                reasoningPhase === 'reasoning_completed' ? 'success' :
                                reasoningPhase === 'reasoning_failed' ? 'error' : 'default'
                              return (
                                <Card key={toolKey} size="small" type="inner" bordered={false} style={{ background: '#fafafa' }}>
                                  <Space direction="vertical" style={{ width: '100%' }} size={4}>
                                    <Space>
                                      <Text strong>{toolName}</Text>
                                      {renderToolCallStatusTag(call?.status)}
                                      {reasoningPhase ? (
                                        <Tag color={reasoningPhaseColor}>
                                          {reasoningPhase.replace('reasoning_', '').replace('_', ' ')}
                                        </Tag>
                                      ) : null}
                                    </Space>
                                    {call?.error ? (
                                      <Text type="danger">{call.error}</Text>
                                    ) : null}
                                    {call?.args !== undefined ? (
                                      <Paragraph
                                        type="secondary"
                                        style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}
                                        ellipsis={{ rows: 3, expandable: true, symbol: 'More' }}
                                      >
                                        {formatArgsPreview(call.args)}
                                      </Paragraph>
                                    ) : null}
                                    {downloadInfo ? (
                                      <Text type="secondary">
                                        Downloaded to: {downloadInfo.destination || JSON.stringify(downloadInfo)}
                                        {Array.isArray(downloadInfo.files) && downloadInfo.files.length ? (
                                          <>
                                            <br />Files: {downloadInfo.files.length}
                                          </>
                                        ) : null}
                                      </Text>
                                    ) : null}
                                    {resultContent ? (
                                      <Paragraph
                                        style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}
                                        ellipsis={{ rows: 4, expandable: true, symbol: 'More' }}
                                      >
                                        {typeof resultContent === 'string'
                                          ? resultContent
                                          : JSON.stringify(resultContent, null, 2)}
                                      </Paragraph>
                                    ) : null}
                                    {reasoning ? (
                                      <Card size="small" style={{ background: '#fff' }} title="Reasoning 反馈">
                                        <Space direction="vertical" style={{ width: '100%' }} size={4}>
                                          <Space wrap>
                                            <Tag color={
                                              String(reasoning.status || '').toLowerCase() === 'valid' ? 'green' :
                                              String(reasoning.status || '').toLowerCase() === 'invalid' ? 'red' :
                                              'gold'
                                            }>
                                              {String(reasoning.status || 'uncertain').toUpperCase()}
                                            </Tag>
                                            {typeof reasoning.confidence === 'number' ? (
                                              <Tag>confidence: {(reasoning.confidence as number).toFixed(2)}</Tag>
                                            ) : null}
                                          </Space>
                                          {reasoning.rationale ? (
                                            <Paragraph style={{ marginBottom: 0 }}>
                                              {String(reasoning.rationale)}
                                            </Paragraph>
                                          ) : null}
                                          {Array.isArray(reasoning.issues) && reasoning.issues.length ? (
                                            <Space direction="vertical" size={0} style={{ width: '100%' }}>
                                              <Text strong>Issues</Text>
                                              <ul style={{ margin: 0, paddingLeft: 18 }}>
                                                {reasoning.issues.map((it: any, idx: number) => (
                                                  <li key={idx}>
                                                    <Text>{String(it)}</Text>
                                                  </li>
                                                ))}
                                              </ul>
                                            </Space>
                                          ) : null}
                                          {Array.isArray(reasoning.recommended_actions) && reasoning.recommended_actions.length ? (
                                            <Space direction="vertical" size={0} style={{ width: '100%' }}>
                                              <Text strong>Recommended actions</Text>
                                              <ul style={{ margin: 0, paddingLeft: 18 }}>
                                                {reasoning.recommended_actions.map((it: any, idx: number) => (
                                                  <li key={idx}>
                                                    <Text>{String(it)}</Text>
                                                  </li>
                                                ))}
                                              </ul>
                                            </Space>
                                          ) : null}
                                        </Space>
                                      </Card>
                                    ) : null}
                                  </Space>
                                </Card>
                              )
                            })}
                          </Space>
                        </Card>
                      ) : null}

                      {executionInfo?.output ? (
                        <Card type="inner" title="Latest Output">
                          <Paragraph ellipsis={{ rows: 6, expandable: true, symbol: 'More' }}>
                            {typeof executionInfo.output === 'string'
                              ? executionInfo.output
                              : JSON.stringify(executionInfo.output, null, 2)}
                          </Paragraph>
                        </Card>
                      ) : null}
                      {executionInfo?.message ? (
                        <Text type="danger">{executionInfo.message}</Text>
                      ) : null}
                    </Space>
                  </Card>
                </Space>
              )
            }}
          </Form.List>
        </Form>
      ) : (
        <Paragraph type="secondary">No plan data available. The workflow has not produced a plan yet.</Paragraph>
      )}
    </Drawer>
  )
}


