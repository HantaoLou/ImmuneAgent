import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChatInput, ChatMessages, ChatSection } from '@llamaindex/chat-ui'
import '@llamaindex/chat-ui/styles/markdown.css'
import '@llamaindex/chat-ui/styles/pdf.css'
import '@llamaindex/chat-ui/styles/editor.css'
import { useChat } from '@ai-sdk/react'
import { Button, Tooltip, message, notification, Space } from 'antd'
import { FolderOutlined, ProjectOutlined } from '@ant-design/icons'
import {
  getChatHistory,
  getSessionById,
  type ChatHistoryResponse,
} from '../../services/sessions-service'
import { type UIMessage } from 'ai'
import { ArtifactsDrawer } from './artifacts-drawer-component'
import { ActionRequestComponent } from './action-request-component'
import { ActionAwareChatTransport } from './action-aware-transport'
import {
  PlanReviewDrawer,
  type PlanSummaryData,
  type ExecutionStateMap,
  type PlanStepFormValues,
} from './plan-review-drawer'

interface ChatInterfaceProps {
  sessionId: string
}

function toMessage(history: ChatHistoryResponse): UIMessage {
  const role =
    history.role === 'ai'
      ? 'assistant'
      : (history.role as 'user' | 'assistant' | 'system')
  return {
    id: `${history.session_id}-${history.timestamp}`,
    role,
    parts: [{ text: history.message ?? '', type: 'text' }],
  }
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  sessionId,
}: ChatInterfaceProps) => {
  const [artifactsDrawerVisible, setArtifactsDrawerVisible] = useState(false)
  const [pendingAction, setPendingAction] = useState<any>(null)
  const [_processedActionIds, setProcessedActionIds] = useState<Set<string>>(new Set())
  const processedActionIdsRef = useRef<Set<string>>(new Set())
  const [planSummary, setPlanSummary] = useState<PlanSummaryData | null>(null)
  const [planExecutionState, setPlanExecutionState] = useState<ExecutionStateMap>({})
  const [planDrawerVisible, setPlanDrawerVisible] = useState(false)
  const [planDrawerMode, setPlanDrawerMode] = useState<'view' | 'confirm'>('view')
  const [planConfirmRequest, setPlanConfirmRequest] = useState<any>(null)
  const [planConfirmActionId, setPlanConfirmActionId] = useState<string | number | null>(null)
  // 工具调用请求队列：当有未确认的计划时，将工具调用请求放入队列
  const toolActionQueueRef = useRef<any[]>([])
  // 计划确认超时定时器（10分钟 = 600秒，与后端超时时间一致）
  const planConfirmTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // 存储 handlePlanDrawerConfirm 的引用，以便在超时回调中访问
  const handlePlanDrawerConfirmRef = useRef<((args: {
    planId?: string | null
    planText?: string
    steps: PlanStepFormValues[]
    actionId?: string | number
  }) => Promise<void>) | null>(null)

  const transportInstanceRef = useRef<ActionAwareChatTransport | null>(null)

  const normalizePlanSummary = useCallback(
    (raw: any): PlanSummaryData | null => {
      if (!raw || typeof raw !== 'object') return null
      const steps = (raw.steps ?? []).map((step: any, index: number) => ({
        step_id: step?.step_id ?? step?.stepId ?? `step-${index + 1}`,
        title: step?.title ?? '',
        description: step?.description ?? '',
        objective: step?.objective ?? '',
        tools: step?.tools ?? step?.toolchain ?? [],
        recommended_tools: step?.recommended_tools ?? [],
        notes: step?.notes ?? '',
        inputs: step?.inputs ?? [],
        outputs: step?.outputs ?? [],
        status: step?.status ?? 'pending',
      }))
      return {
        planId: raw?.planId ?? raw?.plan_id ?? null,
        totalSteps: raw?.totalSteps ?? steps.length,
        originalQuestion: raw?.originalQuestion ?? raw?.original_question ?? '',
        planText: raw?.planText ?? raw?.plan_text ?? '',
        steps,
      }
    },
    [],
  )

  const handleToggleDrawer = () => {
    setArtifactsDrawerVisible(!artifactsDrawerVisible)
  }

  const handleDrawerClose = () => {
    setArtifactsDrawerVisible(false)
  }

  // 将 registerActionRequest 提取为 useCallback，以便在其他函数中访问
  const registerActionRequest = useCallback((actionData: any) => {
    if (!actionData) return
    // 使用 timestamp 作为唯一标识，因为每次请求都会有新的timestamp
    // 这样可以避免因为action_id相同而被跳过
    const actionId =
      actionData.timestamp ||  // 优先使用timestamp，因为它是动态生成的
      actionData.action_id ||
      `${actionData.tool_name}_${Date.now()}` ||  // 如果都没有，使用tool_name + 时间戳
      JSON.stringify(actionData)
    const resolvedActionId = String(actionId)
    
    // 检查是否已处理过（使用timestamp作为唯一标识，因为每次请求都有新的timestamp）
    if (processedActionIdsRef.current.has(resolvedActionId)) {
      console.log('ChatInterface: action请求已处理过，跳过', resolvedActionId, actionData)
      return
    }
    
    console.log('[ChatInterface] registerActionRequest incoming', {
      raw: actionData,
      resolvedActionId,
      type: actionData.type,
      timestamp: actionData.timestamp,
    })
    
    // 添加到已处理列表
    processedActionIdsRef.current.add(resolvedActionId)
    setProcessedActionIds(prev => {
      if (prev.has(resolvedActionId)) {
        return prev
      }
      return new Set([...prev, resolvedActionId])
    })

    actionData.__action_id = resolvedActionId
    if (actionData.action_id === undefined || actionData.action_id === null || actionData.action_id === '') {
      actionData.action_id = resolvedActionId
    }

    if (actionData.type === 'plan_confirmation_request') {
      console.log('ChatInterface: 收到计划确认请求')
      console.log('[ChatInterface] setPlanConfirmActionId', resolvedActionId)
      setPlanConfirmRequest(actionData)
      setPlanConfirmActionId(resolvedActionId)
      setPlanDrawerMode('confirm')
      setPlanDrawerVisible(true)
      
      // 清除之前的超时定时器（如果存在）
      if (planConfirmTimeoutRef.current) {
        clearTimeout(planConfirmTimeoutRef.current)
      }
      
      // 设置10分钟超时自动确认（与后端超时时间一致）
      // 使用闭包捕获当前值
      const currentPlanConfirmRequest = actionData
      const currentPlanConfirmActionId = resolvedActionId
      planConfirmTimeoutRef.current = setTimeout(() => {
        console.log('ChatInterface: 计划确认超时，自动确认计划')
        // 使用 setState 的回调形式获取最新状态
        setPlanSummary(prevSummary => {
          if (prevSummary && handlePlanDrawerConfirmRef.current) {
            // 自动确认计划
            handlePlanDrawerConfirmRef.current({
              planId: prevSummary.planId ?? null,
              planText: prevSummary.planText ?? '',
              steps: prevSummary.steps ?? [],
              actionId: currentPlanConfirmActionId || currentPlanConfirmRequest?.__action_id || currentPlanConfirmRequest?.timestamp || currentPlanConfirmRequest?.action_id,
            }).catch(err => {
              console.error('自动确认计划失败:', err)
            })
          }
          return prevSummary
        })
      }, 10 * 60 * 1000) // 10分钟 = 600000毫秒
      
      return
    }

    // 如果是工具调用请求，检查是否有未确认的计划
    if (actionData.type === 'tool_action_request' || actionData.type === 'action-request' || actionData.type === 'reasoning_decision_request') {
      // 如果有未确认的计划，将工具调用请求放入队列
      if (planConfirmRequest || planConfirmActionId) {
        console.log('ChatInterface: 计划未确认，将工具调用请求放入队列', actionData)
        toolActionQueueRef.current.push(actionData)
        return
      }
    }

    console.log('ChatInterface: 处理新的action请求', actionData)
    console.log('ChatInterface: 设置pendingAction，actionData.type=', actionData.type, 'tool_name=', actionData.tool_name)
    setPendingAction(actionData)
    console.log('ChatInterface: pendingAction已设置')
  }, [planConfirmRequest, planConfirmActionId])

  const transportInstance = useMemo(() => {
    const transport = new ActionAwareChatTransport({
      api: '/api/chat',
      body: {
        session_id: sessionId,
      },
      headers: {
        Authorization: `Bearer ${localStorage.getItem('auth_token')}`,
      },
    }) as any

    transportInstanceRef.current = transport

    transport.setOnActionRequest((actionData: any) => {
      registerActionRequest(actionData)
    })

    transport.setOnPlanSummary((planData: PlanSummaryData) => {
      if (!planData) return
      console.log('ChatInterface: 捕获计划摘要事件', planData)
      const normalized = normalizePlanSummary(planData)
      if (normalized) {
        setPlanSummary(normalized)
        const initialExecutionState: ExecutionStateMap = {}
        ;(normalized.steps ?? []).forEach((step, idx) => {
          const stepId = step.step_id || `step-${idx + 1}`
          initialExecutionState[stepId] = {
            status: step.status || 'pending',
          }
        })
        setPlanExecutionState(initialExecutionState)
      }
    })

    transport.setOnPlanConfirmRequest((planData: PlanSummaryData) => {
      console.log('ChatInterface: 收到计划确认请求事件', planData)
      setPlanConfirmRequest((prev: any) => prev ?? planData)
      if (planData?.steps) {
        setPlanSummary(prev => ({
          planId: planData.planId ?? (planData as any).plan_id ?? prev?.planId ?? null,
          totalSteps: planData.totalSteps ?? planData.steps.length,
          originalQuestion: planData.originalQuestion ?? prev?.originalQuestion ?? '',
          planText: planData.planText ?? (planData as any).plan_text ?? prev?.planText ?? '',
          steps: planData.steps ?? prev?.steps ?? [],
        }))
      }
      const inferredActionId =
        (planData as any)?.timestamp ||
        (planData as any)?.action_id ||
        planConfirmActionId
      if (inferredActionId) {
        setPlanConfirmActionId(inferredActionId)
      }
      setPlanDrawerMode('confirm')
      setPlanDrawerVisible(true)
    })

    transport.setOnExecutionProgress((progressData: any) => {
      if (!progressData) return
      const stepId = progressData.stepId || `step-${progressData.index || 0}`
      const status = progressData.status || 'running'
      setPlanExecutionState(prev => {
        const prevState = prev[stepId] || {}
        const nextToolsCalled =
          progressData.toolsCalled !== undefined ? progressData.toolsCalled : prevState.toolsCalled || []
        const nextToolResults =
          progressData.toolResults !== undefined ? progressData.toolResults : prevState.toolResults || []
        const nextState = {
          ...prevState,
          status,
          toolsCalled: nextToolsCalled,
          toolResults: nextToolResults,
        }
        if (progressData.output !== undefined) {
          nextState.output = progressData.output
        }
        if (progressData.message !== undefined) {
          nextState.message = progressData.message
        }
        return {
          ...prev,
          [stepId]: nextState,
        }
      })
      setPlanSummary(prev => {
        if (!prev) return prev
        const updatedSteps = prev.steps.map((step, idx) => {
          const currentId = step.step_id || `step-${idx + 1}`
          if (currentId === stepId) {
            return { ...step, status }
          }
          return step
        })
        return { ...prev, steps: updatedSteps }
      })
    })

    transport.setOnExecutionError((errorData: any) => {
      if (!errorData) return
      const stepId = errorData.stepId || `step-${errorData.index || 0}`
      notification.error({
        message: errorData.message || '执行步骤时发生错误',
        description: errorData.description || errorData.reason || '请检查步骤配置或手动干预。',
        duration: 6,
      })
      setPlanExecutionState(prev => {
        const prevState = prev[stepId] || {}
        const nextToolsCalled =
          errorData.toolsCalled !== undefined ? errorData.toolsCalled : prevState.toolsCalled || []
        const nextToolResults =
          errorData.toolResults !== undefined ? errorData.toolResults : prevState.toolResults || []
        return {
          ...prev,
          [stepId]: {
            ...prevState,
            status: errorData.status || 'failed',
            message: errorData.message || errorData.reason,
            toolsCalled: nextToolsCalled,
            toolResults: nextToolResults,
          },
        }
      })
      setPlanSummary(prev => {
        if (!prev) return prev
        const updatedSteps = prev.steps.map((step, idx) => {
          const currentId = step.step_id || `step-${idx + 1}`
          if (currentId === stepId) {
            return { ...step, status: errorData.status || 'failed' }
          }
          return step
        })
        return { ...prev, steps: updatedSteps }
      })

      if (errorData.actionRequest) {
        registerActionRequest(errorData.actionRequest)
      }
    })

    return transport
  }, [sessionId, normalizePlanSummary, registerActionRequest])

  const handler = useChat({
    transport: transportInstance,
  })

  const sendActionResponse = useCallback(
    async (actionId: string | number, responsePayload: any) => {
      const responseData = {
        session_id: sessionId,
        action_id: actionId,
        response: responsePayload,
      }

      const result = await fetch('/api/chat/action-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`,
        },
        body: JSON.stringify(responseData),
      })

      if (!result.ok) {
        throw new Error(await result.text())
      }
    },
    [sessionId],
  )

  const handleActionResponse = async (response: { type: string; args?: any }) => {
    if (!pendingAction) return

    try {
      await sendActionResponse(pendingAction.timestamp, response)
      setPendingAction(null)
      if (pendingAction?.timestamp) {
        setProcessedActionIds(prev => {
          const newSet = new Set(prev)
          newSet.delete(pendingAction.timestamp)
          return newSet
        })
      }
      message.success('操作响应已发送')
    } catch (error) {
      console.error('发送action响应失败:', error)
      message.error('发送响应失败')
    }
  }

  const handlePlanRejection = useCallback(async () => {
    if (!planConfirmRequest) {
      setPlanDrawerVisible(false)
      return
    }
    const actionId =
      planConfirmRequest.timestamp ||
      planConfirmRequest.action_id ||
      planConfirmActionId
    if (!actionId) {
      message.warning('正在准备确认上下文，请稍后再试。')
      return
    }
    const payload = {
      type: 'plan-confirmation',
      status: 'rejected',
    }
    try {
      await sendActionResponse(actionId, payload)
      // 先处理队列中的工具调用请求（在清空计划确认状态之前）
      if (toolActionQueueRef.current.length > 0) {
        console.log('ChatInterface: 计划已拒绝，处理队列中的工具调用请求', toolActionQueueRef.current.length)
        const queuedAction = toolActionQueueRef.current.shift()
        if (queuedAction) {
          // 延迟一小段时间，确保计划确认状态已更新后再处理
          setTimeout(() => {
            // 直接设置 pendingAction，因为此时计划确认状态已清空
            setPendingAction(queuedAction)
          }, 100)
        }
      }
      
      // 清除超时定时器
      if (planConfirmTimeoutRef.current) {
        clearTimeout(planConfirmTimeoutRef.current)
        planConfirmTimeoutRef.current = null
      }
      
      setPlanConfirmRequest(null)
      setPlanConfirmActionId(null)
      setPlanDrawerVisible(false)
      setPlanDrawerMode('view')
      message.success('已拒绝当前计划')
    } catch (error) {
      console.error('发送计划拒绝失败:', error)
      message.error('计划拒绝发送失败')
    }
  }, [planConfirmRequest, planConfirmActionId, sendActionResponse, message, registerActionRequest])

  const handlePlanDrawerConfirm = useCallback(
    async ({
      planId,
      planText,
      steps,
      actionId: incomingActionId,
    }: {
      planId?: string | null
      planText?: string
      steps: PlanStepFormValues[]
      actionId?: string | number
    }) => {
      if (!planConfirmRequest && !planConfirmActionId && !incomingActionId) {
        message.warning('当前没有需要确认的计划。')
        setPlanDrawerVisible(false)
        setPlanDrawerMode('view')
        return
      }
      const actionId =
        incomingActionId ||
        planConfirmActionId ||
        planConfirmRequest?.__action_id ||
        planConfirmRequest?.timestamp ||
        planConfirmRequest?.action_id
      if (!actionId) {
        message.warning('正在准备确认上下文，请稍后再试。')
        return
      }
      console.log('[ChatInterface] Submitting plan confirmation', {
        actionId,
        planId,
        planText,
        steps,
      })
      const requestPayload = {
        type: 'plan-confirmation',
        status: 'confirmed',
        plan: {
          planId: planId ?? planSummary?.planId ?? null,
          planText: planText ?? '',
          steps: steps ?? [],
        },
        steps: steps ?? [],
        planText: planText ?? '',
      }
      try {
        await sendActionResponse(actionId, requestPayload)
        setPlanSummary(prev =>
          prev
            ? {
                ...prev,
                planText: planText ?? prev.planText,
                steps: steps ?? prev.steps,
              }
            : prev,
        )
        // 先处理队列中的工具调用请求（在清空计划确认状态之前）
        if (toolActionQueueRef.current.length > 0) {
          console.log('ChatInterface: 计划已确认，处理队列中的工具调用请求', toolActionQueueRef.current.length)
          const queuedAction = toolActionQueueRef.current.shift()
          if (queuedAction) {
            // 延迟一小段时间，确保计划确认状态已更新后再处理
            setTimeout(() => {
              // 直接设置 pendingAction，因为此时计划确认状态已清空
              setPendingAction(queuedAction)
            }, 100)
          }
        }
        
        // 清除超时定时器
        if (planConfirmTimeoutRef.current) {
          clearTimeout(planConfirmTimeoutRef.current)
          planConfirmTimeoutRef.current = null
        }
        
        setPlanConfirmRequest(null)
        setPlanConfirmActionId(null)
        setPlanDrawerVisible(false)
        setPlanDrawerMode('view')
        message.success('计划已确认')
      } catch (error) {
        console.error('发送计划确认失败:', error)
        message.error('计划确认发送失败')
      }
    },
    [planConfirmRequest, planConfirmActionId, planSummary, sendActionResponse, registerActionRequest, message],
  )
  
  // 更新 handlePlanDrawerConfirm 的 ref，以便超时回调可以访问
  useEffect(() => {
    handlePlanDrawerConfirmRef.current = handlePlanDrawerConfirm
  }, [handlePlanDrawerConfirm])
  
  // 组件卸载时清除超时定时器
  useEffect(() => {
    return () => {
      if (planConfirmTimeoutRef.current) {
        clearTimeout(planConfirmTimeoutRef.current)
        planConfirmTimeoutRef.current = null
      }
    }
  }, [])

  // 调试：监听pendingAction变化
  useEffect(() => {
    if (pendingAction) {
      console.log('[ChatInterface] pendingAction状态变化:', { 
        type: pendingAction.type, 
        tool_name: pendingAction.tool_name, 
        timestamp: pendingAction.timestamp 
      })
    } else {
      console.log('[ChatInterface] pendingAction已清空')
    }
  }, [pendingAction])

  useEffect(() => {
    console.log('[ChatInterface] Plan confirm state update', {
      planConfirmActionId,
      planConfirmRequest,
      planDrawerVisible,
      planDrawerMode,
    })
  }, [planConfirmActionId, planConfirmRequest, planDrawerVisible, planDrawerMode])

  useEffect(() => {
    let isMounted = true
    getChatHistory(sessionId).then((messages) => {
      if (!isMounted) return
      handler.setMessages(messages.map(toMessage))
    })
    getSessionById(sessionId)
      .then((sessionData) => {
        if (!isMounted || !sessionData) return
        let configuration: any = sessionData.configuration
        if (typeof configuration === 'string') {
          try {
            configuration = JSON.parse(configuration)
          } catch (error) {
            console.error('Failed to parse session configuration:', error)
            configuration = {}
          }
        }
        const planState = configuration?.plan_state
        if (!planState) return
        const normalizedSummary = normalizePlanSummary(planState.summary)
        if (normalizedSummary) {
          const executionState = planState.execution_state ?? {}
          const mergedSteps = normalizedSummary.steps.map((step) => {
            const state = executionState[step.step_id ?? '']
            if (state?.status) {
              return { ...step, status: state.status }
            }
            return step
          })
          setPlanSummary({ ...normalizedSummary, steps: mergedSteps })
          setPlanExecutionState(executionState)
        }
      })
      .catch((error) => {
        console.error('Failed to load session plan state:', error)
      })
    return () => {
      isMounted = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, normalizePlanSummary])

  useEffect(() => {
    const fixInputHeight = () => {
      const chatSection = document.querySelector('[class*="chat"]') || document.querySelector('section')
      if (!chatSection) return

      const textareas = chatSection.querySelectorAll('textarea')
      const inputs = chatSection.querySelectorAll('input[type="text"]')

      const applyFixedHeight = (element: Element) => {
        if (element instanceof HTMLElement) {
          element.style.minHeight = '40px'
          element.style.maxHeight = '40px'
          element.style.height = '40px'
          element.style.resize = 'none'
          element.style.overflow = 'hidden'
          element.style.boxSizing = 'border-box'
        }
      }

      textareas.forEach(applyFixedHeight)
      inputs.forEach(applyFixedHeight)
    }

    fixInputHeight()

    const observer = new MutationObserver(() => {
      fixInputHeight()
    })

    const chatSection = document.querySelector('[class*="chat"]') || document.querySelector('section')
    if (chatSection) {
      observer.observe(chatSection, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style'],
      })
    }

    const messageObserver = new MutationObserver(() => {
      fixInputHeight()
    })

    const messagesContainer = document.querySelector('[class*="messages"]') || document.querySelector('[class*="Messages"]')
    if (messagesContainer) {
      messageObserver.observe(messagesContainer, {
        childList: true,
        subtree: true,
      })
    }

    return () => {
      observer.disconnect()
      messageObserver.disconnect()
    }
  }, [handler.messages])

  return (
    <div
      style={{
        height: 'calc(100vh - 120px)',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <ChatSection handler={handler}>
          <ChatMessages>
            <ChatMessages.List></ChatMessages.List>
            <ChatMessages.Loading />
          </ChatMessages>

          {/* ActionRequestComponent 放在 ChatMessages 和 ChatInput 之间 */}
          {pendingAction && (
            <div style={{
              position: 'relative',
              zIndex: 1001,
              backgroundColor: '#fff',
              padding: '0 16px',
              flexShrink: 0,
              margin: '16px 0',
            }}>
              <ActionRequestComponent
                actionData={pendingAction}
                onResponse={handleActionResponse}
                sessionId={sessionId}
              />
            </div>
          )}
          
          <ChatInput>
            <ChatInput.Form>
              <div style={{ position: 'relative', width: '100%' }}>
                <ChatInput.Field placeholder="Type your message..." />
                <div
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '8px',
                    zIndex: 10,
                  }}
                >
                  <Space size="small">
                    <Tooltip title={artifactsDrawerVisible ? 'Hide attachments' : 'View attachments'}>
                      <Button
                        type="text"
                        shape="circle"
                        icon={<FolderOutlined />}
                        onClick={handleToggleDrawer}
                        style={{
                          width: '32px',
                          height: '32px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: artifactsDrawerVisible ? '#1890ff' : 'rgba(255, 255, 255, 0.9)',
                          border: '1px solid #d9d9d9',
                          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                          color: artifactsDrawerVisible ? '#fff' : '#000',
                        }}
                      />
                    </Tooltip>
                    <Tooltip title={planSummary ? 'View execution plan' : 'No plan generated yet'}>
                      <Button
                        type="text"
                        shape="circle"
                        icon={<ProjectOutlined />}
                        disabled={!planSummary}
                        onClick={() => {
                          setPlanDrawerMode(planConfirmRequest ? 'confirm' : 'view')
                          setPlanDrawerVisible(true)
                        }}
                        style={{
                          width: '32px',
                          height: '32px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: planDrawerVisible ? '#1890ff' : 'rgba(255, 255, 255, 0.9)',
                          border: '1px solid #d9d9d9',
                          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                          color: planDrawerVisible ? '#fff' : '#000',
                        }}
                      />
                    </Tooltip>
                  </Space>
                </div>
              </div>
              <ChatInput.Upload />
              <ChatInput.Submit />
            </ChatInput.Form>
          </ChatInput>
        </ChatSection>
      </div>

      <ArtifactsDrawer
        sessionId={sessionId}
        visible={artifactsDrawerVisible}
        onClose={handleDrawerClose}
      />
      <PlanReviewDrawer
        open={planDrawerVisible}
        mode={planConfirmRequest ? 'confirm' : planDrawerMode}
        planSummary={planSummary}
        executionState={planExecutionState}
        confirmActionId={
          planConfirmActionId ||
          planConfirmRequest?.__action_id ||
          planConfirmRequest?.timestamp ||
          planConfirmRequest?.action_id ||
          undefined
        }
        onClose={() => {
          if (planConfirmRequest) {
            message.info('Please confirm or reject the plan to continue.')
          } else {
            setPlanDrawerVisible(false)
          }
        }}
        onConfirm={handlePlanDrawerConfirm}
        onReject={planConfirmRequest ? handlePlanRejection : undefined}
      />
    </div>
  )
}

export default ChatInterface
