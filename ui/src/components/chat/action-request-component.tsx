import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Card, Button, Space, Typography, Descriptions, Modal, Form, Input, App, Select, Radio, InputNumber, Tag, Alert } from 'antd'
import { CheckOutlined, EditOutlined, CloseOutlined, ArrowDownOutlined } from '@ant-design/icons'
import { FileUpload } from '../common/file-upload-component'
import { ArrayInput } from '../common/array-input-component'
import { getToolDemoData, hasToolDemoData } from '../../config/tool-demo-data'

const { TextArea } = Input

const { Text } = Typography

interface ActionRequestProps {
  actionData: {
    type: string
    tool_name: string
    tool_args: Record<string, any>
    tool_info: {
      name: string
      description: string
      args_schema: any
      service_id?: string | null
    }
    description: string
    timestamp: string
    mode?: string
    error?: string
    step_id?: string
    task?: string
    reasoning?: {
      status: string
      confidence: number
      rationale: string
      recommended_actions?: string[]
    }
  }
  onResponse: (response: { type: string; action?: string; args?: any }) => void
  sessionId: string
}

export const ActionRequestComponent: React.FC<ActionRequestProps> = ({
  actionData,
  onResponse,
  sessionId,
}) => {
  const { message } = App.useApp()
  
  // 调试：组件挂载时打印日志
  useEffect(() => {
    console.log('[ActionRequestComponent] Component mounted', {
      type: actionData.type,
      tool_name: actionData.tool_name,
      timestamp: actionData.timestamp,
      mode: actionData.mode,
    })
    return () => {
      console.log('[ActionRequestComponent] Component unmounted')
    }
  }, [])
  
  const isRetryMode = useMemo(() => {
    return actionData.mode === 'retry' || actionData.type === 'tool_retry'
  }, [actionData.mode, actionData.type])
  
  const isReasoningDecisionMode = useMemo(() => {
    return actionData.type === 'reasoning_decision_request' || actionData.mode === 'reasoning_failed'
  }, [actionData.type, actionData.mode])
  
  // 初始化编辑状态：只有在非reasoning决策模式下的retry模式才默认打开编辑
  const [isEditing, setIsEditing] = useState(isRetryMode && !isReasoningDecisionMode)
  const [editedArgs, setEditedArgs] = useState(actionData.tool_args)
  const [form] = Form.useForm()
  // 记录哪些参数是通过use demo设置的（需要显示download按钮）
  const [demoParams, setDemoParams] = useState<Set<string>>(new Set())
  
  // 5分钟自动关闭定时器的引用
  const autoCloseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  
  // 5分钟自动关闭定时器
  useEffect(() => {
    // 清除之前的定时器（如果存在）
    if (autoCloseTimerRef.current) {
      clearTimeout(autoCloseTimerRef.current)
    }
    
    // 设置新的定时器
    autoCloseTimerRef.current = setTimeout(() => {
      message.warning('Tool invocation request timed out (5 minutes), auto-closing')
      onResponse({ type: 'abort', action: 'abort' })
      autoCloseTimerRef.current = null
    }, 5 * 60 * 1000) // 5分钟 = 300000毫秒

    // 清理函数：组件卸载或actionData变化时清除定时器
    return () => {
      if (autoCloseTimerRef.current) {
        clearTimeout(autoCloseTimerRef.current)
        autoCloseTimerRef.current = null
      }
    }
  }, [actionData.timestamp, onResponse, message])

  // 只有在非reasoning决策模式下的retry模式才自动打开编辑
  useEffect(() => {
    if (isRetryMode && !isReasoningDecisionMode) {
      setIsEditing(true)
    } else {
      setIsEditing(false)
    }
  }, [isRetryMode, isReasoningDecisionMode, actionData.timestamp])

  useEffect(() => {
    setEditedArgs(actionData.tool_args)
    form.setFieldsValue(actionData.tool_args || {})
  }, [actionData.tool_args, actionData.timestamp, form])

  const buildParamsFromFormValues = useCallback((values: Record<string, any>) => {
    const params: Record<string, any> = {}

    Object.keys(values).forEach((key) => {
      const value = values[key]
      if (value === undefined || value === null || value === '') {
        return
      }

      if (Array.isArray(value)) {
        params[key] = value
      }
      else if (typeof value === 'string' && value.trim().startsWith('{')) {
        try {
          params[key] = JSON.parse(value)
        } catch {
          params[key] = value
        }
      }
      else if (typeof value === 'string') {
        try {
          const parsed = JSON.parse(value)
          params[key] = parsed
        } catch {
          params[key] = value
        }
      }
      else {
        params[key] = value
      }
    })

    return params
  }, [])

  // 从 tool_name 推断 service_id 和 tool_name
  // tool_name 可能是 "service_id.tool_name" 格式，或者只是 "tool_name"
  const getServiceIdAndToolName = useMemo(() => {
    const toolName = actionData.tool_name || actionData.tool_info?.name || ''
    
    // 如果 tool_name 包含点号，尝试分割
    if (toolName.includes('.')) {
      const parts = toolName.split('.')
      if (parts.length >= 2) {
        return {
          serviceId: parts[0],
          toolName: parts.slice(1).join('.')
        }
      }
    }
    
    // 否则，尝试从所有服务中查找该工具
    // 这里我们使用一个常见的服务列表
    const commonServices = ['airr', 'anarci', 'sabdab', 'metabcr', 'af3', 'fdg', 'bcell', 'bioinformatics', 'annotation', 'scrna', 'lgblast', 'oas', 'geo']
    
    // 尝试从每个服务中查找该工具
    for (const serviceId of commonServices) {
      if (hasToolDemoData(serviceId, toolName)) {
        return { serviceId, toolName }
      }
    }
    
    // 如果找不到，返回 null
    return { serviceId: null, toolName }
  }, [actionData.tool_name, actionData.tool_info?.name])

  // 清除定时器的辅助函数
  const clearAutoCloseTimer = () => {
    if (autoCloseTimerRef.current) {
      clearTimeout(autoCloseTimerRef.current)
      autoCloseTimerRef.current = null
    }
  }

  const handleAccept = () => {
    clearAutoCloseTimer()
    onResponse({ type: 'accept' })
  }

  // 使用传入的 sessionId，不再创建新的临时session

  // 判断是否是输出路径参数
  const isOutputPathParam = (paramName: string, param: any): boolean => {
    const lowerName = paramName.toLowerCase()
    const lowerTitle = (param.title || '').toLowerCase()
    const lowerDesc = (param.description || '').toLowerCase()
    
    const outputKeywords = ['output_path', 'outputpath', 'output_dir', 'outputdir', 'output']
    return outputKeywords.some(keyword => 
      lowerName.includes(keyword) || 
      lowerTitle.includes(keyword) || 
      lowerDesc.includes(keyword)
    )
  }

  // 获取输出路径默认值
  const getOutputPathDefault = (): string => {
    if (!sessionId) {
      return '/opt/antibody_gen/artifacts/{session_id}'
    }
    return `/opt/antibody_gen/artifacts/${sessionId}`
  }

  // 当 sessionId 或工具 schema 变化时，更新输出路径参数的值
  useEffect(() => {
    if (isEditing && actionData.tool_info?.args_schema?.properties && sessionId) {
      const properties = actionData.tool_info.args_schema.properties
      const updates: Record<string, string> = {}
      
      Object.keys(properties).forEach((paramName) => {
        const param = properties[paramName]
        if (isOutputPathParam(paramName, param)) {
          updates[paramName] = `/opt/antibody_gen/artifacts/${sessionId}`
        }
      })
      
      if (Object.keys(updates).length > 0) {
        form.setFieldsValue(updates)
      }
    }
  }, [sessionId, actionData.tool_info?.args_schema, form, isEditing])

  // 解析 schema 中的 $ref 引用
  const resolveSchemaRefs = useCallback((schema: any, defs?: any): any => {
    if (!schema || typeof schema !== 'object') {
      return schema
    }

    // 如果 schema 有 $ref，需要解析引用
    if (schema.$ref) {
      const refPath = schema.$ref
      // 处理 "#/$defs/XXX" 格式的引用
      if (refPath.startsWith('#/$defs/')) {
        const defName = refPath.replace('#/$defs/', '')
        const schemaDefs = defs || {}
        const resolved = schemaDefs[defName]
        if (resolved) {
          // 递归解析 resolved 中的引用
          return resolveSchemaRefs(resolved, schemaDefs)
        }
      }
      return schema
    }

    // 递归处理对象和数组
    if (Array.isArray(schema)) {
      return schema.map(item => resolveSchemaRefs(item, defs))
    }

    const resolved: any = {}
    for (const key in schema) {
      if (key === '$defs') {
        // 保留 $defs，但不在这里处理
        resolved[key] = schema[key]
        continue
      }
      resolved[key] = resolveSchemaRefs(schema[key], defs)
    }

    return resolved
  }, [])

  // 获取展开后的参数 schema（处理 $ref 引用）
  const getResolvedParameterSchema = useMemo(() => {
    const argsSchema = actionData.tool_info?.args_schema
    if (!argsSchema) return null

    // 先解析整个 schema
    const resolved = resolveSchemaRefs(argsSchema, argsSchema.$defs)

    // 如果 properties 中只有一个参数，检查是否需要展开
    const properties = resolved.properties || {}
    const propertyKeys = Object.keys(properties)

    if (propertyKeys.length === 1) {
      const firstKey = propertyKeys[0]
      const firstParam = properties[firstKey]

      // 情况1: 如果这个参数有 $ref（还未解析），尝试从 $defs 中获取
      if (firstParam.$ref) {
        const refPath = firstParam.$ref
        if (refPath.startsWith('#/$defs/')) {
          const defName = refPath.replace('#/$defs/', '')
          const defs = argsSchema.$defs || {}
          const defSchema = defs[defName]

          if (defSchema && defSchema.properties) {
            // 返回展开后的参数定义
            return {
              properties: defSchema.properties,
              required: defSchema.required || [],
              title: resolved.title,
              type: resolved.type,
            }
          }
        }
      }
      
      // 情况2: 如果这个参数已经被解析为一个对象，且它有 properties
      if (firstParam && typeof firstParam === 'object' && !Array.isArray(firstParam) && firstParam.properties) {
        return {
          properties: firstParam.properties,
          required: firstParam.required || [],
          title: firstParam.title || resolved.title,
          type: firstParam.type || resolved.type,
        }
      }
    }

    // 否则返回原始解析后的 schema
    return {
      properties: resolved.properties || {},
      required: resolved.required || [],
      title: resolved.title,
      type: resolved.type,
    }
  }, [actionData.tool_info?.args_schema, resolveSchemaRefs])

  // 处理使用Demo（表单级别，填充整个表单）
  const handleUseDemoForForm = useCallback(() => {
    const { serviceId, toolName } = getServiceIdAndToolName
    
    if (!serviceId || !toolName) {
      message.warning('Unable to determine service ID or tool name, cannot load demo data')
      return
    }

    const demoData = getToolDemoData(serviceId, toolName)
    if (!demoData) {
      message.warning(`No demo data available: ${serviceId}/${toolName}`)
      return
    }

    // 获取解析后的 schema
    const resolvedSchema = getResolvedParameterSchema
    if (!resolvedSchema || !resolvedSchema.properties) {
      message.error('Unable to load tool schema')
      return
    }

    const properties = resolvedSchema.properties
    const formValues: Record<string, any> = {}
    const newDemoParams = new Set<string>()

    // 获取输出路径默认值
    const getOutputPathDefaultValue = (): string => {
      if (!sessionId) {
        return '/opt/antibody_gen/artifacts/{session_id}'
      }
      return `/opt/antibody_gen/artifacts/${sessionId}`
    }

    // 遍历所有参数，填充 demo 数据
    Object.keys(properties).forEach((paramName) => {
      const param = properties[paramName]
      const isOutputPath = isOutputPathParam(paramName, param)

      // 如果是输出路径参数，使用默认值
      if (isOutputPath) {
        formValues[paramName] = getOutputPathDefaultValue()
        return
      }

      // 如果 demo 数据中有该参数的值，使用 demo 数据
      if (paramName in demoData) {
        const demoValue = demoData[paramName]
        
        // 处理不同类型的值
        if (demoValue === null || demoValue === undefined) {
          // null 或 undefined，根据参数类型设置
          if (param.type === 'array') {
            formValues[paramName] = []
          } else if (param.type === 'object') {
            formValues[paramName] = {}
          } else {
            formValues[paramName] = ''
          }
        } else {
          formValues[paramName] = demoValue
          
          // 如果是文件路径类型（字符串且看起来像路径），标记为 demo 参数
          if (typeof demoValue === 'string' && (
            demoValue.startsWith('/opt/antibody_gen/artifacts/') ||
            demoValue.startsWith('/data_new/workspace/antibody_gen/') ||
            /^[A-Za-z]:[\\/]/.test(demoValue)
          )) {
            newDemoParams.add(paramName)
          }
        }
      } else {
        // demo 数据中没有该参数，使用默认值或空值
        if (param.default !== undefined) {
          formValues[paramName] = param.default
        } else if (param.type === 'array') {
          formValues[paramName] = []
        } else if (param.type === 'object') {
          formValues[paramName] = {}
        } else {
          formValues[paramName] = ''
        }
      }
    })

    // 批量设置表单值
    form.setFieldsValue(formValues)

    // 更新状态
    setDemoParams(newDemoParams)

    message.success('Demo data loaded successfully')
  }, [getServiceIdAndToolName, form, message, getResolvedParameterSchema, sessionId, isOutputPathParam])

  // 处理文件上传成功
  const handleFileUploadSuccess = useCallback((paramName: string, artifact: any) => {
    console.log('File upload success callback:', { paramName, artifact })
    
    // 使用确认上传后返回的预签名下载链接（8小时有效期）
    // 这是后端 confirm-upload 接口返回的 oss_direct_url
    let fileUrl: string
    
    if (artifact.oss_direct_url) {
      // 使用确认上传后返回的预签名下载URL（8小时有效期）
      fileUrl = artifact.oss_direct_url
      console.log('Using pre-signed download URL from confirm-upload:', fileUrl)
    } else {
      // 如果没有预签名URL，说明上传流程有问题
      console.error('Missing oss_direct_url in artifact response:', artifact)
      message.error('Upload succeeded but download link not received, please retry')
      return
    }
    
    console.log('Setting form field:', paramName, 'to value:', fileUrl)
    
    // 设置表单字段值
    form.setFieldsValue({ [paramName]: fileUrl })
    
    // 验证设置是否成功
    const updatedFields = form.getFieldsValue()
    const currentValue = updatedFields[paramName]
    
    if (currentValue !== fileUrl) {
      console.warn('Form field value mismatch. Expected:', fileUrl, 'Got:', currentValue)
      form.setFieldValue(paramName, fileUrl)
    } else {
      console.log('Form field value set successfully!')
    }
  }, [form, message])

  const handleEdit = () => {
    console.log('[ActionRequestComponent] Clicked edit parameters button, opening parameter edit dialog')
    setIsEditing(true)
    
    // 准备表单初始值，将对象和数组转换为字符串格式（用于 JSON 类型的输入框）
    const formInitialValues: Record<string, any> = {}
    const resolvedSchema = getResolvedParameterSchema
    
    Object.keys(editedArgs).forEach((key) => {
      const value = editedArgs[key]
      
      if (value === undefined || value === null) {
        formInitialValues[key] = ''
        return
      }
      
      // 获取参数的 schema 信息
      const param = resolvedSchema?.properties?.[key]
      const uiType = param?.ui_type
      const isArray = isArrayType(param)
      const isObject = isObjectType(param)
      
      // 如果是对象类型或 JSON 类型，转换为 JSON 字符串
      if (isObject || uiType === 'json') {
        if (typeof value === 'string') {
          // 已经是字符串，尝试解析验证
          try {
            JSON.parse(value)
            formInitialValues[key] = value
          } catch {
            // 不是有效的 JSON 字符串，可能是文件路径，直接使用
            formInitialValues[key] = value
          }
        } else {
          // 是对象，转换为 JSON 字符串
          formInitialValues[key] = JSON.stringify(value, null, 2)
        }
      }
      // 如果是数组类型且 ui_type 是 array_input，转换为 JSON 字符串
      else if (isArray && uiType === 'array_input') {
        if (typeof value === 'string') {
          // 已经是字符串，尝试解析验证
          try {
            JSON.parse(value)
            formInitialValues[key] = value
          } catch {
            // 不是有效的 JSON 字符串，直接使用
            formInitialValues[key] = value
          }
        } else if (Array.isArray(value)) {
          // 是数组，转换为 JSON 字符串
          formInitialValues[key] = JSON.stringify(value, null, 2)
        } else {
          formInitialValues[key] = ''
        }
      }
      // 其他类型直接使用
      else {
        formInitialValues[key] = value
      }
    })
    
    // 设置表单初始值
    form.setFieldsValue(formInitialValues)
  }

  const handleEditSubmit = () => {
    console.log('[ActionRequestComponent] Submitting parameter changes', { isReasoningDecisionMode })
    form.validateFields().then((values) => {
      const params = buildParamsFromFormValues(values)
      console.log('[ActionRequestComponent] Parameter validation passed', { params })

      setEditedArgs(params)
      setIsEditing(false)
      clearAutoCloseTimer()
      
      // 如果在reasoning决策模式下修改参数，应该发送retry响应而不是edit响应
      if (isReasoningDecisionMode) {
        console.log('[ActionRequestComponent] Reasoning decision mode: Parameter changes completed, sending retry response', { params })
        onResponse({
          type: 'retry',
          action: 'retry',
          args: params,
        })
      } else {
        console.log('[ActionRequestComponent] Normal mode: Parameter changes completed, sending edit response', { params })
        onResponse({ 
          type: 'edit', 
          args: { args: params } 
        })
      }
    }).catch((errorInfo) => {
      console.error('Validation failed:', errorInfo)
      message.error('Please fill in all required fields')
    })
  }

  const handleRetryConfirm = () => {
    // 在reasoning决策模式下，点击"重试工具"应该先弹出参数修改对话框
    if (isReasoningDecisionMode) {
      console.log('[ActionRequestComponent] Reasoning decision mode: Clicked retry tool, opening parameter edit dialog', {
        isReasoningDecisionMode,
        tool_name: actionData.tool_name,
      })
      setIsEditing(true)
      return
    }
    
    console.log('[ActionRequestComponent] Non-reasoning decision mode: Clicked retry tool, sending retry response directly')
    
    // 非reasoning决策模式下的重试逻辑（直接发送响应）
    form.validateFields().then((values) => {
      const params = buildParamsFromFormValues(values)
      setEditedArgs(params)
      setIsEditing(false)
      clearAutoCloseTimer()
      onResponse({
        type: 'retry',
        action: 'retry',
        args: params,
      })
    }).catch((errorInfo) => {
      console.error('Retry validation failed:', errorInfo)
      message.error('Please correct the highlighted fields before retrying')
    })
  }

  const handleAbort = () => {
    clearAutoCloseTimer()
    onResponse({ type: 'abort', action: 'abort' })
  }

  const handleReject = () => {
    clearAutoCloseTimer()
    if (isRetryMode) {
      onResponse({ type: 'skip', action: 'skip' })
    } else {
      onResponse({
        type: 'response',
        args: 'User rejected this tool invocation',
      })
    }
  }

  const handleContinue = () => {
    clearAutoCloseTimer()
    onResponse({ type: 'continue', action: 'continue' })
  }

  // 判断是否是数组类型
  const isArrayType = (param: any): boolean => {
    // 检查直接类型
    if (param.type === 'array' || param.type === 'Array') {
      return true
    }
    // 检查 anyOf 中是否包含数组类型
    if (param.anyOf && Array.isArray(param.anyOf)) {
      return param.anyOf.some((item: any) => item.type === 'array')
    }
    // 检查 oneOf 中是否包含数组类型
    if (param.oneOf && Array.isArray(param.oneOf)) {
      return param.oneOf.some((item: any) => item.type === 'array')
    }
    return false
  }

  // 判断是否是对象类型
  const isObjectType = (param: any): boolean => {
    return param.type === 'object' || param.type === 'Object'
  }

  // 判断数组参数是否支持FASTA上传（所有字符型数组都支持）
  const isSequenceArrayParam = (_paramName: string, param: any): boolean => {
    if (!isArrayType(param)) {
      return false
    }
    
    // 检查数组元素的类型 - 只要是字符串数组就支持FASTA上传
    const itemsType = param.items?.type
    const isStringArray = itemsType === 'string' || !itemsType // 默认认为是string
    
    return isStringArray
  }

  // 处理文件下载
  const handleDownload = useCallback(async (filePath: string) => {
    if (!filePath || !filePath.trim()) {
      message.warning('Please enter a file path')
      return
    }
    
    try {
      const response = await fetch(`/api/sessions/files/download?path=${encodeURIComponent(filePath)}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `HTTP ${response.status}`)
      }

      // 获取文件名
      const contentDisposition = response.headers.get('Content-Disposition')
      let fileName = 'download'
      if (contentDisposition) {
        const fileNameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
        if (fileNameMatch && fileNameMatch[1]) {
          fileName = fileNameMatch[1].replace(/['"]/g, '')
        }
      }

      // 下载文件
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = fileName
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      message.success('File downloaded successfully')
    } catch (error) {
      console.error('Download failed:', error)
      message.error(`Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }, [message])

  const renderParameterForm = () => {
    const currentArgs = actionData.tool_args || {}
    
    // 使用解析后的 schema
    const resolvedSchema = getResolvedParameterSchema
    if (resolvedSchema && resolvedSchema.properties) {
      const properties = resolvedSchema.properties
      const required = resolvedSchema.required || []
      
      const { serviceId, toolName } = getServiceIdAndToolName
      const hasDemoData = serviceId && toolName && hasToolDemoData(serviceId, toolName)

      return (
        <Form form={form} layout="vertical">
          {hasDemoData && (
            <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                type="default"
                onClick={handleUseDemoForForm}
                style={{ marginBottom: 16 }}
              >
                Use Demo
              </Button>
            </div>
          )}
          {Object.keys(properties).map((paramName) => {
            const param = properties[paramName]
            const isRequired = required.includes(paramName)
            const isOutputPath = isOutputPathParam(paramName, param)
            const isDemoParam = demoParams.has(paramName)
            
            // 从配置中提取信息
            const title = param.title || paramName
            const description = param.description || ''
            const helpText = param.help_text || param['help_text']
            const placeholder = param.placeholder || param['placeholder']
            const uiType = param.ui_type || param['ui_type']
            const options = param.options || param['options']
            const supportUpload = param.support_upload || param['support_upload'] || false
            const supportFileTypes = param.support_file_types || param['support_file_types'] || []
            const minValue = param.min
            const maxValue = param.max
            const defaultValue = param.default

            // 构建label
            const labelContent = (
              <div>
                <div>
                  <Text strong>{title}</Text>
                  {isRequired && <Tag color="red" style={{ marginLeft: 8 }}>Required</Tag>}
                  {!isRequired && <Tag color="default" style={{ marginLeft: 8 }}>Optional</Tag>}
                </div>
                {description && (
                  <div style={{ marginTop: 6 }}>
                    <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.6 }}>
                      {description}
                    </Text>
                  </div>
                )}
                {helpText && (
                  <div style={{ marginTop: 4 }}>
                    <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.5, fontStyle: 'italic', opacity: 0.8 }}>
                      💡 {helpText}
                    </Text>
                  </div>
                )}
              </div>
            )

            // 根据ui_type渲染输入组件
            let inputComponent: React.ReactNode = null

            // 获取当前表单值（用于判断是否需要显示下载按钮）
            const currentValue = form.getFieldValue(paramName)
            const displayValue = typeof currentValue === 'string' ? currentValue : String(currentValue || '')
            
            // 检查是否是文件路径
            const isFilePath = (value: string): boolean => {
              if (typeof value !== 'string') return false
              const trimmed = value.trim()
              if (trimmed.startsWith('{') || trimmed.startsWith('[')) return false
              return (
                value.startsWith('/opt/antibody_gen/artifacts/') ||
                value.startsWith('/data_new/workspace/') ||
                /^[A-Za-z]:[\\/]/.test(value)
              )
            }
            
            const shouldShowDownload = (isDemoParam || (supportUpload && currentValue)) && 
              typeof displayValue === 'string' && isFilePath(displayValue)

            // 如果是demo参数或supportUpload且有文件路径值，显示下载按钮
            if (shouldShowDownload) {
              inputComponent = (
                <Space.Compact style={{ width: '100%', display: 'flex' }}>
                  <Input
                    value={displayValue}
                    style={{ flex: 1 }}
                  />
                  <Button
                    type="primary"
                    icon={<ArrowDownOutlined />}
                    onClick={() => handleDownload(displayValue)}
                    title="Download file"
                  >
                    Download
                  </Button>
                </Space.Compact>
              )
            }
            // 如果还没有设置inputComponent，根据uiType渲染
            if (!inputComponent) {
              // select类型
              if (uiType === 'select' && options && Array.isArray(options)) {
                inputComponent = (
                  <Select
                    placeholder={placeholder || `Select ${paramName}`}
                    options={options.map((opt: string) => ({ label: opt, value: opt }))}
                  />
                )
              }
              // multiselect类型
              else if (uiType === 'multiselect' && options && Array.isArray(options)) {
                inputComponent = (
                  <Select
                    mode="multiple"
                    placeholder={placeholder || `Select ${paramName}`}
                    options={options.map((opt: string) => ({ label: opt, value: opt }))}
                  />
                )
              }
              // radio类型
              else if (uiType === 'radio' && options && Array.isArray(options)) {
                inputComponent = (
                  <Radio.Group>
                    <Space>
                      {options.map((opt: string) => (
                        <Radio key={opt} value={opt}>
                          {opt}
                        </Radio>
                      ))}
                    </Space>
                  </Radio.Group>
                )
              }
              // number类型
              else if (uiType === 'number' || param.type === 'integer') {
                inputComponent = (
                  <InputNumber
                    placeholder={placeholder || (defaultValue ? `Default: ${defaultValue}` : `Enter ${paramName}`)}
                    min={minValue}
                    max={maxValue}
                    style={{ width: '100%' }}
                  />
                )
              }
              // json类型
              else if (uiType === 'json') {
                inputComponent = (
                  <TextArea
                    rows={6}
                    placeholder={placeholder || (defaultValue ? `Default: ${JSON.stringify(defaultValue)}` : `Enter JSON, e.g., {"key": "value"}`)}
                    style={{ fontFamily: 'monospace' }}
                  />
                )
              }
              // array_input类型
              else if (uiType === 'array_input') {
                inputComponent = (
                  <TextArea
                    rows={6}
                    placeholder={placeholder || (defaultValue ? `Default: ${JSON.stringify(defaultValue)}` : `Enter JSON array, e.g., ["item1", "item2"]`)}
                    style={{ fontFamily: 'monospace' }}
                  />
                )
              }
              // array类型（非array_input）
              else if (isArrayType(param) && uiType !== 'array_input') {
                const isSequenceArray = isSequenceArrayParam(paramName, param)
                inputComponent = (
                  <ArrayInput
                    placeholder={placeholder || (defaultValue ? `Default: ${JSON.stringify(defaultValue)}` : 'Click "Add Item" to add array elements')}
                    itemPlaceholder={`Enter ${paramName} value`}
                    sessionId={sessionId}
                    supportFastaUpload={isSequenceArray}
                  />
                )
              }
              // object类型
              else if (isObjectType(param)) {
                inputComponent = (
                  <TextArea
                    rows={6}
                    placeholder={placeholder || (defaultValue ? `Default: ${JSON.stringify(defaultValue)}` : `Enter JSON object, e.g., {"key": "value"}`)}
                    style={{ fontFamily: 'monospace' }}
                  />
                )
              }
              // text类型或默认类型
              else {
                inputComponent = (
                  <Input
                    placeholder={placeholder || (isOutputPath ? getOutputPathDefault() : (defaultValue ? `Default: ${defaultValue}` : `Enter ${paramName}`))}
                  />
                )
              }
            }

            // 构建操作按钮（Upload）
            const actionButtons: React.ReactNode[] = []
            
            if (supportUpload && !isOutputPath && sessionId) {
              actionButtons.push(
                <FileUpload
                  key={`upload-${paramName}`}
                  sessionId={sessionId}
                  buttonProps={{ type: 'default', size: 'small' }}
                  buttonText="Upload"
                  accept={supportFileTypes.length > 0 ? supportFileTypes.map((ext: string) => `.${ext}`).join(',') : undefined}
                  onUploadSuccess={(artifact) => handleFileUploadSuccess(paramName, artifact)}
                  showUrl={false}
                />
              )
            }
            
            // 构建验证规则
            const rules: any[] = []
            
            if (isRequired) {
              rules.push({
                validator: (_: any, value: any) => {
                  if (value === undefined || value === null || value === '') {
                    return Promise.reject(new Error(`Please provide ${paramName}`))
                  }
                  if (typeof value === 'string' && value.trim() === '') {
                    return Promise.reject(new Error(`Please provide ${paramName}`))
                  }
                  return Promise.resolve()
                }
              })
            }

            // 对象类型JSON验证
            if (isObjectType(param)) {
              rules.push({
                validator: (_: any, value: any) => {
                  if (!value && !isRequired) return Promise.resolve()
                  if (!value && isRequired) return Promise.reject(new Error(`Please provide ${paramName}`))
                  const isFilePath = typeof value === 'string' && (
                    value.startsWith('/opt/antibody_gen/artifacts/') ||
                    value.startsWith('/data_new/workspace/') ||
                    /^[A-Za-z]:[\\/]/.test(value)
                  )
                  if (isFilePath) return Promise.resolve()
                  try {
                    const parsed = typeof value === 'string' ? JSON.parse(value) : value
                    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
                      return Promise.reject(new Error('Please enter a valid JSON object'))
                    }
                    return Promise.resolve()
                  } catch (e) {
                    return Promise.reject(new Error('Invalid JSON format'))
                  }
                }
              })
            }

            // array_input类型JSON数组验证（同时支持JSON数组和HTTP地址）
            if (uiType === 'array_input') {
              rules.push({
                validator: (_: any, value: any) => {
                  if (!value && !isRequired) return Promise.resolve()
                  if (!value && isRequired) return Promise.reject(new Error(`Please provide ${paramName}`))
                  
                  // 检查是否是文件路径（本地路径）
                  const isFilePath = typeof value === 'string' && (
                    value.startsWith('/opt/antibody_gen/artifacts/') ||
                    value.startsWith('/data_new/workspace/') ||
                    /^[A-Za-z]:[\\/]/.test(value)
                  )
                  if (isFilePath) return Promise.resolve()
                  
                  // 检查是否是HTTP/HTTPS URL
                  const isHttpUrl = typeof value === 'string' && (
                    value.startsWith('http://') ||
                    value.startsWith('https://')
                  )
                  if (isHttpUrl) return Promise.resolve()
                  
                  // 检查是否是有效的JSON数组
                  try {
                    const parsed = typeof value === 'string' ? JSON.parse(value) : value
                    if (!Array.isArray(parsed)) {
                      return Promise.reject(new Error('Please enter a valid JSON array, e.g., ["item1", "item2"], or an HTTP/HTTPS URL'))
                    }
                    return Promise.resolve()
                  } catch (e) {
                    return Promise.reject(new Error('Invalid format. Please enter a valid JSON array (e.g., ["item1", "item2"]), an HTTP/HTTPS URL, or a file path'))
                  }
                }
              })
            }

            // 数组类型（非array_input）验证
            if (isArrayType(param) && uiType !== 'array_input') {
              rules.push({
                validator: (_: any, value: any) => {
                  if (!value && !isRequired) return Promise.resolve()
                  if (!value && isRequired) return Promise.reject(new Error(`Please provide at least one item for ${paramName}`))
                  if (!Array.isArray(value)) {
                    return Promise.reject(new Error('Please provide a valid array'))
                  }
                  return Promise.resolve()
                }
              })
            }

            // 数字类型min/max验证
            if (uiType === 'number' || param.type === 'integer') {
              rules.push({
                validator: (_: any, value: any) => {
                  if (value === null || value === undefined) {
                    if (isRequired) return Promise.reject(new Error(`Please provide ${paramName}`))
                    return Promise.resolve()
                  }
                  const numValue = typeof value === 'number' ? value : Number(value)
                  if (isNaN(numValue)) {
                    return Promise.reject(new Error(`Please enter a valid number for ${paramName}`))
                  }
                  if (minValue !== undefined && numValue < minValue) {
                    return Promise.reject(new Error(`${paramName} must be at least ${minValue}`))
                  }
                  if (maxValue !== undefined && numValue > maxValue) {
                    return Promise.reject(new Error(`${paramName} must be at most ${maxValue}`))
                  }
                  return Promise.resolve()
                }
              })
            }

            // JSON类型验证
            if (uiType === 'json') {
              rules.push({
                validator: (_: any, value: any) => {
                  if (!value && !isRequired) return Promise.resolve()
                  if (!value && isRequired) return Promise.reject(new Error(`Please provide ${paramName}`))
                  try {
                    const parsed = typeof value === 'string' ? JSON.parse(value) : value
                    if (typeof parsed !== 'object') {
                      return Promise.reject(new Error('Please enter a valid JSON (object or array)'))
                    }
                    return Promise.resolve()
                  } catch (e) {
                    return Promise.reject(new Error('Invalid JSON format'))
                  }
                }
              })
            }

            // multiselect验证
            if (uiType === 'multiselect') {
              rules.push({
                validator: (_: any, value: any) => {
                  if (!value && !isRequired) return Promise.resolve()
                  if (!value && isRequired) return Promise.reject(new Error(`Please select at least one option for ${paramName}`))
                  if (!Array.isArray(value)) {
                    return Promise.reject(new Error(`Please select options for ${paramName}`))
                  }
                  if (value.length === 0 && isRequired) {
                    return Promise.reject(new Error(`Please select at least one option for ${paramName}`))
                  }
                  return Promise.resolve()
                }
              })
            }

            return (
              <Form.Item
                key={paramName}
                label={labelContent}
                name={paramName}
                rules={rules}
                extra={actionButtons.length > 0 ? (
                  <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {actionButtons}
                  </div>
                ) : undefined}
              >
                {inputComponent}
              </Form.Item>
            )
          })}
        </Form>
      )
    }
    
    // 如果没有 schema，基于当前参数动态生成表单
    const argKeys = Object.keys(currentArgs)
    if (argKeys.length === 0) {
      return (
        <div>
          <Text type="secondary">No parameter definition</Text>
        </div>
      )
    }
    
    return (
      <Form
        form={form}
        layout="vertical"
        initialValues={editedArgs}
      >
        {argKeys.map((key) => {
          const value = currentArgs[key]
          const valueType = typeof value
          const isString = valueType === 'string'
          
          return (
            <Form.Item
              key={key}
              label={key}
              name={key}
              tooltip={`Current value type: ${valueType}`}
            >
              <Input
                addonAfter={
                  // 对于字符串类型参数，显示上传按钮（如果不是输出路径）
                  isString && sessionId && !key.toLowerCase().includes('output') ? (
                    <FileUpload
                      key={`upload-${key}`}
                      sessionId={sessionId}
                      buttonProps={{ type: 'default', size: 'small' }}
                      buttonText="Upload"
                      onUploadSuccess={(artifact) => {
                        handleFileUploadSuccess(key, artifact)
                      }}
                      showUrl={false}
                    />
                  ) : null
                }
              />
            </Form.Item>
          )
        })}
      </Form>
    )
  }

  return (
    <Card
      title={
        isReasoningDecisionMode ? (
          <Space size="small">
            <Text strong style={{ fontSize: '15px' }}>Result Evaluation</Text>
            <Text type="secondary" style={{ fontSize: '13px' }}>{actionData.tool_name}</Text>
          </Space>
        ) : (
          <Space>
            <Text strong>
              {isRetryMode ? 'Tool Retry Request' : 'Tool Invocation Request'}
            </Text>
            <Text type="secondary">({actionData.tool_name})</Text>
          </Space>
        )
      }
      style={{ 
        margin: '16px 0',
        position: 'relative',
        zIndex: isReasoningDecisionMode ? 1000 : 1001,
        backgroundColor: '#fff',
        boxShadow: isReasoningDecisionMode ? '0 2px 8px rgba(0,0,0,0.1)' : 'none',
      }}
      size={isReasoningDecisionMode ? 'small' : 'default'}
      actions={[
        isReasoningDecisionMode ? (
          <Space key="reasoning-actions" size="small" style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button
              key="retry"
              type="primary"
              icon={<CheckOutlined />}
              onClick={handleRetryConfirm}
              size="small"
            >
              Retry Tool
            </Button>
            <Button
              key="continue"
              type="default"
              icon={<ArrowDownOutlined />}
              onClick={handleContinue}
              size="small"
            >
              Continue Process
            </Button>
            <Button
              key="abort"
              danger
              icon={<CloseOutlined />}
              onClick={handleAbort}
              size="small"
            >
              Abort Task
            </Button>
          </Space>
        ) : isRetryMode ? (
          <Button
            key="retry"
            type="primary"
            icon={<CheckOutlined />}
            onClick={handleRetryConfirm}
          >
            Retry Tool
          </Button>
        ) : (
          <Button
            key="accept"
            type="primary"
            icon={<CheckOutlined />}
            onClick={handleAccept}
          >
            Execute Tool
          </Button>
        ),
        // 在reasoning决策模式下，不显示"Edit Parameters"按钮，因为用户应该通过"重试工具"按钮来修改参数
        !isReasoningDecisionMode && (
          <Button
            key="edit"
            icon={<EditOutlined />}
            onClick={handleEdit}
          >
            Edit Parameters
          </Button>
        ),
        !isReasoningDecisionMode && (isRetryMode ? (
          <Button
            key="skip"
            danger
            icon={<CloseOutlined />}
            onClick={handleReject}
          >
            Skip Tool
          </Button>
        ) : (
          <Button
            key="reject"
            danger
            icon={<CloseOutlined />}
            onClick={handleReject}
          >
            Reject Execution
          </Button>
        )),
        isRetryMode && !isReasoningDecisionMode && (
          <Button key="abort" onClick={handleAbort} type="default">
            Abort Task
          </Button>
        ),
      ]}
    >
      {isReasoningDecisionMode && actionData.reasoning && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ 
            padding: '12px 16px', 
            background: actionData.reasoning.status === 'invalid' ? '#fff2f0' : actionData.reasoning.status === 'uncertain' ? '#fffbe6' : '#f6ffed',
            border: `1px solid ${actionData.reasoning.status === 'invalid' ? '#ffccc7' : actionData.reasoning.status === 'uncertain' ? '#ffe58f' : '#b7eb8f'}`,
            borderRadius: '6px',
            marginBottom: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
              <Tag 
                color={actionData.reasoning.status === 'invalid' ? 'red' : actionData.reasoning.status === 'uncertain' ? 'orange' : 'green'}
                style={{ margin: 0, fontSize: '12px', padding: '2px 8px' }}
              >
                {actionData.reasoning.status === 'invalid' ? 'Failed' : actionData.reasoning.status === 'uncertain' ? 'Uncertain' : 'Passed'}
              </Tag>
              {actionData.reasoning.confidence !== undefined && (
                <Text type="secondary" style={{ fontSize: '12px', marginLeft: 8 }}>
                  Confidence: {(actionData.reasoning.confidence * 100).toFixed(0)}%
                </Text>
              )}
            </div>
            <Text style={{ fontSize: '13px', lineHeight: '1.6', color: '#595959', display: 'block' }}>
              {actionData.reasoning.rationale}
            </Text>
            {actionData.reasoning.recommended_actions && actionData.reasoning.recommended_actions.length > 0 && (
              <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(0,0,0,0.06)' }}>
                <Text strong style={{ fontSize: '12px', color: '#8c8c8c', display: 'block', marginBottom: 6 }}>Recommendations:</Text>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {actionData.reasoning.recommended_actions.map((action: string, index: number) => (
                    <Tag key={index} style={{ fontSize: '12px', margin: 0 }}>
                      {action}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {isRetryMode && !isReasoningDecisionMode && (
        <Alert
          type="error"
          showIcon
          message="Tool execution failed"
          description={actionData.error || 'Please adjust the parameters and retry this tool.'}
          style={{ marginBottom: 16 }}
        />
      )}
      {isReasoningDecisionMode ? (
        <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
          <div style={{ marginBottom: 8 }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>Tool: </Text>
            <Text code style={{ fontSize: '12px' }}>{actionData.tool_name}</Text>
          </div>
          <div style={{ 
            background: '#fafafa', 
            padding: '8px 12px', 
            borderRadius: '4px',
            border: '1px solid #f0f0f0',
          }}>
            <Text type="secondary" style={{ fontSize: '11px', display: 'block', marginBottom: 4 }}>Parameters:</Text>
            <pre style={{ 
              fontSize: '11px', 
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'inherit',
              color: '#595959',
            }}>
              {JSON.stringify(actionData.tool_args, null, 2)}
            </pre>
          </div>
        </div>
      ) : (
        <Descriptions column={1} size="small">
          <Descriptions.Item label="Tool">
            <Text code>{actionData.tool_name}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Description">
            <Text>{actionData.tool_info.description}</Text>
          </Descriptions.Item>
          {isRetryMode && actionData.task && (
            <Descriptions.Item label="Task">
              <Text>{actionData.task}</Text>
            </Descriptions.Item>
          )}
          {isRetryMode && actionData.error && (
            <Descriptions.Item label="Last Error">
              <Text type="danger">{actionData.error}</Text>
            </Descriptions.Item>
          )}
          <Descriptions.Item label="Current Parameters">
            <pre style={{ fontSize: '12px', background: '#f5f5f5', padding: '8px', borderRadius: '4px' }}>
              {JSON.stringify(actionData.tool_args, null, 2)}
            </pre>
          </Descriptions.Item>
        </Descriptions>
      )}

      <Modal
        title={isReasoningDecisionMode ? "Edit Tool Parameters (Retry Tool)" : "Edit Tool Parameters"}
        open={isEditing}
        onOk={handleEditSubmit}
        onCancel={() => {
          setIsEditing(false)
        }}
        width={600}
        okText={isReasoningDecisionMode ? "Confirm Retry" : "Apply"}
        cancelText="Cancel"
        zIndex={2000}
        mask={true}
        maskClosable={false}
        styles={{
          body: {
            maxHeight: '50vh',
            overflowY: 'auto',
            paddingRight: '8px'
          }
        }}
      >
        {renderParameterForm()}
      </Modal>
    </Card>
  )
}
