import React, { useEffect, useState, useRef, useMemo } from 'react'
import { Button, Card, Col, Drawer, Input, Row, Space, Typography, App, Select, Tag, Form, Spin, Radio, InputNumber } from 'antd'
import { AppstoreOutlined, RocketOutlined, ApiOutlined, ThunderboltOutlined, ArrowDownOutlined, InfoCircleOutlined, ExperimentOutlined } from '@ant-design/icons'
import { listServices, listTools, type ServiceInfo, type ToolInfo } from '../services/tools-service'
import { FileUpload } from '../components/common/file-upload-component'
import { ArrayInput } from '../components/common/array-input-component'
import { colors, shadows, spacing, borderRadius } from '../styles/tokens'
import { parseToolDescription } from '../utils/tool-description-parser'
import { getToolDemoData, hasToolDemoData } from '../config/tool-demo-data'

const { TextArea } = Input
const { Title, Text } = Typography

// 隐藏滚动条但保持滚动功能的样式
const scrollbarHiddenStyles = `
  .tools-page-scroll::-webkit-scrollbar {
    display: none;
  }
  .tools-page-scroll {
    -ms-overflow-style: none;
    scrollbar-width: none;
  }
`


const ToolsPage: React.FC = () => {
  const { message } = App.useApp()
  const [services, setServices] = useState<ServiceInfo[]>([])
  const [serviceId, setServiceId] = useState<string>()
  const [loading, setLoading] = useState(false)
  const [loadingTools, setLoadingTools] = useState(false)
  const [invokeOpen, setInvokeOpen] = useState(false)
  const [currentService, setCurrentService] = useState<string | null>(null)
  const [tools, setTools] = useState<ToolInfo[]>([])
  const [selectedTool, setSelectedTool] = useState<string>("")
  const [selectedToolSchema, setSelectedToolSchema] = useState<any>(null)
  const [selectedToolInfo, setSelectedToolInfo] = useState<ToolInfo | null>(null)
  const [result, setResult] = useState<any>(null)
  const [showScrollHint, setShowScrollHint] = useState(false)
  const [canScrollDown, setCanScrollDown] = useState(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [tempSessionId, setTempSessionId] = useState<string | null>(null)
  const [form] = Form.useForm()
  const [downloadPath, setDownloadPath] = useState<string>('')
  const [downloading, setDownloading] = useState(false)
  // 记录哪些参数是disabled的（通过upload或use demo设置后）
  const [disabledParams, setDisabledParams] = useState<Set<string>>(new Set())
  // 记录哪些参数是通过use demo设置的（需要显示download按钮）
  const [demoParams, setDemoParams] = useState<Set<string>>(new Set())

  useEffect(() => {
    listServices().then(setServices).catch(() => message.error('Failed to fetch services'))
  }, [message])

  // 检查是否可以滚动并显示提示
  useEffect(() => {
    const checkScrollability = () => {
      if (scrollContainerRef.current) {
        const { scrollHeight, clientHeight, scrollTop } = scrollContainerRef.current
        const canScroll = scrollHeight > clientHeight
        const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10 // 10px 容差
        
        setShowScrollHint(canScroll && !isAtBottom)
        setCanScrollDown(!isAtBottom && canScroll)
      }
    }

    checkScrollability()
    window.addEventListener('resize', checkScrollability)
    
    if (scrollContainerRef.current) {
      scrollContainerRef.current.addEventListener('scroll', checkScrollability)
    }
    
    return () => {
      window.removeEventListener('resize', checkScrollability)
      if (scrollContainerRef.current) {
        scrollContainerRef.current.removeEventListener('scroll', checkScrollability)
      }
    }
  }, [services])

  // 每次打开抽屉时自动创建新的临时session
  useEffect(() => {
    if (invokeOpen) {
      // 每次打开抽屉时创建新的session
      const createSession = async () => {
        try {
          const resp = await fetch('/api/sessions/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            },
            body: JSON.stringify({ usecase: 'research' }),
          })

          // 检查响应状态
          if (!resp.ok) {
            // 尝试读取错误信息
            let errorText = ''
            try {
              const errorData = await resp.json()
              errorText = errorData.detail || errorData.message || `HTTP ${resp.status}`
    } catch {
              // 如果不是JSON，尝试读取文本
              errorText = await resp.text().catch(() => `HTTP ${resp.status} Error`)
            }
            throw new Error(errorText || `Failed to create session: ${resp.status}`)
          }

          const data = await resp.json()
          const sessionId = data.id || null
          if (!sessionId) {
            throw new Error('Session ID not returned from server')
          }
          setTempSessionId(sessionId)
        } catch (error) {
          console.error('Failed to create temp session:', error)
          const errorMessage = error instanceof Error ? error.message : 'Failed to create temporary session'
          message.error(`Failed to create temporary session: ${errorMessage}`)
          setTempSessionId(null)
        }
      }
      createSession()
    }
  }, [invokeOpen, message])

  // 当 tempSessionId 或工具 schema 变化时，更新输出路径参数的值
  useEffect(() => {
    if (selectedToolSchema && selectedToolSchema.properties && tempSessionId) {
      const properties = selectedToolSchema.properties
      const updates: Record<string, string> = {}
      
      // 判断是否是输出路径参数的函数
      const isOutputPathParam = (param: any): boolean => {
        return param.placeholder === '/path/to/output'
      }
      
      Object.keys(properties).forEach((paramName) => {
        const param = properties[paramName]
        if (isOutputPathParam(param)) {
          updates[paramName] = `/opt/antibody_gen/artifacts/${tempSessionId}`
        }
      })
      
      if (Object.keys(updates).length > 0) {
        form.setFieldsValue(updates)
      }
    }
  }, [tempSessionId, selectedToolSchema, form])

  const loadToolsByService = async (svcId: string) => {
    setLoadingTools(true)
    try {
      const toolsList = await listTools(svcId)
      setTools(toolsList)
      if (toolsList.length > 0) {
        const firstTool = toolsList[0]
        setSelectedTool(firstTool.name)
        setSelectedToolSchema(firstTool.args_schema)
        const tool = toolsList.find(t => t.name === firstTool.name)
        setSelectedToolInfo(tool || null)
        // 初始化表单参数
        initializeFormParams(firstTool.args_schema)
      }
    } catch (error) {
      console.error('Failed to load tools:', error)
      message.error('Failed to load tools')
      setTools([])
      setSelectedTool("")
      setSelectedToolSchema(null)
      setSelectedToolInfo(null)
    } finally {
      setLoadingTools(false)
    }
  }

  // 判断是否是输出路径参数
  const isOutputPathParam = (param: any): boolean => {
    return param.placeholder === '/path/to/output'
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

  // 判断是否是对象类型
  const isObjectType = (param: any): boolean => {
    return param.type === 'object' || param.type === 'Object'
  }

  // 获取输出路径默认值
  const getOutputPathDefault = (): string => {
    if (!tempSessionId) {
      return '/opt/antibody_gen/artifacts/{session_id}'
    }
    return `/opt/antibody_gen/artifacts/${tempSessionId}`
  }

  // 初始化表单参数
  const initializeFormParams = (schema: any) => {
    if (!schema) {
      form.setFieldsValue({})
      return
    }

    // 解析 schema，处理 $ref 引用
    const resolved = resolveSchemaRefs(schema)
    
    // 检查是否需要展开嵌套结构
    const properties = resolved.properties || {}
    const propertyKeys = Object.keys(properties)
    
    let actualProperties = properties

    // 如果只有一个参数且它引用了 $defs，则展开它
    if (propertyKeys.length === 1) {
      const firstKey = propertyKeys[0]
      const firstParam = properties[firstKey]

      if (firstParam.$ref) {
        const refPath = firstParam.$ref
        if (refPath.startsWith('#/$defs/')) {
          const defName = refPath.replace('#/$defs/', '')
          const defs = schema.$defs || {}
          const defSchema = defs[defName]

          if (defSchema && defSchema.properties) {
            actualProperties = defSchema.properties
          }
        }
      }
    }

    if (!actualProperties || Object.keys(actualProperties).length === 0) {
      form.setFieldsValue({})
      return
    }
    
    const initialParams: Record<string, any> = {}
    Object.keys(actualProperties).forEach((key) => {
      const prop = actualProperties[key]
      // 如果是输出路径参数，使用默认值
      if (isOutputPathParam(prop)) {
        initialParams[key] = getOutputPathDefault()
      } else if (prop.default !== undefined) {
        // 根据类型设置默认值
        if (prop.type === 'array') {
          initialParams[key] = Array.isArray(prop.default) ? prop.default : []
        } else {
          initialParams[key] = prop.default
        }
      } else if (prop.type === 'array') {
        initialParams[key] = []
      } else {
        initialParams[key] = ''
      }
    })
    form.setFieldsValue(initialParams)
  }


  const onInvoke = async () => {
    if (!currentService) {
      message.warning('Please select a service')
      return
    }
    
    // 先验证表单
    try {
      await form.validateFields()
    } catch (error) {
      message.error('Please fill in all required fields')
      return
    }
    
    // 从表单组装JSON参数
    const formValues = form.getFieldsValue()
    
    // 检查原始 schema 是否有嵌套结构（如 args 参数）
    const resolvedSchema = getResolvedParameterSchema
    const originalProperties = selectedToolSchema?.properties || {}
    const originalPropertyKeys = Object.keys(originalProperties)
    
    // 判断是否需要嵌套结构
    let needsNestedStructure = false
    let nestedKey = ''
    if (originalPropertyKeys.length === 1) {
      const firstKey = originalPropertyKeys[0]
      const firstParam = originalProperties[firstKey]
      if (firstParam.$ref) {
        needsNestedStructure = true
        nestedKey = firstKey
      }
    }
    
    const params: Record<string, any> = {}
    const actualParams: Record<string, any> = {}
    
    // 遍历表单值，转换为正确的类型
    Object.keys(formValues).forEach((key) => {
      const value = formValues[key]
      if (value === undefined || value === null || value === '') {
        // 跳过空值
        return
      }
      
      // 获取参数定义（使用解析后的 schema）
      const param = resolvedSchema?.properties?.[key]
      const isArray = param && (param.type === 'array' || param.type === 'Array')
      const isObject = param && (param.type === 'object' || param.type === 'Object')
      
      // 如果是数组类型，直接使用（ArrayInput组件已经返回数组）
      if (isArray) {
        // ArrayInput组件返回的已经是数组格式
        if (Array.isArray(value)) {
          actualParams[key] = value
        } else if (value && typeof value === 'string') {
          // 如果是字符串，尝试解析JSON
          try {
            const parsed = JSON.parse(value)
            actualParams[key] = Array.isArray(parsed) ? parsed : [parsed]
          } catch {
            // 解析失败，转换为数组
            actualParams[key] = [value]
          }
        } else {
          actualParams[key] = value || []
        }
      } else if (isObject) {
        // 对象类型，尝试解析JSON
        try {
          // 如果已经是对象，直接使用
          if (typeof value === 'object' && !Array.isArray(value)) {
            actualParams[key] = value
          } else {
            // 否则尝试解析JSON字符串
            const parsed = JSON.parse(value)
            actualParams[key] = parsed
          }
        } catch (e) {
          // JSON解析失败，使用原始值
          console.warn(`Failed to parse JSON for ${key}:`, e)
          actualParams[key] = value
        }
      } else {
        // 普通类型，尝试解析JSON（可能是字符串形式的JSON）
        try {
          const parsed = JSON.parse(value)
          actualParams[key] = parsed
        } catch {
          // 不是JSON，直接使用字符串值
          actualParams[key] = value
        }
      }
    })
    
    // 如果需要嵌套结构，将参数包装在嵌套键中
    if (needsNestedStructure && nestedKey) {
      params[nestedKey] = actualParams
    } else {
      Object.assign(params, actualParams)
    }
    
    setLoading(true)
    try {
      const ret = await fetch('/api/tools/invoke', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` },
        body: JSON.stringify({ service_id: currentService, tool_name: selectedTool || currentService, params }),
      }).then(r => r.json())
      setResult(ret?.result)
      message.success('Invoked successfully')
    } catch (e) {
      message.error('Invocation failed')
    } finally {
      setLoading(false)
    }
  }

  // 清除所有工具相关的状态
  const clearToolState = () => {
    setSelectedTool("")
    setSelectedToolSchema(null)
    setSelectedToolInfo(null)
    setResult(null)
    setDownloadPath('')
    setTempSessionId(null)
    setLoadingTools(false)
    setDisabledParams(new Set())
    setDemoParams(new Set())
    form.resetFields()
  }

  // 解析工具描述
  const parsedDescription = useMemo(() => {
    if (!selectedToolInfo?.description) return null
    return parseToolDescription(selectedToolInfo.description)
  }, [selectedToolInfo?.description])

  // 解析 schema，处理 $ref 引用
  const resolveSchemaRefs = (schema: any): any => {
    if (!schema || typeof schema !== 'object') {
      return schema
    }

    // 如果 schema 有 $ref，需要解析引用
    if (schema.$ref) {
      const refPath = schema.$ref
      // 处理 "#/$defs/XXX" 格式的引用
      if (refPath.startsWith('#/$defs/')) {
        const defName = refPath.replace('#/$defs/', '')
        const defs = selectedToolSchema?.$defs || {}
        const resolved = defs[defName]
        if (resolved) {
          // 递归解析 resolved 中的引用
          return resolveSchemaRefs(resolved)
        }
      }
      return schema
    }

    // 递归处理对象和数组
    if (Array.isArray(schema)) {
      return schema.map(item => resolveSchemaRefs(item))
    }

    const resolved: any = {}
    for (const key in schema) {
      if (key === '$defs') {
        // 保留 $defs，但不在这里处理
        resolved[key] = schema[key]
        continue
      }
      resolved[key] = resolveSchemaRefs(schema[key])
    }

    return resolved
  }

  // 获取展开后的参数 schema（处理 $ref 引用）
  const getResolvedParameterSchema = useMemo(() => {
    if (!selectedToolSchema) return null

    // 先解析整个 schema
    const resolved = resolveSchemaRefs(selectedToolSchema)

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
          const defs = selectedToolSchema.$defs || {}
          const defSchema = defs[defName]

          if (defSchema && defSchema.properties) {
            // 返回展开后的参数定义
            return {
              properties: defSchema.properties,
              required: defSchema.required || [],
              // 保留原始 schema 的其他信息
              title: resolved.title,
              type: resolved.type,
            }
          }
        }
      }
      
      // 情况2: 如果这个参数已经被解析为一个对象，且它有 properties（说明它是从 $defs 解析出来的）
      // 例如：firstParam = { properties: {...}, required: [...], type: "object" }
      if (firstParam && typeof firstParam === 'object' && !Array.isArray(firstParam) && firstParam.properties) {
        // 直接使用这个对象的 properties
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
  }, [selectedToolSchema])

  // 处理文件上传成功
  const handleFileUploadSuccess = React.useCallback((paramName: string, artifact: any) => {
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
      message.error('上传成功但未获取到下载链接，请重试')
      return
    }
    
    // 设置表单字段值
    form.setFieldValue(paramName, fileUrl)
    
    // 设置该参数为disabled
    setDisabledParams(prev => new Set(prev).add(paramName))
    
    // 移除demo标记（如果是upload，不是demo）
    setDemoParams(prev => {
      const newSet = new Set(prev)
      newSet.delete(paramName)
      return newSet
    })
    
    const fileName = artifact.original_file_name || artifact.file_name || 'file'
    message.success(`File uploaded successfully: ${fileName}`)
  }, [form, message])

  // 处理使用Demo（表单级别，填充整个表单）
  const handleUseDemoForForm = React.useCallback(() => {
    if (!currentService || !selectedTool) {
      message.warning('Please select a service and tool first')
      return
    }

    const demoData = getToolDemoData(currentService, selectedTool)
    if (!demoData) {
      message.warning(`No demo data available for ${currentService}/${selectedTool}`)
      return
    }

    // 获取解析后的 schema
    const resolvedSchema = getResolvedParameterSchema
    if (!resolvedSchema || !resolvedSchema.properties) {
      message.error('Failed to load tool schema')
      return
    }

    const properties = resolvedSchema.properties
    const formValues: Record<string, any> = {}
    const newDisabledParams = new Set<string>()
    const newDemoParams = new Set<string>()

    // 获取输出路径默认值（内联函数，避免依赖问题）
    const getOutputPathDefaultValue = (): string => {
      if (!tempSessionId) {
        return '/opt/antibody_gen/artifacts/{session_id}'
      }
      return `/opt/antibody_gen/artifacts/${tempSessionId}`
    }

    // 遍历所有参数，填充 demo 数据
    Object.keys(properties).forEach((paramName) => {
      const param = properties[paramName]
      const isOutputPath = isOutputPathParam(param)

      // 如果是输出路径参数，使用默认值
      if (isOutputPath) {
        formValues[paramName] = getOutputPathDefaultValue()
        newDisabledParams.add(paramName)
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
            newDisabledParams.add(paramName)
          } else if (typeof demoValue === 'string' && demoValue.trim() !== '') {
            // 非空字符串也标记为 disabled（用户可以通过清空来编辑）
            newDisabledParams.add(paramName)
          } else if (Array.isArray(demoValue) && demoValue.length > 0) {
            // 非空数组也标记为 disabled
            newDisabledParams.add(paramName)
          } else if (typeof demoValue === 'object' && Object.keys(demoValue).length > 0) {
            // 非空对象也标记为 disabled
            newDisabledParams.add(paramName)
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
    setDisabledParams(newDisabledParams)
    setDemoParams(newDemoParams)

    message.success('Demo data loaded successfully')
  }, [currentService, selectedTool, form, message, getResolvedParameterSchema, tempSessionId])

  // 处理文件下载
  const handleDownload = React.useCallback(async (filePath: string) => {
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

  // 渲染参数表单
  const renderParameterFields = useMemo(() => {
    // 使用解析后的 schema
    const resolvedSchema = getResolvedParameterSchema
    if (!resolvedSchema || !resolvedSchema.properties || Object.keys(resolvedSchema.properties).length === 0) {
      return (
        <div>
          <Text type="secondary">No parameters defined for this tool</Text>
        </div>
      )
    }

    const properties = resolvedSchema.properties
    const required = resolvedSchema.required || []
    console.log(properties, required)

    return (
      <Form form={form} layout="vertical">
        {Object.keys(properties).map((paramName) => {
          const param = properties[paramName]
          const isRequired = required.includes(paramName)
          const isOutputPath = isOutputPathParam(param)
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
          // 对于 supportUpload 为 true 的参数，需要根据是否有值动态显示
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
          // select类型
          else if (uiType === 'select' && options && Array.isArray(options)) {
            inputComponent = (
              <Select
                placeholder={placeholder || `Select ${paramName}`}
                options={options.map((opt: string) => ({ label: opt, value: opt }))}
                defaultValue={defaultValue}
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
                defaultValue={defaultValue ? (Array.isArray(defaultValue) ? defaultValue : [defaultValue]) : undefined}
              />
            )
          }
          // radio类型
          else if (uiType === 'radio' && options && Array.isArray(options)) {
            inputComponent = (
              <Radio.Group defaultValue={defaultValue}>
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
                defaultValue={defaultValue}
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
                sessionId={tempSessionId}
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

          // 构建操作按钮（Upload）
          // 只要 supportUpload 为 true，就应该显示 Upload 按钮（无论是否有值）
          // 如果有值，会同时显示 Download 按钮（在 inputComponent 中）
          const actionButtons: React.ReactNode[] = []
          
          if (supportUpload && !isOutputPath && tempSessionId) {
            // 对于 supportUpload 为 true 的参数，始终显示 Upload 按钮
            actionButtons.push(
              <FileUpload
                key={`upload-${paramName}`}
                sessionId={tempSessionId}
                buttonProps={{ type: 'default', size: 'small' }}
                buttonText="Upload"
                accept={supportFileTypes.length > 0 ? supportFileTypes.map((ext: string) => `.${ext}`).join(',') : undefined}
                onUploadSuccess={(artifact) => handleFileUploadSuccess(paramName, artifact)}
                showUrl={false}
              />
            )
          }

          // 移除单个参数的 Use Demo 按钮（已上移到表单级别）

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
                  value.startsWith('/data_new/workspace/antibody_gen/') ||
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
                  value.startsWith('/data_new/workspace/antibody_gen/') ||
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
                <div style={{ marginTop: spacing[2], display: 'flex', gap: spacing[2], flexWrap: 'wrap' }}>
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
  }, [form, disabledParams, demoParams, selectedToolSchema, tempSessionId, handleDownload, handleFileUploadSuccess, isArrayType, isSequenceArrayParam, isObjectType, isOutputPathParam])


  // Generate color gradient based on service ID
  function getServiceColor(sid: string): { bg: string; border: string; icon: string } {
    const colors = [
      { bg: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', border: '#667eea', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', border: '#f5576c', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: '#4facfe', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', border: '#43e97b', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)', border: '#fa709a', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)', border: '#30cfd0', icon: '#ffffff' },
      { bg: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)', border: '#a8edea', icon: '#2c3e50' },
      { bg: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)', border: '#fcb69f', icon: '#2c3e50' },
    ]
    const hash = sid.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
    return colors[hash % colors.length]
  }

  return (
    <>
      <style>{scrollbarHiddenStyles}</style>
      <div style={{ 
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        maxWidth: '100%', 
        overflowX: 'hidden',
        overflowY: 'hidden',
        background: 'linear-gradient(to bottom, #f8fafc 0%, #ffffff 100%)',
      }}>
        {/* 固定的标题栏 */}
        <div style={{ 
          flexShrink: 0,
          padding: `${spacing[6]} ${spacing[6]} 0 ${spacing[6]}`,
          paddingBottom: 0,
          borderBottom: `2px solid ${colors.border.primary}`,
          background: 'linear-gradient(to bottom, #f8fafc 0%, #ffffff 100%)',
          zIndex: 10,
          position: 'sticky',
          top: 0
        }}>
          <Space size="large" style={{ width: '100%' }}>
            <div style={{
              width: 56,
              height: 56,
              borderRadius: borderRadius.lg,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: shadows.lg
            }}>
              <AppstoreOutlined style={{ fontSize: 28, color: '#ffffff' }} />
            </div>
            <div>
              <Title level={2} style={{ margin: 0, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Services Catalog
              </Title>
              <Text type="secondary" style={{ fontSize: 14 }}>Discover and invoke available MCP services</Text>
            </div>
          </Space>
        </div>

        {/* 可滚动的卡片区域 */}
        <div 
          ref={scrollContainerRef}
          className="tools-page-scroll"
          style={{
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            padding: `${spacing[6]} ${spacing[6]} ${spacing[6]} ${spacing[6]}`,
            position: 'relative'
          }}
        >
          <Row gutter={[16, 16]}>
        {services.map((s) => {
          const sid = (s as any)?.id ?? (s as any)
          const transport = (s as any)?.transport
          const about = (s as any)?.about
          const host = (s as any)?.host
          const initial = (sid ?? '?').toString().slice(0, 1).toUpperCase()
          const serviceColor = getServiceColor(sid)
          const isSelected = serviceId === sid
          
          return (
            <Col key={sid} xs={24} sm={12} md={8} lg={6} xl={6}>
              <Card
                hoverable
                onClick={async () => { 
                  // 清除之前的状态，确保打开新抽屉时是干净的状态
                  clearToolState()
                  
                  setServiceId(sid)
                  setCurrentService(sid)
                  setInvokeOpen(true)
                  await loadToolsByService(sid as string) 
                }}
                style={{ 
                  borderColor: isSelected ? serviceColor.border : colors.border.primary,
                  borderWidth: isSelected ? 2 : 1,
                  cursor: 'pointer',
                  borderRadius: borderRadius.lg,
                  boxShadow: isSelected ? shadows.lg : shadows.base,
                  transition: 'all 0.3s ease',
                  background: '#ffffff',
                  overflow: 'hidden',
                  position: 'relative'
                }}
                bodyStyle={{ padding: spacing[4] }}
              >
                {/* Decorative gradient bar */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 4,
                  background: serviceColor.bg,
                  opacity: isSelected ? 1 : 0.7
                }} />
                
                <div style={{ display: 'flex', gap: spacing[3], alignItems: 'flex-start', marginTop: spacing[2] }}>
                  <div style={{
                    width: 52,
                    height: 52,
                    borderRadius: borderRadius.md,
                    background: serviceColor.bg,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 800,
                    fontSize: 20,
                    flexShrink: 0,
                    color: serviceColor.icon,
                    boxShadow: shadows.md,
                    transition: 'transform 0.2s ease'
                  }}>
                    {initial}
                  </div>
                  <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8, marginBottom: spacing[2] }}>
                      <span style={{ 
                        fontWeight: 700, 
                        fontSize: 15,
                        wordBreak: 'break-word', 
                        overflowWrap: 'break-word',
                        color: colors.text.primary,
                        lineHeight: 1.3
                      }} title={sid}>{sid}</span>
                      <Tag 
                        color={isSelected ? 'blue' : 'default'}
                        style={{ 
                          margin: 0,
                          fontSize: 11,
                          padding: '2px 8px',
                          borderRadius: borderRadius.base,
                          flexShrink: 0
                        }}
                      >
                        {(transport || 'MCP').toUpperCase()}
                      </Tag>
                    </div>
                    <div style={{ 
                      marginTop: spacing[1], 
                      wordBreak: 'break-word', 
                      overflowWrap: 'break-word', 
                      lineHeight: 1.5,
                      minHeight: 40
                    }}>
                      <Text style={{ fontSize: 12, color: colors.text.secondary }}>{about || 'MCP Service'}</Text>
                    </div>
                    {host && (
                      <div style={{ marginTop: spacing[2], wordBreak: 'break-all', overflowWrap: 'break-word' }}>
                        <Tag 
                          style={{ 
                            fontSize: 10, 
                            color: colors.text.tertiary,
                            background: colors.background.secondary,
                            border: 'none',
                            margin: 0,
                            padding: '2px 6px'
                          }}
                        >
                          <ApiOutlined style={{ marginRight: 4 }} />
                          {host.length > 20 ? `${host.substring(0, 20)}...` : host}
                        </Tag>
                      </div>
                    )}
                  </div>
                </div>
                <div style={{ 
                  marginTop: spacing[3],
                  paddingTop: spacing[2],
                  borderTop: `1px solid ${colors.border.primary}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing[2]
                }}>
                  <RocketOutlined style={{ fontSize: 14, color: serviceColor.border }} />
                  <span style={{ fontSize: 12, color: colors.text.tertiary }}>Click to invoke</span>
                </div>
              </Card>
            </Col>
          )
        })}
          </Row>

          {/* 滚动提示 */}
          {showScrollHint && canScrollDown && (
            <div
              style={{
                position: 'fixed',
                bottom: spacing[6],
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(0, 0, 0, 0.7)',
                color: '#ffffff',
                padding: `${spacing[2]} ${spacing[4]}`,
                borderRadius: borderRadius.lg,
                display: 'flex',
                alignItems: 'center',
                gap: spacing[2],
                fontSize: 12,
                zIndex: 100,
                animation: 'fadeInOut 3s ease-in-out infinite',
                pointerEvents: 'none',
                boxShadow: shadows.lg
              }}
            >
              <ArrowDownOutlined />
              <span>Scroll down for more services</span>
            </div>
          )}
        </div>

        <style>{`
          @keyframes fadeInOut {
            0% { opacity: 0; transform: translateX(-50%) translateY(10px); }
            20% { opacity: 1; transform: translateX(-50%) translateY(0); }
            80% { opacity: 1; transform: translateX(-50%) translateY(0); }
            100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
          }
        `}</style>
        </div>

      <Drawer
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: spacing[3] }}>
            <ThunderboltOutlined style={{ fontSize: 20, color: colors.primary[500] }} />
            <span>{currentService ? `Invoke: ${currentService}` : 'Invoke Tool'}</span>
          </div>
        }
        width={600}
        open={invokeOpen}
        onClose={() => {
          // 清除所有缓存的状态
          clearToolState()
          setInvokeOpen(false)
        }}
        destroyOnClose
        styles={{ 
          body: { overflowX: 'hidden', padding: spacing[4] },
          header: { 
            borderBottom: `2px solid ${colors.border.primary}`,
            padding: spacing[4]
          }
        }}
      >
        <Spin spinning={loadingTools} tip="正在加载工具配置信息...">
        <Space direction="vertical" style={{ width: '100%', maxWidth: '100%' }} size="large">
          <div style={{ width: '100%', maxWidth: '100%' }}>
            {currentService && (
              <div style={{ marginBottom: spacing[4], width: '100%' }}>
                <Text strong style={{ fontSize: 14, display: 'block', marginBottom: spacing[2] }}>
                  <ApiOutlined style={{ marginRight: spacing[2], color: colors.primary[500] }} />
                  Tool Selection
                </Text>
                <Select
                  style={{ width: '100%' }}
                  placeholder="Select a tool"
                  value={selectedTool || undefined}
                    onChange={(v) => {
                      // 清除原有内容（但保留 tempSessionId，因为它是为整个抽屉会话创建的）
                      setResult(null)
                      setDownloadPath('')
                      // 注意：不清除 tempSessionId，保持它在整个抽屉打开期间有效
                      setDisabledParams(new Set())
                      setDemoParams(new Set())
                      form.resetFields()
                      
                      // 设置新工具
                      setSelectedTool(v)
                      const tool = tools.find(t => t.name === v)
                      setSelectedToolInfo(tool || null)
                      setSelectedToolSchema(tool?.args_schema || null)
                      
                      // 重新初始化表单参数
                      if (tool?.args_schema) {
                        initializeFormParams(tool.args_schema)
                      }
                    }}
                  options={tools.map(t => ({ label: t.name, value: t.name }))}
                  size="large"
                    loading={loadingTools}
                    disabled={loadingTools}
                />
              </div>
            )}
            
            {/* Tool Description and Parameter Info */}
            {selectedToolInfo && (
              <div style={{ 
                marginBottom: spacing[4],
                padding: spacing[3],
                background: colors.background.secondary,
                borderRadius: borderRadius.md,
                border: `1px solid ${colors.border.primary}`
              }}>
                {parsedDescription && (
                  <>
                    {/* 工具摘要 */}
                    {parsedDescription.summary && (
                      <div style={{ marginBottom: spacing[3] }}>
                        <Text strong style={{ fontSize: 13, display: 'block', marginBottom: spacing[1] }}>
                          <InfoCircleOutlined style={{ marginRight: spacing[1], color: colors.primary[500] }} />
                          Tool Description
                        </Text>
                        <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                          {parsedDescription.summary}
                        </Text>
                      </div>
                    )}
                    
                    {/* 返回值说明 */}
                    {parsedDescription.returns && (
                      <div style={{ marginBottom: spacing[3], paddingTop: spacing[2], borderTop: `1px solid ${colors.border.secondary}` }}>
                        <Text strong style={{ fontSize: 13, display: 'block', marginBottom: spacing[1] }}>
                          <RocketOutlined style={{ marginRight: spacing[1], color: colors.primary[500] }} />
                          Returns
                        </Text>
                        <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.6 }}>
                          {parsedDescription.returns}
                        </Text>
                      </div>
                    )}
                  </>
                )}
                
                {/* 如果解析失败，显示原始描述 */}
                {!parsedDescription && selectedToolInfo.description && (
                  <div style={{ marginBottom: spacing[2] }}>
                    <Text strong style={{ fontSize: 13, display: 'block', marginBottom: spacing[1] }}>
                      <InfoCircleOutlined style={{ marginRight: spacing[1], color: colors.primary[500] }} />
                      Tool Description
                    </Text>
                    <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                      {selectedToolInfo.description}
                    </Text>
                  </div>
                )}
              </div>
            )}
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing[2] }}>
                <Text strong style={{ fontSize: 14 }}>
                  Parameters
              </Text>
                {currentService && selectedTool && hasToolDemoData(currentService, selectedTool) && (
                  <Button
                    type="dashed"
                    icon={<ExperimentOutlined />}
                    onClick={handleUseDemoForForm}
                    size="small"
                style={{ 
                      borderRadius: borderRadius.md,
                    }}
                  >
                    Use Demo
                  </Button>
                )}
              </div>
              {renderParameterFields}
            </div>
            <div style={{ marginTop: spacing[4], display: 'flex', gap: spacing[2] }}>
              <Button 
                type="primary" 
                onClick={onInvoke} 
                loading={loading}
                size="large"
                icon={<RocketOutlined />}
                style={{
                  borderRadius: borderRadius.md,
                  boxShadow: shadows.md,
                  height: 44
                }}
              >
                {loading ? 'Invoking...' : 'Invoke Tool'}
              </Button>
              <Button 
                onClick={() => {
                  // 清除所有缓存的状态
                  clearToolState()
                  setInvokeOpen(false)
                }}
                size="large"
              >
                Cancel
              </Button>
            </div>
          </div>
          {!!result && (
            <div style={{ 
              width: '100%', 
              maxWidth: '100%',
              marginTop: spacing[4],
              padding: spacing[4],
              background: colors.background.secondary,
              borderRadius: borderRadius.md,
              border: `1px solid ${colors.border.primary}`
            }}>
              <Text strong style={{ fontSize: 14, display: 'block', marginBottom: spacing[2] }}>
                Execution Result
              </Text>
              <TextArea
                rows={8}
                value={typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                style={{ 
                  width: '100%', 
                  maxWidth: '100%', 
                  wordBreak: 'break-word', 
                  overflowWrap: 'break-word',
                  fontFamily: 'monospace',
                  borderRadius: borderRadius.md,
                  background: '#ffffff'
                }}
              />
              {/* File Download Section */}
              <div style={{ 
                marginTop: spacing[4],
                padding: spacing[3],
                background: colors.background.secondary,
                borderRadius: borderRadius.md,
                border: `1px solid ${colors.border.primary}`
              }}>
                <Text strong style={{ fontSize: 14, display: 'block', marginBottom: spacing[2] }}>
                  File Download
                </Text>
                <Input.Group compact style={{ display: 'flex', gap: spacing[2] }}>
                  <Input
                    placeholder="Enter file path (e.g., /opt/antibody_gen/artifacts/{session_id}/file.txt)"
                    value={downloadPath}
                    onChange={(e) => setDownloadPath(e.target.value)}
                    style={{ flex: 1 }}
                    size="large"
                  />
                  <Button
                    type="primary"
                    icon={<ArrowDownOutlined />}
                    onClick={async () => {
                      if (!downloadPath.trim()) {
                        message.warning('Please enter a file path')
                        return
                      }
                      
                      setDownloading(true)
                      try {
                        const response = await fetch(`/api/sessions/files/download?path=${encodeURIComponent(downloadPath)}`, {
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
                        const contentDisposition = response.headers.get('content-disposition')
                        let filename = 'download'
                        if (contentDisposition) {
                          const filenameMatch = contentDisposition.match(/filename="?(.+?)"?$/)
                          if (filenameMatch) {
                            filename = filenameMatch[1]
                          }
                        }

                        // 创建下载链接
                        const blob = await response.blob()
                        const url = window.URL.createObjectURL(blob)
                        const a = document.createElement('a')
                        a.href = url
                        a.download = filename
                        document.body.appendChild(a)
                        a.click()
                        document.body.removeChild(a)
                        window.URL.revokeObjectURL(url)

                        message.success('File downloaded successfully')
                      } catch (error) {
                        console.error('Download error:', error)
                        const errorMessage = error instanceof Error ? error.message : 'Download failed'
                        message.error(`Download failed: ${errorMessage}`)
                      } finally {
                        setDownloading(false)
                      }
                    }}
                    loading={downloading}
                    size="large"
                    style={{
                      borderRadius: borderRadius.md,
                    }}
                  >
                    Download
                  </Button>
                </Input.Group>
              </div>
            </div>
          )}
        </Space>
        </Spin>
      </Drawer>
    </>
  )
}

export default ToolsPage


