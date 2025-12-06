import React, { useState, useEffect } from 'react'
import { Input, Button, Space, Upload, message } from 'antd'
import { PlusOutlined, DeleteOutlined, EditOutlined, CheckOutlined, CloseOutlined, UploadOutlined } from '@ant-design/icons'
import { colors, spacing, borderRadius } from '../../styles/tokens'
import { uploadArtifact } from '../../services/artifacts-service'

interface ArrayInputProps {
  value?: any[]
  onChange?: (value: any[]) => void
  placeholder?: string
  itemPlaceholder?: string
  disabled?: boolean
  sessionId?: string | null
  supportFastaUpload?: boolean
}

export const ArrayInput: React.FC<ArrayInputProps> = ({
  value = [],
  onChange,
  placeholder = 'Click "Add Item" to add array elements',
  itemPlaceholder = 'Enter value',
  disabled = false,
  sessionId = null,
  supportFastaUpload = false,
}) => {
  const [items, setItems] = useState<string[]>([])
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingValue, setEditingValue] = useState<string>('')
  const [uploading, setUploading] = useState(false)

  // 同步外部value到内部items
  useEffect(() => {
    if (value && Array.isArray(value)) {
      const stringItems = value.map(item => item === null || item === undefined ? '' : String(item))
      // 只有当items不同时才更新，避免循环更新
      if (JSON.stringify(stringItems) !== JSON.stringify(items)) {
        setItems(stringItems)
      }
    } else if (value === undefined || value === null || value === '') {
      if (items.length > 0) {
        setItems([])
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  // 添加新项
  const handleAdd = () => {
    const newItems = [...items, '']
    setItems(newItems)
    setEditingIndex(newItems.length - 1)
    setEditingValue('')
    // 添加新项时，保留所有项（包括空项），让用户有机会输入
    onChange?.(newItems)
  }

  // 开始编辑
  const handleStartEdit = (index: number) => {
    setEditingIndex(index)
    setEditingValue(items[index])
  }

  // 保存编辑
  const handleSaveEdit = (index: number) => {
    const newItems = [...items]
    newItems[index] = editingValue
    setItems(newItems)
    setEditingIndex(null)
    setEditingValue('')
    // 保存时，过滤掉完全空白的项
    const filteredItems = newItems.filter(item => item.trim() !== '')
    onChange?.(filteredItems.length > 0 ? filteredItems : [])
  }

  // 取消编辑
  const handleCancelEdit = () => {
    const editingIndexValue = editingIndex
    setEditingIndex(null)
    setEditingValue('')
    
    // 如果取消的是空项，删除它
    if (editingIndexValue !== null && editingIndexValue < items.length) {
      const currentItem = items[editingIndexValue]
      if (!currentItem || currentItem.trim() === '') {
        const newItems = items.filter((_, i) => i !== editingIndexValue)
        setItems(newItems)
        const filteredItems = newItems.filter(item => item.trim() !== '')
        onChange?.(filteredItems.length > 0 ? filteredItems : [])
      }
    }
  }

  // 删除项
  const handleDelete = (index: number) => {
    const newItems = items.filter((_, i) => i !== index)
    setItems(newItems)
    setEditingIndex(null)
    // 删除后，过滤掉完全空白的项
    const filteredItems = newItems.filter(item => item.trim() !== '')
    onChange?.(filteredItems.length > 0 ? filteredItems : [])
  }

  // 更新编辑中的值
  const handleEditChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditingValue(e.target.value)
  }

  // 处理输入框回车
  const handleEditKeyPress = (e: React.KeyboardEvent<HTMLInputElement>, index: number) => {
    if (e.key === 'Enter') {
      handleSaveEdit(index)
    } else if (e.key === 'Escape') {
      handleCancelEdit()
    }
  }

  // 处理文件上传（直接回填文件路径，不解析文件内容）
  const handleFileUpload = async (file: File) => {
    if (!sessionId) {
      message.error('Session ID is required for file upload')
      return false
    }

    setUploading(true)
    try {
      // 1. 上传文件
      const artifact = await uploadArtifact(sessionId, file, 'File for array input')
      
      // 2. 优先使用OSS直接下载URL，如果没有则使用后端API下载URL
      // 如果都没有，则回退到本地路径格式（兼容旧数据）
      let fileUrl: string
      
      if (artifact.oss_direct_url) {
        // 使用OSS直接URL（公共URL或预签名URL）
        fileUrl = artifact.oss_direct_url
      } else if (artifact.download_url) {
        // 使用后端API下载URL（完整URL）
        fileUrl = `${window.location.origin}${artifact.download_url}`
      } else {
        // 回退到本地路径格式（兼容旧数据或OSS未启用的情况）
        const artifactSessionId = artifact.session_id
        let fileName = artifact.file_name || artifact.original_file_name
        
        if (!artifactSessionId || !fileName) {
          console.error('Missing session_id or file_name in artifact:', artifact)
          message.error('Upload succeeded but missing required information')
          return false
        }
        
        // 处理OSS object_key格式：artifacts/{session_id}/{file_name}
        // 或本地路径格式：C:/opt/antibody_gen/artifacts/{session_id}/file.csv
        if (fileName.startsWith('artifacts/')) {
          const pathParts = fileName.split('/')
          fileName = pathParts[pathParts.length - 1]
        } else if (fileName.includes('/') || fileName.includes('\\')) {
          const pathParts = fileName.split(/[/\\]/)
          fileName = pathParts[pathParts.length - 1]
        }
        
        fileUrl = `/opt/antibody_gen/artifacts/${artifactSessionId}/${fileName}`
      }
      
      // 3. 直接将文件URL添加到数组中
      const newItems = [...items, fileUrl]
      setItems(newItems)
      onChange?.(newItems.filter(item => item.trim() !== ''))
      
      const displayFileName = artifact.original_file_name || artifact.file_name || 'file'
      message.success(`File uploaded successfully: ${displayFileName}`)
      return false // 阻止默认上传行为
    } catch (error) {
      console.error('Failed to upload file:', error)
      message.error('Failed to upload file')
      return false
    } finally {
      setUploading(false)
    }
  }

  return (
    <div>
      <div style={{ 
        border: `1px solid ${colors.border.primary}`,
        borderRadius: borderRadius.md,
        padding: spacing[2],
        minHeight: '80px',
        maxHeight: '300px',
        overflowY: 'auto',
        backgroundColor: disabled ? colors.background.tertiary : colors.background.primary
      }}>
        {items.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            color: colors.text.tertiary,
            padding: spacing[4],
            fontSize: 13
          }}>
            {placeholder}
          </div>
        ) : (
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            {items.map((item, index) => (
              <div
                key={index}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing[2],
                  padding: spacing[2],
                  backgroundColor: colors.background.secondary,
                  borderRadius: borderRadius.base,
                  border: `1px solid ${colors.border.primary}`,
                }}
              >
                <div style={{ 
                  minWidth: '24px', 
                  textAlign: 'center',
                  color: colors.text.secondary,
                  fontSize: 12,
                  fontWeight: 500
                }}>
                  {index + 1}.
                </div>
                
                {editingIndex === index ? (
                  <Input
                    value={editingValue}
                    onChange={handleEditChange}
                    onKeyDown={(e) => handleEditKeyPress(e, index)}
                    onBlur={() => handleSaveEdit(index)}
                    placeholder={itemPlaceholder}
                    autoFocus
                    style={{ flex: 1 }}
                    size="small"
                  />
                ) : (
                  <div
                    style={{
                      flex: 1,
                      padding: `${spacing[1]} ${spacing[2]}`,
                      backgroundColor: '#ffffff',
                      borderRadius: borderRadius.base,
                      border: `1px solid ${colors.border.primary}`,
                      minHeight: '24px',
                      display: 'flex',
                      alignItems: 'center',
                      cursor: disabled ? 'not-allowed' : 'pointer',
                      fontSize: 13,
                      color: colors.text.primary,
                    }}
                    onClick={() => !disabled && handleStartEdit(index)}
                  >
                    {item || <span style={{ color: colors.text.tertiary }}>{itemPlaceholder}</span>}
                  </div>
                )}
                
                <Space size="small">
                  {editingIndex === index ? (
                    <>
                      <Button
                        type="primary"
                        size="small"
                        icon={<CheckOutlined />}
                        onClick={() => handleSaveEdit(index)}
                        disabled={disabled}
                      />
                      <Button
                        size="small"
                        icon={<CloseOutlined />}
                        onClick={handleCancelEdit}
                        disabled={disabled}
                      />
                    </>
                  ) : (
                    <>
                      <Button
                        type="text"
                        size="small"
                        icon={<EditOutlined />}
                        onClick={() => handleStartEdit(index)}
                        disabled={disabled}
                        title="Edit"
                      />
                      <Button
                        type="text"
                        size="small"
                        icon={<DeleteOutlined />}
                        onClick={() => handleDelete(index)}
                        disabled={disabled}
                        danger
                        title="Delete"
                      />
                    </>
                  )}
                </Space>
              </div>
            ))}
          </Space>
        )}
      </div>
      
      <Space.Compact style={{ width: '100%', marginTop: spacing[2] }}>
        <Button
          type="dashed"
          onClick={handleAdd}
          disabled={disabled}
          icon={<PlusOutlined />}
          style={{
            flex: 1,
          }}
        >
          Add Item
        </Button>
        {supportFastaUpload && sessionId && (
          <Upload
            accept="*"
            showUploadList={false}
            beforeUpload={(file) => {
              handleFileUpload(file)
              return false // 阻止默认上传行为
            }}
            disabled={disabled || uploading}
          >
            <Button
              type="dashed"
              disabled={disabled || uploading}
              loading={uploading}
              icon={<UploadOutlined />}
              title="Upload file and add file path to array"
            >
              Upload File
            </Button>
          </Upload>
        )}
      </Space.Compact>
    </div>
  )
}

