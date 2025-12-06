import React, { useState } from 'react'
import { Button, Upload, message, Input, Typography } from 'antd'
import { UploadOutlined, LinkOutlined, CopyOutlined } from '@ant-design/icons'
import { uploadArtifact, type UploadArtifactResponse } from '../../services/artifacts-service'
import { UploadProgressFloatButton, type UploadProgressInfo } from './upload-progress-float-button'

const { Text } = Typography

export interface FileUploadProps {
  /** Session ID for uploading files */
  sessionId: string
  /** Optional description for the uploaded file */
  description?: string
  /** Callback when upload succeeds */
  onUploadSuccess?: (artifact: UploadArtifactResponse) => void
  /** Callback when upload fails */
  onUploadError?: (error: Error) => void
  /** Button text, default is "Upload" */
  buttonText?: string
  /** Button style props */
  buttonProps?: {
    type?: 'default' | 'primary' | 'dashed' | 'link' | 'text'
    size?: 'small' | 'middle' | 'large'
    disabled?: boolean
  }
  /** Whether to show the download URL after upload */
  showUrl?: boolean
  /** Custom display component for the URL */
  urlDisplay?: (url: string, artifact: UploadArtifactResponse) => React.ReactNode
  /** Accepted file types (e.g., ".fasta,.fa" or "image/*") */
  accept?: string
}

export const FileUpload: React.FC<FileUploadProps> = ({
  sessionId,
  description,
  onUploadSuccess,
  onUploadError,
  buttonText = 'Upload',
  buttonProps = { type: 'primary', size: 'middle' },
  showUrl = true,
  urlDisplay,
  accept,
}) => {
  const [uploading, setUploading] = useState(false)
  const [uploadedArtifact, setUploadedArtifact] = useState<UploadArtifactResponse | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string>('')
  const [uploadProgress, setUploadProgress] = useState<UploadProgressInfo | null>(null)

  const handleUpload = async (file: File) => {
    if (!sessionId) {
      message.error('Session ID is required')
      return false
    }

    try {
      setUploading(true)
      
      // 初始化上传进度信息
      const totalChunks = file.size >= 100 * 1024 * 1024 ? Math.ceil(file.size / (10 * 1024 * 1024)) : undefined
      setUploadProgress({
        fileName: file.name,
        fileSize: file.size,
        progress: 0,
        status: 'uploading',
        uploadedBytes: 0,
        totalChunks,
        currentChunk: 0,
      })

      const artifact = await uploadArtifact(
        sessionId,
        file,
        description,
        (progress: number, info?: { currentChunk?: number; totalChunks?: number; uploadedBytes?: number }) => {
          // 使用函数式更新，避免依赖prev导致的不必要刷新
          setUploadProgress(prev => {
            if (!prev) return null
            // 只在关键值变化时更新，避免不必要的重新渲染
            const newProgress = Math.round(progress)
            const newUploadedBytes = info?.uploadedBytes ?? (progress / 100) * file.size
            
            // 如果进度和已上传字节数没有实际变化，不更新
            if (prev.progress === newProgress && prev.uploadedBytes === newUploadedBytes) {
              return prev
            }
            
            return {
              ...prev,
              progress: newProgress,
              uploadedBytes: newUploadedBytes,
              currentChunk: info?.currentChunk ?? prev.currentChunk,
              totalChunks: info?.totalChunks ?? prev.totalChunks,
            }
          })
        }
      )
      
      // 优先使用OSS直接URL，否则使用后端API URL
      const fullUrl = artifact.oss_direct_url 
        ? artifact.oss_direct_url 
        : `${window.location.origin}${artifact.download_url}`
      
      setUploadedArtifact(artifact)
      setDownloadUrl(fullUrl)
      
      // 更新进度为成功状态
      setUploadProgress(prev => prev ? {
        ...prev,
        progress: 100,
        status: 'success',
        uploadedBytes: file.size,
      } : null)
      
      message.success(`File "${artifact.original_file_name}" uploaded successfully`)
      
      if (onUploadSuccess) {
        onUploadSuccess(artifact)
      }
      
      return false // Prevent default upload behavior
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Upload failed')
      
      // 更新进度为错误状态
      setUploadProgress(prev => prev ? {
        ...prev,
        status: 'error',
        errorMessage: err.message,
      } : null)
      
      message.error(`Upload failed: ${err.message}`)
      
      if (onUploadError) {
        onUploadError(err)
      }
      
      return false
    } finally {
      setUploading(false)
    }
  }

  const handleCloseProgress = () => {
    setUploadProgress(null)
  }

  const handleCopyUrl = () => {
    if (downloadUrl) {
      navigator.clipboard.writeText(downloadUrl)
        .then(() => {
          message.success('URL copied to clipboard')
        })
        .catch(() => {
          message.error('Failed to copy URL')
        })
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const uploadProps = {
    beforeUpload: (file: File) => {
      handleUpload(file)
      return false // Prevent default upload, we handle it manually
    },
    showUploadList: false,
    multiple: false,
    accept: accept || '*/*', // Use provided accept or default to all file types
  }

  // If file is uploaded and showUrl is true, show the URL
  if (uploadedArtifact && showUrl) {
    if (urlDisplay) {
      return <>{urlDisplay(downloadUrl, uploadedArtifact)}</>
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <Input.Group compact style={{ display: 'flex', flex: 1, minWidth: '200px' }}>
            <Input
              value={downloadUrl}
              readOnly
              prefix={<LinkOutlined />}
              style={{ flex: 1 }}
              placeholder="Download URL"
            />
            <Button
              icon={<CopyOutlined />}
              onClick={handleCopyUrl}
              title="Copy URL"
            >
              Copy
            </Button>
          </Input.Group>
          <Button
            type="default"
            onClick={() => {
              setUploadedArtifact(null)
              setDownloadUrl('')
            }}
          >
            Upload Another
          </Button>
        </div>
        <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
          <Text type="secondary">
            File: {uploadedArtifact.original_file_name} ({formatFileSize(uploadedArtifact.file_size)})
          </Text>
        </div>
      </div>
    )
  }

  // Default upload button
  return (
    <>
      <Upload {...uploadProps}>
        <Button
          icon={<UploadOutlined />}
          loading={uploading}
          disabled={uploading || !sessionId}
          {...buttonProps}
        >
          {buttonText}
        </Button>
      </Upload>
      
      {/* 上传进度浮动按钮 */}
      {uploadProgress && (
        <UploadProgressFloatButton
          uploadInfo={uploadProgress}
          onClose={handleCloseProgress}
        />
      )}
    </>
  )
}

export default FileUpload

