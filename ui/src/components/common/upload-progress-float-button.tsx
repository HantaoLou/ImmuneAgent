import React, { useState, useEffect, useRef } from 'react'
import { FloatButton, Progress, Modal, Typography, Space, Tag, Divider } from 'antd'
import { CloudUploadOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'

const { Text } = Typography

export interface UploadProgressInfo {
  fileName: string
  fileSize: number
  progress: number
  status: 'uploading' | 'success' | 'error'
  uploadedBytes?: number
  totalChunks?: number
  currentChunk?: number
  uploadSpeed?: number // bytes per second
  estimatedTimeRemaining?: number // seconds
  errorMessage?: string
}

export interface UploadProgressFloatButtonProps {
  uploadInfo: UploadProgressInfo | null
  onClose?: () => void
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.ceil(seconds)}秒`
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.ceil(seconds % 60)
    return `${mins}分${secs}秒`
  } else {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    return `${hours}小时${mins}分钟`
  }
}

export const UploadProgressFloatButton: React.FC<UploadProgressFloatButtonProps> = ({
  uploadInfo,
  onClose,
}) => {
  const [modalVisible, setModalVisible] = useState(false)
  const [uploadSpeed, setUploadSpeed] = useState<number>(0)
  const lastUpdateTimeRef = useRef<number>(Date.now())
  const lastUploadedBytesRef = useRef<number>(0)

  // 优化：使用ref避免不必要的重新渲染，只在uploadInfo变化时更新速度
  useEffect(() => {
    if (uploadInfo?.status === 'uploading' && uploadInfo.uploadedBytes !== undefined) {
      const now = Date.now()
      const timeDiff = (now - lastUpdateTimeRef.current) / 1000 // seconds
      
      if (timeDiff > 0.5 && lastUploadedBytesRef.current > 0) { // 至少间隔0.5秒才更新速度
        const bytesDiff = uploadInfo.uploadedBytes - lastUploadedBytesRef.current
        const speed = bytesDiff / timeDiff
        setUploadSpeed(speed)
      }
      
      lastUpdateTimeRef.current = now
      lastUploadedBytesRef.current = uploadInfo.uploadedBytes
    } else if (uploadInfo?.status !== 'uploading') {
      setUploadSpeed(0)
    }
  }, [uploadInfo?.status, uploadInfo?.uploadedBytes]) // 只依赖关键字段

  // 上传成功后，3秒后自动隐藏
  useEffect(() => {
    if (uploadInfo?.status === 'success') {
      const timer = setTimeout(() => {
        onClose?.()
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [uploadInfo?.status, onClose])

  if (!uploadInfo) {
    return null
  }

  const getStatusColor = () => {
    switch (uploadInfo.status) {
      case 'uploading':
        return '#1890ff'
      case 'success':
        return '#52c41a'
      case 'error':
        return '#ff4d4f'
      default:
        return '#1890ff'
    }
  }

  const getStatusIcon = () => {
    switch (uploadInfo.status) {
      case 'uploading':
        return <CloudUploadOutlined />
      case 'success':
        return <CheckCircleOutlined />
      case 'error':
        return <CloseCircleOutlined />
      default:
        return <CloudUploadOutlined />
    }
  }

  const getProgressStatus = (): 'normal' | 'exception' | 'success' | 'active' => {
    switch (uploadInfo.status) {
      case 'uploading':
        return 'active'
      case 'success':
        return 'success'
      case 'error':
        return 'exception'
      default:
        return 'normal'
    }
  }

  const handleFloatButtonClick = () => {
    setModalVisible(true)
  }

  const handleModalClose = () => {
    setModalVisible(false)
    // 如果上传已完成（成功或失败），关闭进度显示
    if (uploadInfo.status === 'success' || uploadInfo.status === 'error') {
      onClose?.()
    }
  }

  // 计算预计剩余时间
  const estimatedTimeRemaining = uploadSpeed > 0 && uploadInfo.status === 'uploading'
    ? (uploadInfo.fileSize - (uploadInfo.uploadedBytes ?? 0)) / uploadSpeed
    : undefined

  return (
    <>
      <FloatButton
        icon={getStatusIcon()}
        type="primary"
        style={{
          right: 24,
          bottom: 24,
          width: 72,
          height: 72,
          backgroundColor: getStatusColor(),
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        }}
        onClick={handleFloatButtonClick}
      >
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {/* 环形进度条作为背景辅助 */}
          <Progress
            type="circle"
            percent={uploadInfo.status === 'success' ? 100 : Math.round(uploadInfo.progress)}
            size={64}
            strokeWidth={4}
            status={getProgressStatus()}
            format={() => null} // 不显示默认内容，我们自定义显示
            strokeColor={
              uploadInfo.status === 'success'
                ? '#52c41a'
                : uploadInfo.status === 'error'
                ? '#ff4d4f'
                : {
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }
            }
            style={{
              position: 'absolute',
            }}
          />
          {/* 百分比数字显示在中心 */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1,
            }}
          >
            {uploadInfo.status === 'success' ? (
              <CheckCircleOutlined
                style={{
                  fontSize: '28px',
                  color: '#fff',
                }}
              />
            ) : uploadInfo.status === 'error' ? (
              <CloseCircleOutlined
                style={{
                  fontSize: '28px',
                  color: '#fff',
                }}
              />
            ) : (
              <div
                style={{
                  fontSize: '18px',
                  fontWeight: 'bold',
                  color: '#fff',
                  lineHeight: '1.2',
                  textAlign: 'center',
                  textShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
                }}
              >
                {Math.round(uploadInfo.progress)}%
              </div>
            )}
          </div>
        </div>
      </FloatButton>

      <Modal
        title={
          <Space>
            {getStatusIcon()}
            <span>文件上传详情</span>
            <Tag color={uploadInfo.status === 'uploading' ? 'processing' : uploadInfo.status === 'success' ? 'success' : 'error'}>
              {uploadInfo.status === 'uploading' ? '上传中' : uploadInfo.status === 'success' ? '上传成功' : '上传失败'}
            </Tag>
          </Space>
        }
        open={modalVisible}
        onCancel={handleModalClose}
        onOk={handleModalClose}
        okText={uploadInfo.status === 'uploading' ? '后台继续' : '关闭'}
        cancelText={uploadInfo.status === 'uploading' ? undefined : undefined}
        cancelButtonProps={uploadInfo.status === 'uploading' ? { style: { display: 'none' } } : undefined}
        width={500}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <div>
            <Text strong>文件名：</Text>
            <Text>{uploadInfo.fileName}</Text>
          </div>
          
          <div>
            <Text strong>文件大小：</Text>
            <Text>{formatFileSize(uploadInfo.fileSize)}</Text>
          </div>

          <Divider style={{ margin: '12px 0' }} />

          <div>
            <Text strong>上传进度</Text>
            <Progress
              percent={Math.round(uploadInfo.progress)}
              status={getProgressStatus()}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
              <Text type="secondary">
                已上传: {uploadInfo.uploadedBytes ? formatFileSize(uploadInfo.uploadedBytes) : formatFileSize((uploadInfo.progress / 100) * uploadInfo.fileSize)}
              </Text>
              <Text type="secondary">
                剩余: {formatFileSize(((100 - uploadInfo.progress) / 100) * uploadInfo.fileSize)}
              </Text>
            </div>
          </div>

          {uploadInfo.totalChunks && uploadInfo.currentChunk && (
            <div>
              <Text strong>分片进度：</Text>
              <Text>
                {uploadInfo.currentChunk} / {uploadInfo.totalChunks} 分片
              </Text>
              <Progress
                percent={Math.round((uploadInfo.currentChunk / uploadInfo.totalChunks) * 100)}
                size="small"
                style={{ marginTop: 8 }}
              />
            </div>
          )}

          {uploadSpeed > 0 && (
            <div>
              <Text strong>上传速度：</Text>
              <Text>{formatFileSize(uploadSpeed)}/秒</Text>
            </div>
          )}

          {estimatedTimeRemaining && estimatedTimeRemaining > 0 && (
            <div>
              <Text strong>预计剩余时间：</Text>
              <Text>{formatTime(estimatedTimeRemaining)}</Text>
            </div>
          )}

          {uploadInfo.status === 'error' && uploadInfo.errorMessage && (
            <>
              <Divider style={{ margin: '12px 0' }} />
              <div>
                <Text strong type="danger">错误信息：</Text>
                <div style={{ marginTop: 8, padding: 8, backgroundColor: '#fff2f0', borderRadius: 4 }}>
                  <Text type="danger">{uploadInfo.errorMessage}</Text>
                </div>
              </div>
            </>
          )}
        </Space>
      </Modal>
    </>
  )
}

export default UploadProgressFloatButton

