import React, { useEffect, useState } from 'react'
import { Button, List, Typography, Space, Tag, Tooltip, message } from 'antd'
import { DownloadOutlined, DeleteOutlined } from '@ant-design/icons'
import {
  getSessionArtifacts,
  downloadArtifact,
  deleteArtifact,
  type SessionArtifact,
} from '../../services/artifacts-service'

const { Text, Title } = Typography

interface ArtifactsListProps {
  sessionId: string
}

export const ArtifactsList: React.FC<ArtifactsListProps> = ({ sessionId }) => {
  const [artifacts, setArtifacts] = useState<SessionArtifact[]>([])
  const [loading, setLoading] = useState(false)

  const fetchArtifacts = async () => {
    try {
      setLoading(true)
      const data = await getSessionArtifacts(sessionId)
      setArtifacts(data)
    } catch (error) {
      message.error('Failed to fetch artifacts')
      console.error('Error fetching artifacts:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (sessionId) {
      fetchArtifacts()
    }
  }, [sessionId])

  const handleDownload = async (artifact: SessionArtifact) => {
    try {
      await downloadArtifact(artifact)  // 传入artifact对象以支持OSS直接下载
      message.success(`Downloaded ${artifact.original_file_name}`)
    } catch (error) {
      message.error('Failed to download artifact')
      console.error('Error downloading artifact:', error)
    }
  }

  const handleDelete = async (artifact: SessionArtifact) => {
    try {
      await deleteArtifact(artifact.id)
      message.success(`Deleted ${artifact.original_file_name}`)
      fetchArtifacts() // Refresh the list
    } catch (error) {
      message.error('Failed to delete artifact')
      console.error('Error deleting artifact:', error)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) return '🖼️'
    if (mimeType.startsWith('video/')) return '🎥'
    if (mimeType.startsWith('audio/')) return '🎵'
    if (mimeType.includes('pdf')) return '📄'
    if (
      mimeType.includes('text') ||
      mimeType.includes('json') ||
      mimeType.includes('xml')
    )
      return '📝'
    if (
      mimeType.includes('zip') ||
      mimeType.includes('tar') ||
      mimeType.includes('gz')
    )
      return '📦'
    return '📁'
  }

  if (artifacts.length === 0 && !loading) {
    return (
      <div style={{ padding: '16px', textAlign: 'center' }}>
        <Text type="secondary">No artifacts found for this session</Text>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px' }}>
      <div
        style={{
          marginBottom: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={5} style={{ margin: 0 }}>
          Session Artifacts ({artifacts.length})
        </Title>
        <Button
          type="primary"
          size="small"
          onClick={fetchArtifacts}
          loading={loading}
        >
          Refresh
        </Button>
      </div>

      <List
        loading={loading}
        dataSource={artifacts}
        renderItem={(artifact) => (
          <List.Item
            actions={[
              <Tooltip title="Download">
                <Button
                  type="text"
                  icon={<DownloadOutlined />}
                  onClick={() => handleDownload(artifact)}
                  size="small"
                />
              </Tooltip>,
              <Tooltip title="Delete">
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleDelete(artifact)}
                  size="small"
                />
              </Tooltip>,
            ]}
          >
            <List.Item.Meta
              avatar={
                <span style={{ fontSize: '24px' }}>
                  {getFileIcon(artifact.mime_type)}
                </span>
              }
              title={
                <Space>
                  <Text strong>{artifact.original_file_name}</Text>
                  <Tag color="blue">{formatFileSize(artifact.file_size)}</Tag>
                </Space>
              }
              description={
                <Space direction="vertical" size="small">
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {artifact.description || 'No description'}
                  </Text>
                  <Space size="small">
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      Created: {new Date(artifact.created_at).toLocaleString()}
                    </Text>
                    <Tag color="green">{artifact.mime_type}</Tag>
                  </Space>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </div>
  )
}
