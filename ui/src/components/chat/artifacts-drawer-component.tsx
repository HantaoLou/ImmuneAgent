import React, { useEffect, useState } from 'react'
import { Button, Typography, Tooltip, message, Card } from 'antd'
import { DownloadOutlined, FileOutlined, FileImageOutlined, FilePdfOutlined, FileTextOutlined } from '@ant-design/icons'
import { getSessionArtifacts, downloadArtifact, type SessionArtifact } from '../../services/artifacts-service'

// 简单的错误边界组件
class ErrorBoundary extends React.Component<
    { children: React.ReactNode },
    { hasError: boolean }
> {
    constructor(props: { children: React.ReactNode }) {
        super(props)
        this.state = { hasError: false }
    }

    static getDerivedStateFromError(_error: Error) {
        return { hasError: true }
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error('ArtifactsDrawer Error:', error, errorInfo)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '20px',
                    textAlign: 'center',
                    color: '#ff4d4f',
                    backgroundColor: '#fff2f0',
                    border: '1px solid #ffccc7'
                }}>
                    <div>Something went wrong with the artifacts drawer.</div>
                    <Button
                        type="primary"
                        size="small"
                        onClick={() => this.setState({ hasError: false })}
                        style={{ marginTop: '8px' }}
                    >
                        Try Again
                    </Button>
                </div>
            )
        }

        return this.props.children
    }
}

const { Text, Title } = Typography

interface ArtifactsDrawerProps {
    sessionId: string
    visible: boolean
    onClose: () => void
}

const ArtifactsDrawerContent: React.FC<ArtifactsDrawerProps> = ({
    sessionId,
    visible,
    onClose
}) => {
    const [artifacts, setArtifacts] = useState<SessionArtifact[]>([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const fetchArtifacts = async () => {
        try {
            setLoading(true)
            setError(null)
            const data = await getSessionArtifacts(sessionId)
            setArtifacts(data)
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to fetch artifacts'
            setError(errorMessage)
            message.error('Failed to fetch artifacts')
            console.error('Error fetching artifacts:', error)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        if (visible && sessionId) {
            fetchArtifacts()
        }
    }, [visible, sessionId])

    const handleDownload = async (artifact: SessionArtifact) => {
        try {
            await downloadArtifact(artifact)  // 传入artifact对象以支持OSS直接下载
            message.success(`Downloaded ${artifact.original_file_name}`)
        } catch (error) {
            message.error('Failed to download artifact')
            console.error('Error downloading artifact:', error)
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
        if (mimeType.startsWith('image/')) return <FileImageOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
        if (mimeType.includes('pdf')) return <FilePdfOutlined style={{ fontSize: '24px', color: '#ff4d4f' }} />
        if (mimeType.includes('text') || mimeType.includes('json') || mimeType.includes('xml')) return <FileTextOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
        return <FileOutlined style={{ fontSize: '24px', color: '#666' }} />
    }

    return (
        <>
            {/* 背景遮罩层，点击可关闭抽屉 */}
            {visible && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 999,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                    }}
                    onClick={onClose}
                />
            )}

            {/* 抽屉内容 */}
            <div
                style={{
                    position: 'absolute',
                    top: 0,
                    right: visible ? '0' : '-400px',
                    width: '400px',
                    height: 'calc(100% - 160px)', // 再次削减高度，确保完全不遮挡输入框
                    backgroundColor: '#fff',
                    borderLeft: '1px solid #f0f0f0',
                    boxShadow: visible ? '-2px 0 8px rgba(0, 0, 0, 0.15)' : 'none',
                    transition: 'right 0.3s ease',
                    zIndex: 1000,
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                }}
            >
                {visible && (
                    <>
                        <div
                            style={{
                                padding: '16px',
                                borderBottom: '1px solid #f0f0f0',
                                backgroundColor: '#fafafa', // 与内容区域保持一致
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                flexShrink: 0,
                            }}
                        >
                            <Title level={4} style={{ margin: 0, color: '#262626' }}>Artifacts</Title>
                            <Button
                                type="text"
                                onClick={onClose}
                                style={{
                                    padding: '4px 8px',
                                    color: '#8c8c8c',
                                    fontSize: '16px',
                                    lineHeight: 1,
                                }}
                                title="Close drawer"
                            >
                                ✕
                            </Button>
                        </div>
                        <div style={{
                            padding: '16px',
                            flex: 1,
                            overflow: 'auto',
                            backgroundColor: '#fafafa',
                        }}>
                            <div style={{
                                marginBottom: 16,
                                fontSize: '14px',
                                color: '#8c8c8c',
                            }}>
                                Generated files and documents from this conversation
                            </div>


                            {loading ? (
                                <div style={{
                                    textAlign: 'center',
                                    padding: '40px 20px',
                                    color: '#999'
                                }}>
                                    <div>Loading artifacts...</div>
                                </div>
                            ) : error ? (
                                <div style={{
                                    textAlign: 'center',
                                    padding: '40px 20px',
                                    color: '#ff4d4f'
                                }}>
                                    <div>Error: {error}</div>
                                    <Button
                                        type="primary"
                                        size="small"
                                        onClick={fetchArtifacts}
                                        style={{ marginTop: '16px' }}
                                    >
                                        Retry
                                    </Button>
                                </div>
                            ) : artifacts.length === 0 ? (
                                <div style={{
                                    textAlign: 'center',
                                    padding: '40px 20px',
                                    color: '#999'
                                }}>
                                    <FileOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                                    <div>No artifacts yet</div>
                                    <div style={{ fontSize: '12px', marginTop: '8px' }}>
                                        Files will appear here as they are generated
                                    </div>
                                </div>
                            ) : (
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(2, 1fr)',
                                    gap: '16px',
                                    marginBottom: '16px',
                                    padding: '0 4px' // 添加左右内边距，避免贴边
                                }}>
                                    {artifacts.map((artifact) => (
                                        <Card
                                            key={artifact.id}
                                            size="small"
                                            hoverable
                                            style={{
                                                height: '140px', // 增加高度，提供更多空间
                                                display: 'flex',
                                                flexDirection: 'column',
                                                justifyContent: 'space-between',
                                                borderRadius: '12px', // 增加圆角
                                                border: '1px solid #e8e8e8',
                                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.06)', // 添加轻微阴影
                                                transition: 'all 0.2s ease', // 添加过渡动画
                                            }}
                                            bodyStyle={{
                                                padding: '16px', // 增加内边距
                                                height: '100%',
                                                display: 'flex',
                                                flexDirection: 'column',
                                                justifyContent: 'space-between'
                                            }}
                                        >
                                            <div style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                marginBottom: '8px'
                                            }}>
                                                {getFileIcon(artifact.mime_type)}
                                                <div style={{ marginLeft: '10px', flex: 1, minWidth: 0 }}>
                                                    <Text strong style={{
                                                        fontSize: '13px',
                                                        color: '#262626',
                                                        display: 'block',
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                        whiteSpace: 'nowrap'
                                                    }}>
                                                        {artifact.original_file_name.length > 18
                                                            ? artifact.original_file_name.substring(0, 18) + '...'
                                                            : artifact.original_file_name
                                                        }
                                                    </Text>
                                                </div>
                                                <Tooltip title="Download">
                                                    <Button
                                                        type="text"
                                                        size="small"
                                                        icon={<DownloadOutlined />}
                                                        onClick={() => handleDownload(artifact)}
                                                        style={{
                                                            color: '#1890ff',
                                                            fontSize: '14px',
                                                            marginLeft: '8px'
                                                        }}
                                                    />
                                                </Tooltip>
                                            </div>
                                            <div style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                marginLeft: '32px' // 与文件名左对齐
                                            }}>
                                                <Text type="secondary" style={{
                                                    fontSize: '11px',
                                                    color: '#8c8c8c'
                                                }}>
                                                    {formatFileSize(artifact.file_size)}
                                                </Text>
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </>
    )
}

// 导出包装了错误边界的组件
export const ArtifactsDrawer: React.FC<ArtifactsDrawerProps> = (props) => {
    return (
        <ErrorBoundary>
            <ArtifactsDrawerContent {...props} />
        </ErrorBoundary>
    )
}
