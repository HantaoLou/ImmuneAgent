import React, { useState } from 'react'
import { Card, Progress, Typography, Space, Tag, Timeline, Alert, Button } from 'antd'
import { 
  CheckCircleOutlined, 
  ClockCircleOutlined, 
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'

const { Title, Text } = Typography

interface ExecutionStatusProps {
  isExecuting: boolean
  currentStep: number
  totalSteps: number
  executionLog: string[]
  onStart?: () => void
  onPause?: () => void
  onResume?: () => void
  onStop?: () => void
}

const ExecutionStatus: React.FC<ExecutionStatusProps> = ({
  isExecuting,
  currentStep,
  totalSteps,
  executionLog,
  onStart,
  onPause,
  onResume,
  onStop
}) => {
  const [isPaused, setIsPaused] = useState(false)
  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0

  const handlePause = () => {
    setIsPaused(true)
    onPause?.()
  }

  const handleResume = () => {
    setIsPaused(false)
    onResume?.()
  }

  const getStatusColor = () => {
    if (isPaused) return colors.warning[500]
    if (isExecuting) return colors.primary[500]
    return colors.success[500]
  }

  const getStatusText = () => {
    if (isPaused) return 'Paused'
    if (isExecuting) return 'Executing'
    return 'Completed'
  }

  const getStatusIcon = () => {
    if (isPaused) return <PauseCircleOutlined />
    if (isExecuting) return <PlayCircleOutlined />
    return <CheckCircleOutlined />
  }

  return (
    <Card
      title={
        <Space>
          {getStatusIcon()}
          <Title level={5} style={{ margin: 0 }}>
            Execution Status
          </Title>
          <Tag color={getStatusColor()}>
            {getStatusText()}
          </Tag>
        </Space>
      }
      extra={
        <Space>
          {!isExecuting && !isPaused && onStart && (
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={onStart}
              size="small"
            >
              Start
            </Button>
          )}
          {isExecuting && !isPaused && onPause && (
            <Button
              icon={<PauseCircleOutlined />}
              onClick={handlePause}
              size="small"
            >
              Pause
            </Button>
          )}
          {isPaused && onResume && (
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleResume}
              size="small"
            >
              Resume
            </Button>
          )}
          {isExecuting && onStop && (
            <Button
              danger
              icon={<ExclamationCircleOutlined />}
              onClick={onStop}
              size="small"
            >
              Stop
            </Button>
          )}
        </Space>
      }
      style={{ marginBottom: spacing[4] }}
    >
      {/* Progress Bar */}
      <div style={{ marginBottom: spacing[4] }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: spacing[2] }}>
          <Text strong>Progress</Text>
          <Text>{currentStep} / {totalSteps} steps</Text>
        </div>
        <Progress
          percent={Math.round(progress)}
          status={isPaused ? 'active' : isExecuting ? 'active' : 'success'}
          strokeColor={getStatusColor()}
        />
      </div>

      {/* Execution Timeline */}
      {executionLog.length > 0 && (
        <div>
          <Text strong style={{ marginBottom: spacing[2], display: 'block' }}>
            Execution Timeline
          </Text>
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            <Timeline
              items={executionLog.map((log, index) => ({
                dot: index < currentStep ? 
                  <CheckCircleOutlined style={{ color: colors.success[500] }} /> :
                  index === currentStep ? 
                    <ClockCircleOutlined style={{ color: colors.primary[500] }} /> :
                    <ClockCircleOutlined style={{ color: colors.text.tertiary }} />,
                children: (
                  <div>
                    <Text 
                      style={{ 
                        color: index < currentStep ? colors.success[500] : 
                               index === currentStep ? colors.primary[500] : colors.text.tertiary,
                        fontSize: '12px'
                      }}
                    >
                      {log}
                    </Text>
                    <div style={{ fontSize: '10px', color: colors.text.tertiary }}>
                      {new Date().toLocaleTimeString()}
                    </div>
                  </div>
                )
              }))}
            />
          </div>
        </div>
      )}

      {/* Status Alert */}
      {isPaused && (
        <Alert
          message="Execution Paused"
          description="The execution has been paused. Click Resume to continue."
          type="warning"
          showIcon
          style={{ marginTop: spacing[4] }}
        />
      )}

      {!isExecuting && !isPaused && currentStep > 0 && (
        <Alert
          message="Execution Completed"
          description="The execution has been completed successfully."
          type="success"
          showIcon
          style={{ marginTop: spacing[4] }}
        />
      )}
    </Card>
  )
}

export default ExecutionStatus
