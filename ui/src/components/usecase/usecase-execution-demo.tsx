import React, { useState, useEffect } from 'react'
import { Card, Button, Steps, Typography, Space, Tag, Progress, Alert, Spin, App } from 'antd'
import { 
  PlayCircleOutlined, 
  StopOutlined, 
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  DownloadOutlined
} from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import { type Session } from '../../services/sessions-service'
import { getSessionArtifacts, downloadArtifact, type SessionArtifact } from '../../services/artifacts-service'

const { Title, Text } = Typography
const { Step } = Steps

interface UsecaseExecutionDemoProps {
  usecaseName: string
  usecaseDisplayName: string
  onSessionCreated?: (session: Session) => void
}

interface ExecutionStep {
  key: string
  title: string
  description: string
  status: 'wait' | 'process' | 'finish' | 'error'
  icon?: React.ReactNode
}

const UsecaseExecutionDemo: React.FC<UsecaseExecutionDemoProps> = ({
  usecaseName,
  usecaseDisplayName,
  onSessionCreated: _onSessionCreated
}) => {
  const { message } = App.useApp()
  const [session, _setSession] = useState<Session | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionProgress, setExecutionProgress] = useState(0)
  const [artifacts, setArtifacts] = useState<SessionArtifact[]>([])
  const [executionSteps, setExecutionSteps] = useState<ExecutionStep[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [executionLog, setExecutionLog] = useState<string[]>([])

  // Initialize execution steps based on usecase
  useEffect(() => {
    const steps = getExecutionSteps(usecaseName)
    setExecutionSteps(steps)
  }, [usecaseName])

  const getExecutionSteps = (usecase: string): ExecutionStep[] => {
    const stepMap: Record<string, ExecutionStep[]> = {
      research: [
        { key: 'init', title: 'Initialize Research', description: 'Setting up research parameters and context', status: 'wait' },
        { key: 'search', title: 'Literature Search', description: 'Searching relevant papers and articles', status: 'wait' },
        { key: 'analyze', title: 'Content Analysis', description: 'Analyzing and summarizing findings', status: 'wait' },
        { key: 'synthesize', title: 'Synthesis', description: 'Synthesizing research insights', status: 'wait' },
        { key: 'report', title: 'Generate Report', description: 'Creating final research report', status: 'wait' }
      ],
      antibody: [
        { key: 'init', title: 'Initialize Analysis', description: 'Loading antibody sequence data', status: 'wait' },
        { key: 'structure', title: 'Structure Prediction', description: 'Predicting antibody structure', status: 'wait' },
        { key: 'binding', title: 'Binding Analysis', description: 'Analyzing binding sites and interactions', status: 'wait' },
        { key: 'optimization', title: 'Optimization', description: 'Optimizing antibody properties', status: 'wait' },
        { key: 'report', title: 'Generate Report', description: 'Creating analysis report', status: 'wait' }
      ],
      cell: [
        { key: 'init', title: 'Initialize Analysis', description: 'Loading cell data and parameters', status: 'wait' },
        { key: 'classification', title: 'Cell Classification', description: 'Classifying cell types and states', status: 'wait' },
        { key: 'analysis', title: 'Functional Analysis', description: 'Analyzing cell functions and pathways', status: 'wait' },
        { key: 'visualization', title: 'Visualization', description: 'Creating visual representations', status: 'wait' },
        { key: 'report', title: 'Generate Report', description: 'Creating analysis report', status: 'wait' }
      ],
      immunity: [
        { key: 'init', title: 'Initialize Analysis', description: 'Setting up immune system parameters', status: 'wait' },
        { key: 'response', title: 'Immune Response', description: 'Analyzing immune response patterns', status: 'wait' },
        { key: 'pathways', title: 'Pathway Analysis', description: 'Analyzing immune pathways', status: 'wait' },
        { key: 'disease', title: 'Disease Association', description: 'Identifying disease associations', status: 'wait' },
        { key: 'report', title: 'Generate Report', description: 'Creating analysis report', status: 'wait' }
      ],
      immunology: [
        { key: 'init', title: 'Initialize Research', description: 'Setting up immunology research parameters', status: 'wait' },
        { key: 'data', title: 'Data Collection', description: 'Collecting immunological data', status: 'wait' },
        { key: 'analysis', title: 'Data Analysis', description: 'Analyzing immunological patterns', status: 'wait' },
        { key: 'validation', title: 'Validation', description: 'Validating research findings', status: 'wait' },
        { key: 'report', title: 'Generate Report', description: 'Creating research report', status: 'wait' }
      ],
      deepagents: [
        { key: 'init', title: 'Initialize Agent', description: 'Setting up deep agent parameters', status: 'wait' },
        { key: 'planning', title: 'Task Planning', description: 'Planning multi-step tasks', status: 'wait' },
        { key: 'execution', title: 'Task Execution', description: 'Executing planned tasks', status: 'wait' },
        { key: 'monitoring', title: 'Progress Monitoring', description: 'Monitoring execution progress', status: 'wait' },
        { key: 'completion', title: 'Task Completion', description: 'Completing and summarizing tasks', status: 'wait' }
      ]
    }
    return stepMap[usecase] || [
      { key: 'init', title: 'Initialize', description: 'Initializing process', status: 'wait' },
      { key: 'process', title: 'Processing', description: 'Processing data', status: 'wait' },
      { key: 'complete', title: 'Complete', description: 'Process completed', status: 'wait' }
    ]
  }

  const handleStartExecution = async () => {
    try {
      setIsExecuting(true)
      setExecutionProgress(0)
      setCurrentStep(0)
      setExecutionLog([])
      
      
      
      // Simulate execution steps
      await simulateExecution()
      
    } catch (error) {
      console.error('Failed to start execution:', error)
      message.error('Failed to start execution')
      setIsExecuting(false)
    }
  }

  const simulateExecution = async () => {
    const steps = executionSteps
    const stepDuration = 2000 // 2 seconds per step
    
    for (let i = 0; i < steps.length; i++) {
      // Update current step
      setCurrentStep(i)
      
      // Update step status
      const updatedSteps = [...steps]
      updatedSteps[i].status = 'process'
      setExecutionSteps(updatedSteps)
      
      // Add log entry
      setExecutionLog(prev => [...prev, `Starting ${steps[i].title}...`])
      
      // Simulate step execution
      await new Promise(resolve => setTimeout(resolve, stepDuration))
      
      // Complete step
      updatedSteps[i].status = 'finish'
      setExecutionSteps(updatedSteps)
      setExecutionProgress(((i + 1) / steps.length) * 100)
      
      setExecutionLog(prev => [...prev, `Completed ${steps[i].title}`])
    }
    
    // Execution completed
    setIsExecuting(false)
    setExecutionLog(prev => [...prev, 'Execution completed successfully!'])
    
    // Simulate artifact generation
    await fetchArtifacts()
  }

  const fetchArtifacts = async () => {
    if (!session) return
    
    try {
      const data = await getSessionArtifacts(session.id)
      setArtifacts(data)
    } catch (error) {
      console.error('Failed to fetch artifacts:', error)
    }
  }

  const handleDownloadArtifact = async (artifact: SessionArtifact) => {
    try {
      await downloadArtifact(artifact)  // 传入artifact对象以支持OSS直接下载
      message.success(`Downloaded ${artifact.original_file_name}`)
    } catch (error) {
      message.error('Failed to download artifact')
    }
  }

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'finish':
        return <CheckCircleOutlined style={{ color: colors.success[500] }} />
      case 'process':
        return <Spin size="small" />
      case 'error':
        return <ExclamationCircleOutlined style={{ color: colors.error[500] }} />
      default:
        return null
    }
  }

  return (
    <div style={{ padding: spacing[4] }}>
      <Card
        title={
          <Space>
            <FileTextOutlined style={{ color: colors.primary[500] }} />
            <Title level={4} style={{ margin: 0 }}>
              {usecaseDisplayName} Execution Demo
            </Title>
          </Space>
        }
        extra={
          <Space>
            {!isExecuting && !session && (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleStartExecution}
                style={{ borderRadius: spacing.base }}
              >
                Start Execution
              </Button>
            )}
            {isExecuting && (
              <Button
                icon={<StopOutlined />}
                onClick={() => setIsExecuting(false)}
                style={{ borderRadius: spacing.base }}
              >
                Stop
              </Button>
            )}
            {session && !isExecuting && (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => window.open(`/chat?sessionId=${session.id}`, '_blank')}
                style={{ borderRadius: spacing.base }}
              >
                Open Chat
              </Button>
            )}
          </Space>
        }
        style={{ marginBottom: spacing[4] }}
      >
        {/* Execution Progress */}
        {isExecuting && (
          <div style={{ marginBottom: spacing[4] }}>
            <Text strong>Execution Progress</Text>
            <Progress 
              percent={executionProgress} 
              status={isExecuting ? 'active' : 'success'}
              style={{ marginTop: spacing[2] }}
            />
          </div>
        )}

        {/* Execution Steps */}
        <div style={{ marginBottom: spacing[4] }}>
          <Text strong>Execution Steps</Text>
          <Steps
            current={currentStep}
            direction="vertical"
            size="small"
            style={{ marginTop: spacing[2] }}
          >
            {executionSteps.map((step) => (
              <Step
                key={step.key}
                title={step.title}
                description={step.description}
                status={step.status}
                icon={getStepIcon(step.status)}
              />
            ))}
          </Steps>
        </div>

        {/* Execution Log */}
        {executionLog.length > 0 && (
          <div style={{ marginBottom: spacing[4] }}>
            <Text strong>Execution Log</Text>
            <div 
              style={{ 
                backgroundColor: colors.background.secondary,
                padding: spacing[3],
                borderRadius: spacing.base,
                marginTop: spacing[2],
                maxHeight: '200px',
                overflowY: 'auto'
              }}
            >
              {executionLog.map((log, index) => (
                <div key={index} style={{ marginBottom: spacing[1] }}>
                  <Text code style={{ fontSize: '12px' }}>
                    [{new Date().toLocaleTimeString()}] {log}
                  </Text>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Artifacts */}
        {artifacts.length > 0 && (
          <div>
            <Text strong>Generated Artifacts</Text>
            <div style={{ marginTop: spacing[2] }}>
              {artifacts.map((artifact) => (
                <Card
                  key={artifact.id}
                  size="small"
                  style={{ marginBottom: spacing[2] }}
                  actions={[
                    <Button
                      key="download"
                      type="text"
                      icon={<DownloadOutlined />}
                      onClick={() => handleDownloadArtifact(artifact)}
                    >
                      Download
                    </Button>
                  ]}
                >
                  <Space>
                    <FileTextOutlined />
                    <Text strong>{artifact.original_file_name}</Text>
                    <Tag color="blue">{artifact.mime_type}</Tag>
                    <Text type="secondary">
                      {(artifact.file_size / 1024).toFixed(1)} KB
                    </Text>
                  </Space>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Session Info */}
        {session && (
          <Alert
            message="Session Created"
            description={
              <div>
                <Text>Session ID: <Text code>{session.id}</Text></Text>
                <br />
                <Text>Usecase: <Tag color="green">{session.usecase}</Tag></Text>
                <br />
                <Button
                  type="link"
                  onClick={() => window.open(`/chat?sessionId=${session.id}`, '_blank')}
                  style={{ padding: 0 }}
                >
                  Open in Chat Interface →
                </Button>
              </div>
            }
            type="success"
            showIcon
            style={{ marginTop: spacing[4] }}
          />
        )}
      </Card>
    </div>
  )
}

export default UsecaseExecutionDemo
