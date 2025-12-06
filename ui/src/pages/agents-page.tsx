import React, { useState, useEffect, useCallback } from 'react'
import { Row, Col, Spin, App, Typography, Divider } from 'antd'
import { RobotOutlined, ExperimentOutlined } from '@ant-design/icons'
import { colors } from '../styles/tokens'
import { getUsecases, type UsecaseInfo } from '../services/sessions-service'
import UsecaseCard from '../components/common/usecase-card'
import { useNavigate } from 'react-router-dom'
import { useUsecase } from '../contexts/UsecaseContext'


const Agents: React.FC = () => {
  const navigate = useNavigate();
  const { message } = App.useApp()
  const { setSelectedUsecase } = useUsecase()
  const [usecases, setUsecases] = useState<UsecaseInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchUsecases()
  }, [])

  const fetchUsecases = async () => {
    try {
      const usecaseData = await getUsecases()
      setUsecases(usecaseData)
    } catch (error) {
      console.error('Failed to fetch usecases:', error)
      message.error('Failed to load usecases')
    } finally {
      setLoading(false)
    }
  }

  const renderUsecaseCard = useCallback((usecase: UsecaseInfo) => (
    <Col xs={24} sm={12} lg={8} xl={6} key={usecase.name}>
      <UsecaseCard
        usecase={usecase}
        onSelect={async () => {
          // Set usecase in context and navigate to chat
          setSelectedUsecase(usecase)
          navigate('/chat')
        }}
      />
    </Col>
  ), [setSelectedUsecase, navigate])

  if (loading) {
    return (
      <div 
        className="flex flex-col justify-center items-center h-[60vh] rounded-lg m-6"
        style={{
          background: `linear-gradient(135deg, ${colors.background.primary} 0%, ${colors.background.secondary} 100%)`
        }}
      >
        <Spin size="large" />
        <Typography.Text 
          className="mt-4 text-gray-600 text-lg"
        >
          Loading AI Agents...
        </Typography.Text>
      </div>
    )
  }

  return (
    <div 
      className="w-full p-6 mx-auto h-screen overflow-hidden flex flex-col justify-center items-center"
      style={{ 
        background: `linear-gradient(135deg, ${colors.background.primary} 0%, ${colors.background.secondary} 100%)`
      }}
    >
      {/* 页面头部 */}
      <div className="w-full text-center mb-6 py-4 flex-shrink-0">
        <div className="flex items-center justify-center mb-4">
          <RobotOutlined 
            className="text-5xl mr-3"
            style={{ color: colors.primary[500] }}
          />
          <Typography.Title 
            level={1} 
            className="m-0 text-4xl font-bold"
            style={{ 
              background: `linear-gradient(135deg, ${colors.primary[600]} 0%, ${colors.primary[400]} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}
          >
            AI Agent Center
          </Typography.Title>
        </div>
        
        <Typography.Paragraph 
          className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed"
        >
          Choose from our specialized AI agents to accelerate your biomedical research
        </Typography.Paragraph>
        
        <Divider 
          className="my-6"
          style={{ borderColor: colors.border.primary }}
        />
      </div>

      {/* 代理卡片网格 */}
      <div 
        className="w-[80%] bg-white rounded-xl p-6 shadow-lg border flex-1 overflow-auto"
        style={{
          borderColor: colors.border.primary
        }}
      >
        <Row gutter={[24, 24]}>
          {usecases.map(renderUsecaseCard)}
        </Row>
      </div>

      {/* 底部装饰 */}
      <div className="text-center mt-4 py-3 flex-shrink-0">
        <Typography.Text 
          className="text-gray-400 text-sm"
        >
          <ExperimentOutlined className="mr-2" />
          Advanced AI-powered biomedical research platform
        </Typography.Text>
      </div>
    </div>
  )
}

export default Agents