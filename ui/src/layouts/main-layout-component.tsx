import React, { useState, useEffect, useCallback } from 'react'
import { Layout, Button, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  getSessions,
  createSession,
  deleteSession,
  updateSession,
  type Session,
} from '../services/sessions-service'
import {
  PlusOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { colors, spacing } from '../styles/tokens'
import SessionsList from '../components/sidebar/sessions-list-component'
import SessionConfigureModal from '../components/modals/session-configure-modal'
import EmptyState from '../components/common/empty-state'
import Chat from '../pages/chat-page'
import { useUsecase } from '../contexts/UsecaseContext'

const { Sider, Content } = Layout

const MainLayout: React.FC = () => {
  const { selectedUsecase } = useUsecase()
  const navigate = useNavigate()
  
  // State management
  const [selectedSession, setSelectedSession] = useState<string>('')
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(false)
  const [isCreatingSession, setIsCreatingSession] = useState(false)
  
  // Modal states
  const [configureModalVisible, setConfigureModalVisible] = useState(false)
  const [sessionToConfig, setSessionToConfig] = useState<Session | null>(null)

  // Load sessions and handle usecase selection
  useEffect(() => {
    const loadData = async () => {
      try {
        const sessionData = await getSessions() // todo: 依据selectedUsecase筛选sessionData
        setSessions(sessionData)
      } catch (error) {
        console.error('Failed to load data:', error)
        navigate('/agents')
      }
    }
    
    loadData()
  }, [selectedUsecase])

  // Create session from usecase
  const createSessionFromUsecase = async (usecaseName: string) => {
    setIsCreatingSession(true)
    setLoading(true)
    
    try {
      const newSession = await createSession(usecaseName)
      setSessions((prevSessions) => [newSession, ...prevSessions])
      setSelectedSession(newSession.id)
    } catch (error) {
      console.error('Failed to create session from usecase:', error)
    } finally {
      setLoading(false)
      setIsCreatingSession(false)
    }
  }

  // Event handlers
  const handleNewSession = useCallback(async () => {
    if (!selectedUsecase) {
      console.log('No current usecase available for creating session')
      navigate('/agents')
      return
    }
    
    if (isCreatingSession) {
      console.log('Session creation already in progress, skipping...')
      return
    }
    
    await createSessionFromUsecase(selectedUsecase.name)
  }, [selectedUsecase, isCreatingSession])


  const handleSessionSelect = (sessionId: string) => {
    setSelectedSession(sessionId)
  }

  const handleSessionDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId)
      setSessions(prevSessions => 
        prevSessions.filter(session => session.id !== sessionId)
      )
      if (selectedSession === sessionId) {
        setSelectedSession('')
      }
    } catch (error) {
      console.error('Failed to delete session:', error)
    }
  }

  const handleSessionUpdateName = async (sessionId: string, newName: string) => {
    try {
      // 调用后端API保存名称
      const updatedSession = await updateSession(sessionId, { name: newName })
      // 更新本地状态
      setSessions(prevSessions =>
        prevSessions.map(session =>
          session.id === sessionId ? { ...session, name: updatedSession.name || newName } : session
        )
      )
    } catch (error) {
      console.error('Failed to update session name:', error)
      // 即使失败也更新本地状态，保持UI响应
      setSessions(prevSessions =>
        prevSessions.map(session =>
          session.id === sessionId ? { ...session, name: newName } : session
        )
      )
    }
  }

  const handleConfigureSuccess = (updatedSession: Session) => {
    setSessions(sessions.map(session =>
      session.id === updatedSession.id ? updatedSession : session
    ))
  }

  const handleConfigureCancel = () => {
    setConfigureModalVisible(false)
    setSessionToConfig(null)
  }

  const handleSessionConfigure = (session: Session) => {
    setConfigureModalVisible(true)
    setSessionToConfig(session)
  }

  return (
    <>
      <Layout style={{ width: '100%', height: '100vh', overflow: 'hidden' }}>
        <Layout style={{ height: '100vh', overflow: 'hidden' }}>
          {/* Left Sidebar */}
          <Sider
            width={280}
            style={{
              background: colors.background.secondary,
              borderRight: `1px solid ${colors.border.primary}`,
              height: '100vh',
              overflow: 'hidden',
            }}
          >
            <div style={{ 
              padding: spacing[4], 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              overflow: 'hidden',
              boxSizing: 'border-box'
            }}>
              {/* Usecase Info */}
              {selectedUsecase && (
                <div style={{ 
                  marginBottom: spacing[4], 
                  padding: spacing[3], 
                  background: colors.background.primary,
                  borderRadius: spacing.base,
                  border: `1px solid ${colors.border.primary}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: spacing[2] }}>
                    <RocketOutlined style={{ color: colors.primary[500], fontSize: '16px' }} />
                    <Typography.Text strong style={{ color: colors.text.primary, fontSize: '14px' }}>
                      {selectedUsecase.name.charAt(0).toUpperCase() + selectedUsecase.name.slice(1)}
                    </Typography.Text>
                  </div>
                </div>
              )}

              {/* New Session Button */}
              <div style={{ 
                padding: spacing[2], 
                borderBottom: `1px solid ${colors.border.primary}`,
                backgroundColor: colors.background.primary,
                marginBottom: spacing[2]
              }}>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={handleNewSession}
                  loading={loading}
                  style={{ width: '100%', height: 40 }}
                >
                  New Session
                </Button>
              </div>

              {/* Sessions List */}
              <div style={{ 
                flex: 1, 
                overflow: 'hidden', 
                display: 'flex', 
                flexDirection: 'column',
                minHeight: 0
              }}>
                <SessionsList
                  sessions={sessions}
                  selectedSession={selectedSession}
                  onSessionSelect={handleSessionSelect}
                  onSessionDelete={handleSessionDelete}
                  onSessionUpdateName={handleSessionUpdateName}
                  onSessionConfigure={handleSessionConfigure}
                />
              </div>
            </div>
          </Sider>

          {/* Main Content Area */}
          <Content style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            overflow: 'hidden',
          }}>
            {selectedSession !== '' ? (
              <Chat sessionId={selectedSession} />
            ) : (
              <EmptyState />
            )}
          </Content>
        </Layout>
      </Layout>

      {/* Modals */}
      <SessionConfigureModal
        visible={configureModalVisible}
        onCancel={handleConfigureCancel}
        onSuccess={handleConfigureSuccess}
        session={sessionToConfig}
      />
    </>
  )
}

export default MainLayout