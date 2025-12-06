import React, { useState, useEffect, useRef } from 'react'
import { Typography } from 'antd'
import { colors, spacing } from '../../styles/tokens'
import type { Session } from '../../services/sessions-service'
import SessionItem from './session-item-component'

// 自定义滚动条样式和动画
const scrollbarStyles = `
  .sessions-list-scroll::-webkit-scrollbar {
    width: 8px;
  }
  .sessions-list-scroll::-webkit-scrollbar-track {
    background: transparent;
  }
  .sessions-list-scroll::-webkit-scrollbar-thumb {
    background: ${colors.border.primary};
    border-radius: 4px;
  }
  .sessions-list-scroll::-webkit-scrollbar-thumb:hover {
    background: ${colors.text.secondary};
  }
  
  @keyframes fadeInOut {
    0% { opacity: 0; transform: translateX(-50%) translateY(10px); }
    20% { opacity: 1; transform: translateX(-50%) translateY(0); }
    80% { opacity: 1; transform: translateX(-50%) translateY(0); }
    100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
  }
`

const { Text } = Typography

interface SessionsListProps {
  sessions: Session[]
  selectedSession: string
  onSessionSelect: (sessionId: string) => void
  onSessionDelete: (sessionId: string) => void
  onSessionUpdateName?: (sessionId: string, newName: string) => void
  onSessionConfigure: (session: Session) => void
}

const SessionsList: React.FC<SessionsListProps> = ({
  sessions,
  selectedSession,
  onSessionSelect,
  onSessionDelete,
  onSessionUpdateName,
  onSessionConfigure,
}) => {
  const [showScrollHint, setShowScrollHint] = useState(false)
  const [canScrollDown, setCanScrollDown] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 检查是否可以滚动
  useEffect(() => {
    const checkScrollability = () => {
      if (scrollRef.current) {
        const { scrollHeight, clientHeight } = scrollRef.current
        const canScroll = scrollHeight > clientHeight
        setShowScrollHint(canScroll)
        setCanScrollDown(canScroll)
      }
    }

    checkScrollability()
    // 监听窗口大小变化
    window.addEventListener('resize', checkScrollability)
    return () => window.removeEventListener('resize', checkScrollability)
  }, [sessions])

  // 监听滚动事件
  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 5
      setCanScrollDown(!isAtBottom)
    }
  }

  return (
    <>
      <style>{scrollbarStyles}</style>
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        minHeight: 0 // 确保flex子元素可以收缩
      }}>
        <Text
          style={{
            color: colors.text.secondary,
            fontSize: '12px',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: spacing[2],
            display: 'block',
            flexShrink: 0,
          }}
        >
          Recent Sessions
        </Text>
        <div 
          ref={scrollRef}
          className="sessions-list-scroll"
          onScroll={handleScroll}
          style={{ 
            flex: 1, 
            overflow: 'auto',
            minHeight: 0, // 确保可以收缩
            maxHeight: 'calc(100vh - 250px)', // 减少底部留白
            paddingRight: '8px', // 为滚动条留出空间
            marginRight: '-8px', // 负边距补偿，避免影响布局
            paddingLeft: '0px', // 移除左侧内边距
            boxSizing: 'border-box',
            position: 'relative', // 为滚动提示定位
          }}>
          {sessions.map((session) => (
            <SessionItem
              key={session.id}
              session={session}
              isSelected={selectedSession === session.id}
              onSelect={onSessionSelect}
              onDelete={onSessionDelete}
              onUpdateName={onSessionUpdateName}
              onConfigure={onSessionConfigure}
            />
          ))}
          
          {/* 滚动提示 */}
          {showScrollHint && canScrollDown && (
            <div style={{
              position: 'absolute',
              bottom: '8px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: colors.background.primary,
              color: colors.text.secondary,
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              border: `1px solid ${colors.border.primary}`,
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
              zIndex: 10,
              animation: 'fadeInOut 2s ease-in-out',
            }}>
              ↓ 更多会话
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default SessionsList
