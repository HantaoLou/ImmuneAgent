import React, { useState } from 'react'
import { Typography, Button, Input } from 'antd'
import {
  MessageOutlined,
  DeleteOutlined,
  EditOutlined,
  CheckOutlined,
  CloseOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import type { Session } from '../../services/sessions-service'

const { Text } = Typography

interface SessionItemProps {
  session: Session
  isSelected: boolean
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
  onUpdateName?: (sessionId: string, newName: string) => void
  onConfigure: (session: Session) => void
}

const SessionItem: React.FC<SessionItemProps> = ({
  session,
  isSelected,
  onSelect,
  onDelete,
  onUpdateName,
  onConfigure,
}) => {
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(session.name || `Session ${session.id.slice(0, 8)}`)
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation() // Prevent triggering onSelect
    onDelete(session.id)
  }

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsEditing(true)
  }

  const handleSave = (e?: React.MouseEvent | React.KeyboardEvent | React.FocusEvent) => {
    e?.stopPropagation()
    if (onUpdateName && editName.trim()) {
      onUpdateName(session.id, editName.trim())
    }
    setIsEditing(false)
  }

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditName(session.name || `Session ${session.id.slice(0, 8)}`)
    setIsEditing(false)
  }

  const formatTime = (timeString?: string) => {
    if (!timeString) return 'Unknown'
    const date = new Date(timeString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) {
      return 'Today'
    } else if (diffDays === 1) {
      return 'Yesterday'
    } else if (diffDays < 7) {
      return `${diffDays} days ago`
    } else {
      return date.toLocaleDateString()
    }
  }


  return (
    <div
      key={session.id}
      onClick={() => onSelect(session.id)}
      style={{
        padding: spacing[3],
        borderRadius: spacing.base,
        cursor: 'pointer',
        marginBottom: spacing[1],
        background: isSelected ? colors.primary[50] : 'transparent',
        border: isSelected
          ? `1px solid ${colors.primary[200]}`
          : '1px solid transparent',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.background = colors.neutral[50]
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.background = 'transparent'
        }
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: spacing[2],
        }}
      >
        <MessageOutlined
          style={{
            color: isSelected ? colors.primary[500] : colors.text.tertiary,
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          {isEditing ? (
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onPressEnter={handleSave}
              onBlur={handleSave}
              autoFocus
              size="small"
              style={{ marginBottom: spacing[1] }}
            />
          ) : (
            <Text
              strong
              style={{
                color: isSelected ? colors.primary[700] : colors.text.primary,
                fontSize: '14px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: 'block',
                marginBottom: spacing[1],
              }}
            >
              {session.name || `Session ${session.id.slice(0, 8)}`}
            </Text>
          )}
          <Text
            style={{
              color: colors.text.tertiary,
              fontSize: '12px',
              display: 'block',
              marginBottom: spacing[1],
            }}
          >
            Created: {formatTime(session.created_at)}
          </Text>
          {session.last_message && (
            <Text
              style={{
                color: colors.text.tertiary,
                fontSize: '11px',
                display: 'block',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              Last: {session.last_message.slice(0, 30)}...
            </Text>
          )}
        </div>
        <div style={{ display: 'flex', gap: spacing[1] }}>
          {isEditing ? (
            <>
              <Button
                type="text"
                size="small"
                icon={<CheckOutlined />}
                onClick={handleSave}
                style={{
                  color: colors.success[500],
                  opacity: 0.7,
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = '0.7'
                }}
              />
              <Button
                type="text"
                size="small"
                icon={<CloseOutlined />}
                onClick={handleCancel}
                style={{
                  color: colors.error[500],
                  opacity: 0.7,
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = '0.7'
                }}
              />
            </>
          ) : (
            <>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={handleEdit}
                style={{
                  color: colors.text.tertiary,
                  opacity: 0.7,
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = colors.primary[500]
                  e.currentTarget.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = colors.text.tertiary
                  e.currentTarget.style.opacity = '0.7'
                }}
              />
              <Button
                type="text"
                size="small"
                icon={<SettingOutlined />}
                onClick={() => onConfigure(session)}
                style={{
                  color: colors.text.tertiary,
                  opacity: 0.7,
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = colors.primary[500]
                  e.currentTarget.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = colors.text.tertiary
                  e.currentTarget.style.opacity = '0.7'
                }}
              />
              <Button
                type="text"
                size="small"
                icon={<DeleteOutlined />}
                onClick={handleDelete}
                style={{
                  color: colors.text.tertiary,
                  opacity: 0.7,
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = colors.error[500]
                  e.currentTarget.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = colors.text.tertiary
                  e.currentTarget.style.opacity = '0.7'
                }}
              />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default SessionItem
