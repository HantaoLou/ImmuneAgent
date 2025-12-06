import React from 'react'
import { Empty, Typography } from 'antd'
import { colors, spacing } from '../../styles/tokens'

const { Title, Text } = Typography

interface EmptyStateProps {
  title?: string
  description?: string
  icon?: React.ReactNode
}

const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Session Selected',
  description = 'Please select a session from the sidebar to start chatting, or create a new session.',
  icon,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        padding: spacing[8],
        textAlign: 'center',
      }}
    >
      <Empty
        image={icon || Empty.PRESENTED_IMAGE_SIMPLE}
        styles={{
          image: {
            height: 120,
            marginBottom: spacing[4],
          },
        }}
        description={
          <div>
            <Title
              level={4}
              style={{ color: colors.text.secondary, marginBottom: spacing[2] }}
            >
              {title}
            </Title>
            <Text style={{ color: colors.text.tertiary, fontSize: 16 }}>
              {description}
            </Text>
          </div>
        }
      />
    </div>
  )
}

export default EmptyState
