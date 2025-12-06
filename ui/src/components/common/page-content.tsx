import React from 'react'
import { spacing } from '../../styles/tokens'
import { Card } from 'antd'

interface PageContentProps {
  children: React.ReactNode
  style?: React.CSSProperties
}

export const PageContent: React.FC<PageContentProps> = ({ children }) => {
  return (
    <Card
      variant="borderless"
      style={{
        margin: spacing[2],
      }}
    >
      {children}
    </Card>
  )
}

export default PageContent
