import React from 'react'
import { Card, Typography, Tag, Space } from 'antd'
import { 
  ExperimentOutlined, 
  MedicineBoxOutlined, 
  UserOutlined, 
  SafetyOutlined, 
  SearchOutlined, 
  RobotOutlined,
  RocketOutlined
} from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import type { UsecaseInfo } from '../../services/sessions-service'

const { Title, Paragraph } = Typography

interface UsecaseCardProps {
  usecase: UsecaseInfo
  onSelect?: () => void
  compact?: boolean
  className?: string
}

const UsecaseCard: React.FC<UsecaseCardProps> = ({
  usecase,
  onSelect,
  compact = false,
  className = ''
}) => {
  const getUsecaseIcon = (usecaseName: string) => {
    const iconMap: Record<string, React.ReactNode> = {
      research: <SearchOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />,
      antibody: <MedicineBoxOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />,
      cell: <UserOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />,
      immunity: <SafetyOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />,
      immunology: <ExperimentOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />,
      deepagents: <RobotOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />
    }
    return iconMap[usecaseName] || <RocketOutlined style={{ fontSize: compact ? '20px' : '24px', color: colors.primary[500] }} />
  }

  const getUsecaseDisplayName = (usecaseName: string) => {
    const nameMap: Record<string, string> = {
      research: 'Intelligent Research',
      antibody: 'Antibody Analysis',
      cell: 'Cell Analysis', 
      immunity: 'Immunity Analysis',
      immunology: 'Immunology Research',
      deepagents: 'Deep Agents'
    }
    return nameMap[usecaseName] || usecaseName
  }

  const getUsecaseDescription = (usecaseName: string) => {
    const descMap: Record<string, string> = {
      research: 'AI-powered intelligent research assistant with deep literature analysis and research recommendations',
      antibody: 'Professional antibody sequence analysis tool with structure prediction and functional analysis',
      cell: 'Cell biology analysis platform providing cell type identification and functional analysis',
      immunity: 'Immune system analysis tool supporting immune response and disease association research',
      immunology: 'Immunology research platform providing comprehensive immunological data analysis',
      deepagents: 'Deep AI agent system supporting complex multi-step task execution'
    }
    return descMap[usecaseName] || 'AI Intelligent Analysis Tool'
  }

  const getUsecaseTagColor = (usecaseName: string) => {
    const colorMap: Record<string, string> = {
      research: 'blue',
      antibody: 'green', 
      cell: 'purple',
      immunity: 'orange',
      immunology: 'red',
      deepagents: 'cyan'
    }
    return colorMap[usecaseName] || 'default'
  }


  if (compact) {
    return (
      <Card
        hoverable
        className={`bg-white rounded-xl shadow-sm h-32 flex items-center justify-center hover:shadow-lg transition-all duration-300 cursor-pointer ${className}`}
        onClick={() => onSelect?.()}
        styles={{
          body: { 
            padding: '16px', 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center' 
          }
        }}
      >
        <Space direction="vertical" align="center">
          <div className="mb-2">{getUsecaseIcon(usecase.name)}</div>
          <div className="text-sm text-gray-800 font-semibold leading-tight text-center">
            {getUsecaseDisplayName(usecase.name)}
          </div>
          <Tag 
            color={getUsecaseTagColor(usecase.name)}
            style={{ fontSize: '10px' }}
          >
            {usecase.name.toUpperCase()}
          </Tag>
        </Space>
      </Card>
    )
  }

  return (
    <Card
      hoverable
      className="h-full rounded-xl border border-gray-200"
      styles={{ body: { padding: spacing[6] } }}
      onClick={() => onSelect?.()}
    >
      <div style={{ textAlign: 'center', marginBottom: spacing[4] }}>
        {getUsecaseIcon(usecase.name)}
      </div>
      
      <div style={{ textAlign: 'center', marginBottom: spacing[3] }}>
        <Title level={4} style={{ margin: 0, color: colors.text.primary }}>
          {getUsecaseDisplayName(usecase.name)}
        </Title>
        <Tag 
          color={getUsecaseTagColor(usecase.name)}
          style={{ marginTop: spacing[2] }}
        >
          {usecase.name.toUpperCase()}
        </Tag>
      </div>

      <Paragraph 
        style={{ 
          color: colors.text.secondary,
          textAlign: 'center',
          margin: 0,
          fontSize: '14px'
        }}
      >
        {getUsecaseDescription(usecase.name)}
      </Paragraph>
    </Card>
  )
}

export default UsecaseCard
