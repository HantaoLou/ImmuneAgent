import React, { useState, useEffect } from 'react'
import { Modal, Select, Button, Space, Typography, App, Card, Tag } from 'antd'
import { PlusOutlined, RocketOutlined } from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import { getUsecases, type UsecaseInfo } from '../../services/sessions-service'

const { Title, Text } = Typography
const { Option } = Select

interface UsecaseSelectionModalProps {
  visible: boolean
  onCancel: () => void
  onSelect: (usecase: string) => void
}

const UsecaseSelectionModal: React.FC<UsecaseSelectionModalProps> = ({
  visible,
  onCancel,
  onSelect,
}) => {
  const { message } = App.useApp()
  const [usecases, setUsecases] = useState<UsecaseInfo[]>([])
  const [fetchingUsecases, setFetchingUsecases] = useState(false)
  const [selectedUsecase, setSelectedUsecase] = useState<string>('')

  // Fetch usecases when modal opens
  useEffect(() => {
    if (visible) {
      fetchUsecases()
      setSelectedUsecase('') // Reset selection when modal opens
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  const fetchUsecases = async () => {
    setFetchingUsecases(true)
    try {
      const usecaseData = await getUsecases()
      setUsecases(usecaseData)
    } catch (error) {
      console.error('Failed to fetch usecases:', error)
      message.error('Failed to load available usecases')
    } finally {
      setFetchingUsecases(false)
    }
  }

  const handleUsecaseSelect = (usecase: string) => {
    setSelectedUsecase(usecase)
  }

  const handleConfirm = () => {
    if (selectedUsecase) {
      onSelect(selectedUsecase)
    } else {
      message.warning('Please select a use case')
    }
  }

  const handleCancel = () => {
    setSelectedUsecase('')
    onCancel()
  }

  const getUsecaseIcon = () => {
    return <RocketOutlined style={{ color: colors.text.tertiary }} />
  }

  const getUsecaseTagColor = () => {
    return 'default'
  }

  const renderModalTitle = () => (
    <Space>
      <PlusOutlined style={{ color: colors.primary[500] }} />
      <Title level={4} style={{ margin: 0, color: colors.text.primary }}>
        Select Use Case
      </Title>
    </Space>
  )

  const renderModalFooter = () => [
    <Button
      key="cancel"
      onClick={handleCancel}
      style={{ borderRadius: spacing.base }}
    >
      Cancel
    </Button>,
    <Button
      key="confirm"
      type="primary"
      onClick={handleConfirm}
      disabled={!selectedUsecase}
      style={{ borderRadius: spacing.base }}
    >
      Confirm
    </Button>,
  ]

  const renderLoadingState = () => (
    <div style={{ textAlign: 'center', padding: spacing[8] }}>
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
      <Text style={{ color: colors.text.tertiary, marginTop: spacing[2] }}>
        Loading use cases...
      </Text>
    </div>
  )

  const renderUsecaseOption = (usecase: UsecaseInfo) => (
    <Option key={usecase.name} value={usecase.name}>
      <div style={{ display: 'flex', alignItems: 'center', gap: spacing[2] }}>
        <div style={{ fontSize: '16px' }}>{getUsecaseIcon()}</div>
        <div>
          <Text strong>
            {usecase.name.charAt(0).toUpperCase() + usecase.name.slice(1)}
          </Text>
          <Tag
            color={getUsecaseTagColor()}
            style={{ marginLeft: spacing[2], fontSize: '10px' }}
          >
            {usecase.name.toUpperCase()}
          </Tag>
        </div>
      </div>
    </Option>
  )

  const renderUsecaseSelect = () => (
    <div style={{ marginBottom: spacing[4] }}>
      <Select
        placeholder="Select a use case"
        value={selectedUsecase}
        onChange={(value: string | string[]) => {
          if (typeof value === 'string') {
            handleUsecaseSelect(value)
          }
        }}
        style={{ width: '100%' }}
        size="large"
        showSearch
        filterOption={(input, option) => {
          const optionText = option?.children
          if (!optionText) return false

          if (Array.isArray(optionText)) {
            return optionText.some((text) =>
              String(text).toLowerCase().includes(input.toLowerCase()),
            )
          }
          return String(optionText).toLowerCase().includes(input.toLowerCase())
        }}
      >
        {usecases.map(renderUsecaseOption)}
      </Select>
    </div>
  )

  const renderSelectedUsecaseCard = () => {
    if (!selectedUsecase) return null

    return (
      <Card
        style={{
          border: `1px solid ${colors.border.primary}`,
          borderRadius: spacing.base,
          backgroundColor: colors.background.secondary,
        }}
        bodyStyle={{ padding: spacing[4] }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: spacing[3] }}>
          <div style={{ fontSize: '24px' }}>{getUsecaseIcon()}</div>
          <div>
            <Title level={5} style={{ margin: 0, color: colors.text.primary }}>
              {selectedUsecase.charAt(0).toUpperCase() +
                selectedUsecase.slice(1)}
            </Title>
            <Text style={{ color: colors.text.secondary }}>
              Selected use case
            </Text>
          </div>
        </div>
      </Card>
    )
  }

  const renderModalContent = () => (
    <>
      <div style={{ marginBottom: spacing[4] }}>
        <Text style={{ color: colors.text.secondary }}>
          Choose a use case for your new session. Each use case provides
          different AI capabilities and tools.
        </Text>
      </div>

      {fetchingUsecases ? renderLoadingState() : renderUsecaseSelect()}
      {renderSelectedUsecaseCard()}
    </>
  )

  return (
    <Modal
      title={renderModalTitle()}
      open={visible}
      onCancel={handleCancel}
      footer={renderModalFooter()}
      width={600}
      styles={{
        body: {
          padding: spacing[6],
        },
      }}
    >
      {renderModalContent()}
    </Modal>
  )
}

export default UsecaseSelectionModal
