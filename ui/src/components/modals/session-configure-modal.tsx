import React, { useState, useEffect } from 'react'
import { Modal, Form, Input, Button, Space, Typography, App } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import { updateSession, type Session } from '../../services/sessions-service'

const { Title } = Typography
const { TextArea } = Input

interface SessionConfigureModalProps {
  visible: boolean
  onCancel: () => void
  onSuccess: (updatedSession: Session) => void
  session: Session | null
}

const SessionConfigureModal: React.FC<SessionConfigureModalProps> = ({
  visible,
  onCancel,
  onSuccess,
  session,
}) => {
  const { message } = App.useApp()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)

  // Set form values when session changes
  useEffect(() => {
    if (session) {
      // Parse and format configuration for editing
      let formattedConfig = '{}'
      try {
        // Handle both string and object configurations
        let configObj
        if (typeof session.configuration === 'string') {
          configObj = JSON.parse(session.configuration || '{}')
        } else {
          configObj = session.configuration || {}
        }
        formattedConfig = JSON.stringify(configObj, null, 2)
      } catch (error) {
        console.error('Failed to parse session configuration:', error)
        formattedConfig =
          typeof session.configuration === 'string'
            ? session.configuration
            : JSON.stringify(session.configuration || {}, null, 2)
      }

      form.setFieldsValue({
        usecase: session.usecase,
        configuration: formattedConfig,
      })
    }
  }, [session, form])

  const handleSubmit = async (values: any) => {
    if (!session) return

    setLoading(true)
    try {
      // Validate and minify JSON configuration
      let parsedConfig
      try {
        parsedConfig = JSON.parse(values.configuration)
      } catch (error) {
        message.error('Invalid JSON configuration' + error)
        setLoading(false)
        return
      }

      // Update session with minified JSON
      const updatedSession = await updateSession(session.id, { configuration: parsedConfig })

      message.success('Session configuration updated successfully')
      onSuccess(updatedSession)
      onCancel()
    } catch (error) {
      console.error('Failed to update session:', error)
      message.error('Failed to update session configuration')
    } finally {
      setLoading(false)
    }
  }

  const handleCancel = () => {
    form.resetFields()
    onCancel()
  }

  return (
    <Modal
      title={
        <Space>
          <SettingOutlined style={{ color: colors.primary[500] }} />
          <Title level={4} style={{ margin: 0, color: colors.text.primary }}>
            Configure Session
          </Title>
        </Space>
      }
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={600}
      styles={{
        body: {
          padding: spacing[6],
        },
      }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        style={{ marginTop: spacing[4] }}
      >
        <Form.Item label="Use Case" name="usecase">
          <Input
            placeholder="e.g., Antibody Research, Protein Analysis"
            style={{ borderRadius: spacing.base }}
            disabled
            readOnly
          />
        </Form.Item>

        <Form.Item
          label="Configuration (JSON)"
          name="configuration"
          rules={[{ required: true, message: 'Please enter configuration' }]}
          tooltip="Session configuration in JSON format"
        >
          <TextArea
            rows={8}
            placeholder='{\n  "model": "gpt-4",\n  "temperature": 0.7,\n  "maxTokens": 2000\n}'
            style={{ borderRadius: spacing.base, fontFamily: 'monospace' }}
          />
        </Form.Item>

        {/* Form Actions */}
        <Form.Item style={{ marginTop: spacing[6], marginBottom: 0 }}>
          <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button
              onClick={handleCancel}
              style={{ borderRadius: spacing.base }}
            >
              Cancel
            </Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              style={{ borderRadius: spacing.base }}
            >
              Save Configuration
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  )
}

export default SessionConfigureModal
