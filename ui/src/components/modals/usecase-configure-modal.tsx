import React, { useState, useEffect } from 'react'
import { Modal, Form, Input, Button, Space, Typography, App, Collapse, Divider, Card } from 'antd'
import { 
  SettingOutlined,
  ExperimentOutlined, 
  MedicineBoxOutlined, 
  UserOutlined, 
  SafetyOutlined, 
  SearchOutlined, 
  RobotOutlined,
  RocketOutlined,
  ApiOutlined,
  DatabaseOutlined,
  ToolOutlined
} from '@ant-design/icons'
import { colors, spacing } from '../../styles/tokens'
import { type UsecaseInfo } from '../../services/sessions-service'

const { Title, Text } = Typography
const { TextArea } = Input
const { Panel } = Collapse

interface UsecaseConfigureModalProps {
  visible: boolean
  onCancel: () => void
  onSuccess: (updatedUsecase: UsecaseInfo) => void
  usecase: UsecaseInfo | null
}

const UsecaseConfigureModal: React.FC<UsecaseConfigureModalProps> = ({
  visible,
  onCancel,
  onSuccess,
  usecase,
}) => {
  const { message } = App.useApp()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)

  // Get usecase icon based on name
  const getUsecaseIcon = (usecaseName: string) => {
    const iconMap: Record<string, React.ReactNode> = {
      research: <SearchOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />,
      antibody: <MedicineBoxOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />,
      cell: <UserOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />,
      immunity: <SafetyOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />,
      immunology: <ExperimentOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />,
      deepagents: <RobotOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />
    }
    return iconMap[usecaseName] || <RocketOutlined style={{ fontSize: '20px', color: colors.primary[500] }} />
  }

  // Set form values when usecase changes
  useEffect(() => {
    if (usecase && visible) {
      // Parse and format configuration for editing
      let formattedConfig = '{}'
      try {
        // Handle both string and object configurations from usecase
        let configObj
        if (usecase.configuration) {
          if (typeof usecase.configuration === 'string') {
            configObj = JSON.parse(usecase.configuration)
          } else {
            configObj = usecase.configuration
          }
        } else {
          // Default configuration if none exists
          configObj = {
            name: usecase.name,
            description: usecase.description,
            default_configuration: {
              model_config: {
                default_model: {
                  provider: "OpenAI",
                  model: "gpt-4.1",
                  params: {
                    base_url: "https://xiaoai.plus/v1",
                    api_key: "",
                    temperature: 0.2
                  }
                }
              },
              mcp_config: {
                service_ids: []
              },
              tavily_api_key: ""
            }
          }
        }
        formattedConfig = JSON.stringify(configObj, null, 2)
      } catch (error) {
        console.error('Failed to parse usecase configuration:', error)
        // Fallback to default configuration
        const defaultConfig = {
          name: usecase.name,
          description: usecase.description,
          default_configuration: {
            model_config: {
              default_model: {
                provider: "OpenAI",
                model: "gpt-4.1",
                params: {
                  base_url: "https://xiaoai.plus/v1",
                  api_key: "",
                  temperature: 0.2
                }
              }
            },
            mcp_config: {
              service_ids: []
            },
            tavily_api_key: ""
          }
        }
        formattedConfig = JSON.stringify(defaultConfig, null, 2)
      }

      form.setFieldsValue({
        name: usecase.name,
        description: usecase.description,
        configuration: formattedConfig,
      })
    }
  }, [usecase, visible, form])

  const handleSubmit = async (values: any) => {
    if (!usecase) return

    setLoading(true)
    try {
      // Validate and minify JSON configuration
      let parsedConfig
      try {
        parsedConfig = JSON.parse(values.configuration)
      } catch (error) {
        message.error('Invalid JSON configuration')
        setLoading(false)
        return
      }

      // Simulate API call to update usecase configuration
      // In a real implementation, this would call an API endpoint
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const updatedUsecase: UsecaseInfo = {
        name: values.name,
        description: values.description,
        configuration: parsedConfig
      }

      message.success('Usecase configuration updated successfully')
      onSuccess(updatedUsecase)
      onCancel()
    } catch (error) {
      console.error('Failed to update usecase:', error)
      message.error('Failed to update usecase configuration')
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
          {usecase ? getUsecaseIcon(usecase.name) : <SettingOutlined style={{ color: colors.primary[500] }} />}
          <Title level={4} style={{ margin: 0, color: colors.text.primary }}>
            Configure Usecase
          </Title>
        </Space>
      }
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={800}
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
        {/* Basic Information */}
        <Card 
          title={
            <Space>
              <SettingOutlined style={{ color: colors.primary[500] }} />
              <Text strong>Basic Information</Text>
            </Space>
          }
          style={{ marginBottom: spacing[4] }}
        >
          <Form.Item 
            label="Usecase Name" 
            name="name"
            rules={[{ required: true, message: 'Please enter usecase name' }]}
          >
            <Input
              placeholder="e.g., immunity, antibody, research"
              style={{ borderRadius: spacing.base }}
            />
          </Form.Item>

          <Form.Item
            label="Description"
            name="description"
            rules={[{ required: true, message: 'Please enter description' }]}
          >
            <TextArea
              rows={3}
              placeholder="Describe the purpose and capabilities of this usecase..."
              style={{ borderRadius: spacing.base }}
            />
          </Form.Item>
        </Card>

        {/* Advanced Configuration */}
        <Card 
          title={
            <Space>
              <ApiOutlined style={{ color: colors.primary[500] }} />
              <Text strong>Advanced Configuration</Text>
            </Space>
          }
          style={{ marginBottom: spacing[4] }}
        >
          <Collapse defaultActiveKey={['model_config']} ghost>
            <Panel 
              header={
                <Space>
                  <DatabaseOutlined style={{ color: colors.primary[500] }} />
                  <Text strong>Model Configuration</Text>
                </Space>
              } 
              key="model_config"
            >
              <Text type="secondary" style={{ display: 'block', marginBottom: spacing[2] }}>
                Configure AI models for different tasks (default_model, embedding_model, reasoning_model, etc.)
              </Text>
            </Panel>

            <Panel 
              header={
                <Space>
                  <ToolOutlined style={{ color: colors.primary[500] }} />
                  <Text strong>MCP Services</Text>
                </Space>
              } 
              key="mcp_config"
            >
              <Text type="secondary" style={{ display: 'block', marginBottom: spacing[2] }}>
                Configure Model Context Protocol services (metabcr, r_analysis, bcell_analysis, etc.)
              </Text>
            </Panel>

            <Panel 
              header={
                <Space>
                  <SearchOutlined style={{ color: colors.primary[500] }} />
                  <Text strong>External APIs</Text>
                </Space>
              } 
              key="api_config"
            >
              <Text type="secondary" style={{ display: 'block', marginBottom: spacing[2] }}>
                Configure external API keys (Tavily, OpenAI, etc.)
              </Text>
            </Panel>
          </Collapse>

          <Divider />

          <Form.Item
            label="Complete Configuration (JSON)"
            name="configuration"
            rules={[{ required: true, message: 'Please enter configuration' }]}
            tooltip="Complete usecase configuration in JSON format. Includes model_config, mcp_config, and API keys."
          >
            <TextArea
              rows={12}
              placeholder={`{
  "name": "immunity",
  "description": "immunity用例的详细配置和工作流信息",
  "default_configuration": {
    "model_config": {
      "default_model": {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "params": {
          "base_url": "https://xiaoai.plus/v1",
          "api_key": "your-api-key",
          "temperature": 0.2
        }
      }
    },
    "mcp_config": {
      "service_ids": ["metabcr", "r_analysis", "bcell_analysis"]
    },
    "tavily_api_key": "your-tavily-key"
  }
}`}
              style={{ borderRadius: spacing.base, fontFamily: 'monospace' }}
            />
          </Form.Item>
        </Card>

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

export default UsecaseConfigureModal