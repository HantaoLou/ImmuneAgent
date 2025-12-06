import React, { useState, useEffect } from 'react'
import { Card, Input, Button, Form, message, Typography, Space } from 'antd'
import { KeyOutlined, LoginOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const { Title, Text } = Typography

const AuthPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const { login, isAuthenticated } = useAuth()

  useEffect(() => {
    // Check if already authenticated
    if (isAuthenticated) {
      navigate('/agents')
    }
  }, [isAuthenticated, navigate])

  const onFinish = async (values: { token: string }) => {
    setLoading(true)

    try {
      // Use the auth context to login
      login(values.token)

      message.success('Token saved successfully!')

      // Redirect to chat page
      navigate('/agents')
    } catch (error) {
      message.error('Failed to save token. Please try again.' + error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      }}
    >
      <Card
        style={{
          width: 400,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          borderRadius: '12px',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <Space direction="vertical" size="large">
            <KeyOutlined style={{ fontSize: 48, color: '#1890ff' }} />
            <div>
              <Title level={2} style={{ margin: 0, color: '#262626' }}>
                Access Token Required
              </Title>
              <Text type="secondary">
                Please enter your access token to continue
              </Text>
            </div>
          </Space>
        </div>

        <Form
          form={form}
          name="auth"
          onFinish={onFinish}
          autoComplete="off"
          layout="vertical"
        >
          <Form.Item
            name="token"
            label="Access Token"
            rules={[
              {
                required: true,
                message: 'Please enter your access token!',
              },
            ]}
          >
            <Input.Password
              placeholder="Enter your access token"
              size="large"
              prefix={<KeyOutlined />}
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              size="large"
              loading={loading}
              icon={<LoginOutlined />}
              style={{ width: '100%' }}
            >
              Authenticate
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Your token will be stored locally in your browser
          </Text>
        </div>
      </Card>
    </div>
  )
}

export default AuthPage
