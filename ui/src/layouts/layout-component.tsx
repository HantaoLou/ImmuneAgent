import React from 'react'
import { Layout } from 'antd'
import Header from '../layouts/header-component'
import { Outlet } from 'react-router-dom'

const { Content } = Layout

const Home: React.FC = () => {
  // 底部功能卡片数据

  return (
    <Layout style={{ width: '100%', height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {/* 顶部导航 */}
      <Header />

      {/* 内容区域 */}
      <Content style={{ 
        backgroundColor: '#fff', 
        display: 'flex', 
        flexDirection: 'column', 
        width: '100%', 
        maxWidth: '100%', 
        overflow: 'hidden',
        flex: 1,
        minHeight: 0
      }}>
        <Outlet />
      </Content>
    </Layout>
  )
}

export default Home
