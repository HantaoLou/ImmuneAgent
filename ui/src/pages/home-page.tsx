import React from 'react'
import { Card, Row, Col } from 'antd'


const Home: React.FC = () => {
  // 底部功能卡片数据
  const featureCards = [
    {
      icon: (
        <img
          src="/oxford.jpg"
          alt="University of Oxford"
          className="h-24 object-contain"
        />
      ),
      title: 'University of Oxford'
    },
    {
      icon: (
        <img
          src="/imperial.jpg"
          alt="Imperial"
          className="h-24 object-contain"
        />
      ),
      title: 'Imperial College London'
    },
    {
      icon: (
        <img
          src="/quantinuum_nvidia.jpg"
          alt="NVIDIA"
          className="h-24 object-contain"
        />
      ),
      title: 'Quantinuum & NVIDIA'
    },
    {
      icon: (
        <img
          src="/nankai.jpg"
          alt="Nankai University"
          className="h-20 object-contain"
        />
      ),
      title: 'Nankai University'
    },
  ]

  return (
    <>
      <div className="max-w-7xl mx-auto px-8 py-20">
        <div className="flex flex-col lg:flex-row gap-20">
          {/* 左侧内容区域 */}
          <div className="flex-1">
            <div className="mb-16">
              {/* 主标题 */}
              <div className="text-black mb-8 text-6xl font-bold leading-tight">
                <span className="font-bold text-4xl">Welcome to</span>{' '}
                <span
                  className="font-normal text-6xl"
                  style={{ fontFamily: 'Brush Script MT, cursive' }}
                >
                  Immuneagent
                </span>
              </div>
              {/* 副标题 */}
              <div className="text-gray-600 text-2xl font-normal">
                An AI-powered intelligent antibody research analysis platform
              </div>
            </div>

            {/* 描述文字 */}
            <div className="space-y-6 text-left max-w-3xl">
              <div className="text-black text-xl font-normal leading-relaxed m-0">
                Your AI-powered research and analysis assistant.
              </div>
              <div className="text-black text-xl font-normal leading-relaxed m-0">
                Start interactive conversations and perform complex research
                tasks
              </div>
              <div className="text-black text-xl font-normal leading-relaxed m-0">
                with advanced tool integration.
              </div>
            </div>
          </div>

          {/* 右侧内容区域 */}
          <div className="lg:w-96 space-y-8 gap-4 flex flex-col">
            {/* 功能卡片1 */}
            <Card className="rounded-xl shadow-sm p-8 bg-white min-h-48">
              <div className="space-y-6">案例1展示</div>
            </Card>

            {/* 功能卡片2 */}
            <Card className="rounded-xl shadow-sm p-8 bg-white min-h-48">
              <div className="space-y-6">案例2展示</div>
            </Card>
          </div>
        </div>
      </div>

      {/* 底部功能卡片区域-合作方展示 */}
      <div className="bg-gray-900 py-20 w-screen">
        <div className="max-w-7xl mx-auto px-8">
          <Row gutter={[20, 20]} justify="center">
            {featureCards.map((card, index) => (
              <Col flex="auto" key={index}>
                <Card
                  className="bg-white rounded-xl shadow-sm h-32 flex items-center justify-center hover:shadow-lg transition-all duration-300 cursor-pointer"
                  styles={{
                    body: {
                      padding: '8px',
                      width: '100%',
                      height: '100%',
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }
                  }}
                >
                  {card.icon}
                  {/* {card.title && (
                    <div className="text-sm text-gray-800 font-semibold leading-tight absolute bottom-2 nowrap">
                      {card.title}
                    </div>
                  )} */}
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      </div>
    </>
  )
}

export default Home
