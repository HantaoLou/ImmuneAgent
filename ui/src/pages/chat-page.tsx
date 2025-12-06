import React from 'react'
import { ChatInterface } from '../components/chat/chat-interface-component'
import { PageContent } from '../components/common/page-content'

interface ChatProps {
  sessionId: string
}

const Chat: React.FC<ChatProps> = ({ sessionId }) => {
  return (
    <PageContent>
      <ChatInterface sessionId={sessionId} />
    </PageContent>
  )
}

export default Chat
