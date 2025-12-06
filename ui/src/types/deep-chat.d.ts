// Type definitions for deep-chat
declare module 'deep-chat' {
  interface DeepChatProps {
    directConnection?: any
    style?: React.CSSProperties
    initialMessages?: Array<{
      role: 'user' | 'ai'
      text: string
    }>
    messageStyles?: any
    textInput?: any
    submitButtonStyles?: any
    ref?: React.Ref<any>
  }

  const DeepChat: React.ComponentType<DeepChatProps>
  export default DeepChat
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'deep-chat': React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement>,
        HTMLElement
      > & {
        ref?: React.Ref<any>
        directConnection?: any
        style?: React.CSSProperties
        initialMessages?: Array<{
          role: 'user' | 'ai'
          text: string
        }>
        messageStyles?: any
        textInput?: any
        submitButtonStyles?: any
      }
    }
  }
}
