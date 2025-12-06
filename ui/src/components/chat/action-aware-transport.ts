import {
  HttpChatTransport,
  type UIMessage,
  type UIMessageChunk,
} from 'ai'

/**
 * 自定义Transport，用于实时处理action-request事件
 * 直接拦截原始SSE流，实时捕获action请求
 */
export class ActionAwareChatTransport extends HttpChatTransport<UIMessage> {
  private onActionRequest?: (actionData: any) => void
  private onPlanSummary?: (planData: any) => void
  private onPlanConfirmRequest?: (planData: any) => void
  private onExecutionProgress?: (progressData: any) => void
  private onExecutionError?: (errorData: any) => void

  constructor(options: any) {
    super(options)
  }

  setOnActionRequest(callback: (actionData: any) => void) {
    this.onActionRequest = callback
  }

  setOnPlanSummary(callback: (planData: any) => void) {
    this.onPlanSummary = callback
  }

  setOnPlanConfirmRequest(callback: (planData: any) => void) {
    this.onPlanConfirmRequest = callback
  }

  setOnExecutionProgress(callback: (progressData: any) => void) {
    this.onExecutionProgress = callback
  }

  setOnExecutionError(callback: (errorData: any) => void) {
    this.onExecutionError = callback
  }

  // 实现required的processResponseStream方法
  protected processResponseStream(
    stream: ReadableStream<Uint8Array>
  ): ReadableStream<UIMessageChunk> {
    // 使用tee来创建两个流：一个用于action检测，一个用于正常处理
    const [actionStream, normalStream] = stream.tee()

    // 并行处理action流（不阻塞）
    this.processActionStream(actionStream).catch((error) => {
      console.error('ActionAwareChatTransport: 处理action流时出错', error)
    })

    // 处理正常流
    return this.processNormalStream(normalStream)
  }

  private processNormalStream(
    stream: ReadableStream<Uint8Array>
  ): ReadableStream<UIMessageChunk> {
    // 将原始SSE流转换为UIMessageChunk
    const decoder = new TextDecoder()
    const reader = stream.getReader()
    
    return new ReadableStream<UIMessageChunk>({
      async start(controller) {
        let buffer = ''
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              controller.close()
              break
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim()
                if (data === '' || data === '[DONE]') {
                  controller.close()
                  continue
                }

                try {
                  const parsed = JSON.parse(data)
                  
                  // 直接检查是否是action-request事件类型
                  // 如果是action请求，跳过不传递，避免合并到消息中
                  if (
                    parsed.type === 'action-request' ||
                    parsed.type === 'plan-summary' ||
                    parsed.type === 'plan-confirm-request' ||
                    parsed.type === 'execution-progress' ||
                    parsed.type === 'execution-error'
                  ) {
                    // 这是action请求，跳过不传递
                    console.log('ActionAwareChatTransport: 过滤action请求，不传递到消息流', parsed)
                    continue
                  }
                  
                  // 转换为UIMessageChunk格式并传递
                  controller.enqueue(parsed as UIMessageChunk)
                } catch (e) {
                  // 忽略解析错误
                }
              }
            }
          }
        } catch (error) {
          controller.error(error)
        } finally {
          reader.releaseLock()
        }
      },
    })
  }

  // 重写sendMessages来获取原始Response并tee流
  async sendMessages(
    options: Parameters<HttpChatTransport<UIMessage>['sendMessages']>[0]
  ): Promise<ReadableStream<UIMessageChunk>> {
    // 调用父类的sendMessages来获取流，然后tee
    // 但由于我们需要访问原始Response，我们直接调用fetch
    const headersInit = typeof this.headers === 'function' 
      ? await this.headers() 
      : this.headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (headersInit) {
      if (headersInit instanceof Headers) {
        headersInit.forEach((value, key) => {
          headers[key] = value
        })
      } else {
        Object.assign(headers, headersInit)
      }
    }

    const credentialsValue = typeof this.credentials === 'function'
      ? await this.credentials()
      : this.credentials

    // 调用fetch获取原始Response
    const response = await fetch(this.api, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        ...this.body,
        ...options.body,
        messages: options.messages,
      }),
      signal: options.abortSignal,
      credentials: credentialsValue as RequestCredentials | undefined,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    if (response.body == null) {
      throw new Error('Response body is null')
    }

    // 使用tee来创建两个流：一个用于action检测，一个用于正常处理
    const [actionStream, normalStream] = response.body.tee()

    // 并行处理action流（不阻塞）
    this.processActionStream(actionStream).catch((error) => {
      console.error('ActionAwareChatTransport: 处理action流时出错', error)
    })

    // 处理正常流
    return this.processResponseStream(normalStream)
  }

  private async processActionStream(stream: ReadableStream<Uint8Array>): Promise<void> {
    const decoder = new TextDecoder()
    const reader = stream.getReader()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '' || data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              
              // 直接检查是否是action-request事件类型
              // 后端发送的格式是：{"type": "action-request", "actionData": {...}}
              if (parsed.type === 'action-request' && parsed.actionData) {
                console.log('ActionAwareChatTransport: 实时检测到action请求', parsed.actionData)
                
                if (this.onActionRequest) {
                  // 同步调用回调，确保立即触发
                  console.log('ActionAwareChatTransport: 调用onActionRequest回调', parsed.actionData)
                  try {
                    this.onActionRequest(parsed.actionData)
                  } catch (callbackError) {
                    console.error('ActionAwareChatTransport: 回调执行错误', callbackError)
                  }
                } else {
                  console.warn('ActionAwareChatTransport: onActionRequest回调未设置')
                }
                continue
              } else if (parsed.type === 'plan-summary') {
                try {
                  this.onPlanSummary?.(parsed.plan ?? parsed)
                } catch (planError) {
                  console.error('ActionAwareChatTransport: 计划摘要回调执行错误', planError)
                }
              } else if (parsed.type === 'plan-confirm-request') {
                try {
                  this.onPlanConfirmRequest?.(parsed.plan ?? parsed)
                } catch (confirmError) {
                  console.error('ActionAwareChatTransport: 计划确认回调执行错误', confirmError)
                }
              } else if (parsed.type === 'execution-progress') {
                try {
                  this.onExecutionProgress?.(parsed.progress ?? parsed)
                } catch (progressError) {
                  console.error('ActionAwareChatTransport: 执行进度回调执行错误', progressError)
                }
              } else if (parsed.type === 'execution-error') {
                try {
                  this.onExecutionError?.(parsed.error ?? parsed)
                } catch (errorCallback) {
                  console.error('ActionAwareChatTransport: 执行错误回调执行错误', errorCallback)
                }
              }
            } catch (e) {
              // 忽略解析错误，继续处理下一个事件
            }
          }
        }
      }
    } catch (error) {
      console.error('ActionAwareChatTransport: 处理action流时出错', error)
    } finally {
      reader.releaseLock()
    }
  }
}