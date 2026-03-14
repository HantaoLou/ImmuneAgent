'use client';

import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * 全局错误边界组件
 * 捕获 React 组件树中的 JavaScript 错误，记录错误并显示备用 UI
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // 更新 state，下一次渲染显示备用 UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // 记录错误到控制台
    console.error('ErrorBoundary 捕获到错误:', error);
    console.error('错误组件栈:', errorInfo.componentStack);
    
    // TODO: 可以在这里上报错误到监控系统
    // reportError(error, errorInfo);
  }

  handleReset = () => {
    // 刷新页面来重置错误状态
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      // 如果提供了自定义的 fallback，使用它
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // 默认的错误 UI
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          padding: '2rem',
          background: 'linear-gradient(to bottom, #f7f8fa, #e8eef3)',
        }}>
          <div style={{
            maxWidth: '500px',
            padding: '2rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            textAlign: 'center',
          }}>
            <div style={{
              fontSize: '4rem',
              marginBottom: '1rem',
            }}>
              😔
            </div>
            <h2 style={{
              fontSize: '1.5rem',
              color: '#1f2937',
              marginBottom: '1rem',
            }}>
              出现了一些问题
            </h2>
            <p style={{
              color: '#6b7280',
              marginBottom: '1.5rem',
              lineHeight: '1.6',
            }}>
              应用程序遇到了一个错误。请尝试刷新页面，如果问题持续存在，请联系技术支持。
            </p>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details style={{
                marginBottom: '1.5rem',
                textAlign: 'left',
                padding: '1rem',
                background: '#f3f4f6',
                borderRadius: '4px',
              }}>
                <summary style={{
                  cursor: 'pointer',
                  color: '#374151',
                  fontWeight: '500',
                }}>
                  错误详情（仅开发环境可见）
                </summary>
                <pre style={{
                  marginTop: '0.5rem',
                  fontSize: '0.875rem',
                  color: '#ef4444',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {this.state.error.toString()}
                </pre>
              </details>
            )}
            <button
              onClick={this.handleReset}
              style={{
                padding: '0.75rem 2rem',
                fontSize: '1rem',
                color: 'white',
                background: '#3b82f6',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'background 0.2s',
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = '#2563eb';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = '#3b82f6';
              }}
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
