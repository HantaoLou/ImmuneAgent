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
 * Global Error Boundary Component
 * Catches JavaScript errors in React component tree, logs errors and displays fallback UI
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state to show fallback UI on next render
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to console
    console.error('ErrorBoundary caught error:', error);
    console.error('Error component stack:', errorInfo.componentStack);
    
    // TODO: Report error to monitoring system here
    // reportError(error, errorInfo);
  }

  handleReset = () => {
    // Refresh page to reset error state
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      // If custom fallback is provided, use it
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
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
              Something went wrong
            </h2>
            <p style={{
              color: '#6b7280',
              marginBottom: '1.5rem',
              lineHeight: '1.6',
            }}>
              The application encountered an error. Please try refreshing the page. If the problem persists, please contact technical support.
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
                  Error details (visible in development only)
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
                Refresh Page
              </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
