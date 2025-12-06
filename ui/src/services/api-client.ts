// API client configuration with axios and error handling interceptors
import axios, { type AxiosError, type AxiosResponse } from 'axios'
import { notification } from 'antd'

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth tokens or other headers
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    } else {
      console.log('No token found in localStorage')
    }
    
    // 如果是FormData，删除Content-Type让axios自动设置
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type']
    }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  },
)

// Response interceptor for handling errors elegantly
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  (error: AxiosError) => {
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response

      switch (status) {
        case 400:
          notification.error({
            message: 'Bad Request',
            description: 'Please check your input.',
          })
          break
        case 401:
          notification.error({
            message: 'Unauthorized',
            description: 'Please login again.',
          })
          // Redirect to auth page or clear auth token
          localStorage.removeItem('auth_token')
          window.location.href = '/auth'
          break
        case 403:
          notification.error({
            message: 'Access Forbidden',
            description: 'You do not have permission.',
          })
          break
        case 404:
          // Don't show notification for session endpoints as 404 is expected behavior
          if (!error.config?.url?.includes('/sessions/')) {
            notification.error({
              message: 'Not Found',
              description: 'Resource not found.',
            })
          }
          break
        case 422:
          // Validation errors{}
          notification.error({
            message: 'Validation Error',
            description: (data as any)?.detail || 'Validation error',
          })
          break
        case 500:
          notification.error({
            message: 'Server Error',
            description: 'Internal server error. Please try again later.',
          })
          break
        case 502:
        case 503:
        case 504:
          notification.error({
            message: 'Service Unavailable',
            description:
              'Service temporarily unavailable. Please try again later.',
          })
          break
        default:
          notification.error({
            message: 'Request Failed',
            description: `Request failed with status ${status}`,
          })
      }
    } else if (error.request) {
      // Network error or no response
      notification.error({
        message: 'Network Error',
        description:
          'Please check your connection and ensure the backend server is running.',
      })
    } else {
      // Something else happened
      notification.error({
        message: 'Unexpected Error',
        description: 'An unexpected error occurred.',
      })
    }

    return Promise.reject(error)
  },
)

export default apiClient

// Type definitions for common API responses
export interface ApiResponse<T = any> {
  data: T
  message?: string
  success?: boolean
}

export interface ApiError {
  detail: string
  status_code: number
}
