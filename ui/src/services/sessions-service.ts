// Sessions service for managing session data
// This service handles all session-related data operations

import apiClient from './api-client'
import type { AxiosResponse } from 'axios'

// Session interface matching the database model
export interface Session {
  id: string
  user_id: string
  usecase: string
  configuration: string
  name?: string
  created_at?: string
  updated_at?: string
  last_message?: string
  last_message_time?: string
}

export interface UsecaseInfo {
  name: string
  description: string
  configuration?: string | Record<string, any>
}

export interface ChatHistoryResponse {
  session_id: string
  message: string
  timestamp: number
  role: string
}

/**
 * Get all sessions
 * @returns Promise<Session[]> List of all sessions
 */
export const getSessions = async (): Promise<Session[]> => {
  const response: AxiosResponse<Session[]> = await apiClient.get('/sessions')
  return response.data
}

/**
 * Get a specific session by ID
 * @param id - The ID of the session to retrieve
 * @returns Promise<Session> The session if found
 */
export const getSessionById = async (id: string): Promise<Session> => {
  const response: AxiosResponse<Session> = await apiClient.get(
    `/sessions/${id}`,
  )
  return response.data
}

/**
 * Create a new session
 * @param usecase - The usecase for the session
 * @param configuration - Optional configuration string (defaults to '{}')
 * @returns Promise<Session> The newly created session
 */
export const createSession = async (
  usecase: string,
  configuration?: string,
): Promise<Session> => {
  const sessionData = {
    usecase,
    configuration: configuration || '{}',
  }

  const response: AxiosResponse<Session> = await apiClient.post(
    '/sessions',
    sessionData,
  )
  return response.data
}

/**
 * Update a session
 * @param sessionId - The ID of the session to update
 * @param configuration - Configuration object of the Session (optional)
 * @param name - Name of the session (optional)
 * @returns Promise<Session> The updated session
 */
export const updateSession = async (
  sessionId: string,
  options: {
    configuration?: Record<string, any>
    name?: string
  },
): Promise<Session> => {
  const response: AxiosResponse<Session> = await apiClient.put(
    `/sessions/${sessionId}`,
    options,
  )
  return response.data
}

/**
 * Delete a session
 * @param id - The ID of the session to delete
 * @returns Promise<void>
 */
export const deleteSession = async (id: string): Promise<void> => {
  await apiClient.delete(`/sessions/${id}`)
}

/**
 * Get chat history for a session
 * @param sessionId - The ID of the session to retrieve chat history for
 * @returns Promise<ChatHistoryResponse[]> List of chat history items
 */
export const getChatHistory = async (
  sessionId: string,
): Promise<ChatHistoryResponse[]> => {
  const response: AxiosResponse<ChatHistoryResponse[]> = await apiClient.get(
    `/sessions/${sessionId}/chat-history`,
  )
  return response.data
}

/**
 * Get all available usecases
 * @returns Promise<UsecaseInfo[]> List of available usecases
 */
export const getUsecases = async (): Promise<UsecaseInfo[]> => {
  const response: AxiosResponse<UsecaseInfo[]> =
    await apiClient.get('/usecases')
  return response.data
}
