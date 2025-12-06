import apiClient from './api-client'

export interface SessionArtifact {
  id: string
  session_id: string
  file_name: string
  original_file_name: string
  file_size: number
  mime_type: string
  description: string
  created_at: string
  updated_at: string
  download_url?: string
  oss_direct_url?: string  // OSS直接下载URL（预签名URL或公共URL）
}

export interface ArtifactsResponse {
  data: SessionArtifact[]
  message?: string
  success?: boolean
}

export const getSessionArtifacts = async (
  sessionId: string,
): Promise<SessionArtifact[]> => {
  try {
    const response = await apiClient.get<SessionArtifact[]>(
      `/sessions/${sessionId}/artifacts`,
    )
    return response.data
  } catch (error) {
    console.error('Failed to fetch session artifacts:', error)
    throw error
  }
}

export const downloadArtifact = async (
  artifact: SessionArtifact | string,
): Promise<void> => {
  try {
    // 如果传入的是artifact对象且有OSS直接URL，优先使用OSS直接下载
    if (typeof artifact === 'object' && artifact.oss_direct_url) {
      // 使用OSS直接URL下载
      const response = await fetch(artifact.oss_direct_url)
      if (!response.ok) {
        throw new Error(`Failed to download from OSS: ${response.statusText}`)
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = artifact.original_file_name || 'artifact'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
      return
    }
    
    // 否则使用后端API下载（兼容旧数据或OSS未启用的情况）
    const artifactId = typeof artifact === 'string' ? artifact : artifact.id
    const response = await apiClient.get(
      `/sessions/artifacts/${artifactId}/download`,
      {
        responseType: 'blob',
      },
    )

    // Create a blob URL and trigger download
    const blob = new Blob([response.data])
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download =
      response.headers['content-disposition']?.split('filename=')[1] ||
      'artifact'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('Failed to download artifact:', error)
    throw error
  }
}

export const deleteArtifact = async (artifactId: string): Promise<void> => {
  try {
    await apiClient.delete(`/sessions/artifacts/${artifactId}`)
  } catch (error) {
    console.error('Failed to delete artifact:', error)
    throw error
  }
}

export interface UploadArtifactResponse extends SessionArtifact {
  download_url: string
  oss_direct_url?: string  // OSS直接下载URL（如果启用OSS）
}

/**
 * 获取OSS预签名上传URL和下载URL
 */
export const getUploadUrl = async (
  sessionId: string,
  fileName: string,
  contentType?: string,
): Promise<{ upload_url: string; download_url: string; object_key: string; expires_in: number }> => {
  try {
    const params = new URLSearchParams({ file_name: fileName })
    if (contentType) {
      params.append('content_type', contentType)
    }
    
    const response = await apiClient.get<{
      upload_url: string
      download_url: string
      object_key: string
      expires_in: number
    }>(`/sessions/${sessionId}/artifacts/upload-url?${params.toString()}`)
    
    return response.data
  } catch (error) {
    console.error('Failed to get upload URL:', error)
    throw error
  }
}

/**
 * 确认文件已上传到OSS，创建artifact记录
 */
export const confirmUpload = async (
  sessionId: string,
  fileName: string,
  originalFileName: string,
  fileSize: number,
  mimeType: string,
  description?: string,
): Promise<UploadArtifactResponse> => {
  try {
    const formData = new FormData()
    formData.append('file_name', fileName)
    formData.append('original_file_name', originalFileName)
    formData.append('file_size', fileSize.toString())
    formData.append('mime_type', mimeType)
    if (description) {
      formData.append('description', description)
    }

    const response = await apiClient.post<UploadArtifactResponse>(
      `/sessions/${sessionId}/artifacts/confirm-upload`,
      formData,
    )

    return response.data
  } catch (error) {
    console.error('Failed to confirm upload:', error)
    throw error
  }
}

// 分片上传阈值：100MB
const MULTIPART_UPLOAD_THRESHOLD = 100 * 1024 * 1024 // 100MB
const CHUNK_SIZE = 10 * 1024 * 1024 // 10MB per chunk

/**
 * 分片上传文件到OSS
 */
/**
 * 上传单个分片（带重试机制）
 */
const uploadPartWithRetry = async (
  sessionId: string,
  file: File,
  uploadId: string,
  chunkIndex: number,
  maxRetries: number = 3,
): Promise<{ part_number: number; etag: string }> => {
  const start = chunkIndex * CHUNK_SIZE
  const end = Math.min(start + CHUNK_SIZE, file.size)
  const chunk = file.slice(start, end)
  
  let lastError: Error | null = null
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const partFormData = new FormData()
      partFormData.append('file_name', file.name)
      partFormData.append('upload_id', uploadId)
      partFormData.append('part_number', (chunkIndex + 1).toString())
      partFormData.append('file', chunk)
      
      const partResponse = await apiClient.post<{
        etag: string
        part_number: number
      }>(`/sessions/${sessionId}/artifacts/upload-part`, partFormData, {
        timeout: 5 * 60 * 1000, // 5分钟超时（axios配置）
      } as any)
      
      return {
        part_number: partResponse.data.part_number,
        etag: partResponse.data.etag,
      }
    } catch (error: any) {
      lastError = error
      
      // 检查是否是网络错误（可重试）
      if (
        error.message?.includes('timeout') ||
        error.message?.includes('network') ||
        error.message?.includes('ECONNRESET') ||
        error.message?.includes('Failed to fetch') ||
        (error.response?.status >= 500 && error.response?.status < 600)
      ) {
        if (attempt < maxRetries) {
          console.warn(`分片 ${chunkIndex + 1} 上传失败（尝试 ${attempt}/${maxRetries}），将在 ${attempt * 2} 秒后重试...`, error.message)
          await new Promise(resolve => setTimeout(resolve, attempt * 2000)) // 指数退避
          continue
        }
      }
      
      // 非网络错误或已达到最大重试次数，直接抛出
      throw error
    }
  }
  
  throw lastError || new Error(`分片 ${chunkIndex + 1} 上传失败：未知错误`)
}

/**
 * 获取已上传的分片列表（用于续传）
 */
const getUploadedParts = async (
  sessionId: string,
  fileName: string,
  uploadId: string,
): Promise<Array<{ part_number: number; etag: string; size: number }>> => {
  try {
    const response = await apiClient.get<{
      parts: Array<{ part_number: number; etag: string; size: number }>
      upload_id: string
    }>(`/sessions/${sessionId}/artifacts/list-uploaded-parts`, {
      params: {
        file_name: fileName,
        upload_id: uploadId,
      },
    })
    return response.data.parts || []
  } catch (error) {
    console.warn('获取已上传分片列表失败，将从头开始上传:', error)
    return []
  }
}

const uploadArtifactMultipart = async (
  sessionId: string,
  file: File,
  description?: string,
  onProgress?: (progress: number, info?: { currentChunk: number; totalChunks: number; uploadedBytes: number }) => void,
  maxRetries: number = 3,
): Promise<UploadArtifactResponse> => {
  // 1. 初始化分片上传
  const initFormData = new FormData()
  initFormData.append('file_name', file.name)
  initFormData.append('content_type', file.type || 'application/octet-stream')
  
  const initResponse = await apiClient.post<{
    upload_id: string
    download_url: string
    object_key: string
  }>(`/sessions/${sessionId}/artifacts/initiate-multipart-upload`, initFormData)
  
  const { upload_id, download_url } = initResponse.data
  
  // 2. 计算分片数量
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
  
  // 3. 获取已上传的分片列表（续传支持）
  const uploadedPartsMap = new Map<number, { part_number: number; etag: string; size: number }>()
  try {
    const uploadedParts = await getUploadedParts(sessionId, file.name, upload_id)
    uploadedParts.forEach(part => {
      uploadedPartsMap.set(part.part_number, part)
    })
    if (uploadedParts.length > 0) {
      console.log(`发现 ${uploadedParts.length} 个已上传的分片，将从断点继续上传`)
    }
  } catch (error) {
    console.warn('获取已上传分片失败，将从头开始:', error)
  }
  
  const parts: Array<{ part_number: number; etag: string }> = []
  
  try {
    // 4. 上传每个分片（跳过已上传的分片）
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const partNumber = chunkIndex + 1
      
      // 检查分片是否已上传
      const existingPart = uploadedPartsMap.get(partNumber)
      if (existingPart) {
        // 使用已上传的分片信息
        parts.push({
          part_number: existingPart.part_number,
          etag: existingPart.etag,
        })
        
        console.log(`分片 ${partNumber} 已存在，跳过上传`)
        
        // 更新进度
        if (onProgress) {
          const actualUploadedBytes = Math.min((chunkIndex + 1) * CHUNK_SIZE, file.size)
          const chunkProgress = (actualUploadedBytes / file.size) * 90
          onProgress(
            chunkProgress,
            {
              currentChunk: chunkIndex + 1,
              totalChunks,
              uploadedBytes: actualUploadedBytes,
            }
          )
        }
        continue
      }
      
      // 上传新分片（带重试机制）
      try {
        const part = await uploadPartWithRetry(sessionId, file, upload_id, chunkIndex, maxRetries)
        parts.push(part)
        
        // 更新进度 - 基于实际已上传的字节数计算
        if (onProgress) {
          const actualUploadedBytes = Math.min((chunkIndex + 1) * CHUNK_SIZE, file.size)
          const chunkProgress = (actualUploadedBytes / file.size) * 90
          onProgress(
            chunkProgress,
            {
              currentChunk: chunkIndex + 1,
              totalChunks,
              uploadedBytes: actualUploadedBytes,
            }
          )
        }
      } catch (partError: any) {
        // 单个分片上传失败，抛出错误以便外层处理
        console.error(`分片 ${partNumber} 上传失败:`, partError)
        throw new Error(`分片 ${partNumber} 上传失败: ${partError.message || '未知错误'}`)
      }
    }
    
    // 5. 完成分片上传
    const completeFormData = new FormData()
    completeFormData.append('file_name', file.name)
    completeFormData.append('upload_id', upload_id)
    completeFormData.append('parts', JSON.stringify(parts))
    completeFormData.append('original_file_name', file.name)
    completeFormData.append('file_size', file.size.toString())
    completeFormData.append('mime_type', file.type || 'application/octet-stream')
    if (description) {
      completeFormData.append('description', description)
    }
    
    // 更新进度到95%
    if (onProgress) {
      onProgress(95, {
        currentChunk: totalChunks,
        totalChunks,
        uploadedBytes: file.size,
      })
    }
    
    const artifact = await apiClient.post<UploadArtifactResponse>(
      `/sessions/${sessionId}/artifacts/complete-multipart-upload`,
      completeFormData,
    )
    
    // 上传完成（更新进度到100%）
    if (onProgress) {
      onProgress(100, {
        currentChunk: totalChunks,
        totalChunks,
        uploadedBytes: file.size,
      })
    }
    
    // 使用初始化时返回的download_url
    return {
      ...artifact.data,
      oss_direct_url: download_url,
    }
  } catch (error) {
    // 如果上传失败，不取消分片上传，以便后续续传
    // 只有用户明确取消时才调用abort
    console.error('分片上传失败，已上传的分片将保留以便续传:', error)
    throw error
  }
}

/**
 * 上传文件到OSS（使用预签名URL直传）
 * 包含重试机制和错误处理
 * 大文件（>100MB）自动使用分片上传
 */
export const uploadArtifactToOSS = async (
  sessionId: string,
  file: File,
  description?: string,
  maxRetries: number = 3,
  onProgress?: (progress: number, info?: { currentChunk?: number; totalChunks?: number; uploadedBytes?: number }) => void,
): Promise<UploadArtifactResponse> => {
  // 大文件使用分片上传
  if (file.size >= MULTIPART_UPLOAD_THRESHOLD) {
    console.log(`文件大小 ${(file.size / 1024 / 1024).toFixed(2)}MB，使用分片上传`)
    return await uploadArtifactMultipart(sessionId, file, description, onProgress, maxRetries)
  }
  
  // 小文件使用普通上传（带重试机制）
  let lastError: Error | null = null

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // 1. 获取预签名上传URL和下载URL
      const { upload_url, download_url, object_key } = await getUploadUrl(
        sessionId,
        file.name,
        file.type || 'application/octet-stream',
      )

      // 2. 直接上传文件到OSS（使用XMLHttpRequest以支持进度跟踪）
      try {
        const result = await new Promise<UploadArtifactResponse>((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          let timeoutId: ReturnType<typeof setTimeout> | null = null

          // 设置超时
          timeoutId = setTimeout(() => {
            xhr.abort()
            reject(new Error('Upload timeout after 5 minutes'))
          }, 5 * 60 * 1000) // 5分钟超时

          // 监听上传进度
          xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable && onProgress) {
              const percent = (event.loaded / event.total) * 80 // 上传占80%进度
              onProgress(percent, { uploadedBytes: event.loaded })
            }
          })

          // 监听完成
          xhr.addEventListener('load', async () => {
            if (timeoutId) {
              clearTimeout(timeoutId)
            }

            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                // 更新进度到90%（上传完成，等待确认）
                if (onProgress) {
                  onProgress(90, { uploadedBytes: file.size })
                }

                // 3. 确认上传，创建artifact记录（更新进度到95%）
                if (onProgress) {
                  onProgress(95, { uploadedBytes: file.size })
                }
                
                const artifact = await confirmUpload(
                  sessionId,
                  object_key.split('/').pop() || file.name,
                  file.name,
                  file.size,
                  file.type || 'application/octet-stream',
                  description,
                )

                // 4. 上传完成（更新进度到100%）
                if (onProgress) {
                  onProgress(100, { uploadedBytes: file.size })
                }

                // 5. 使用upload-url接口返回的download_url（8小时有效期）
                resolve({
                  ...artifact,
                  oss_direct_url: download_url,
                })
              } catch (error) {
                reject(error)
              }
            } else {
              reject(new Error(`Failed to upload to OSS: ${xhr.status} ${xhr.statusText}`))
            }
          })

          // 监听错误
          xhr.addEventListener('error', () => {
            if (timeoutId) {
              clearTimeout(timeoutId)
            }
            reject(new Error('Network error during upload'))
          })

          // 监听取消
          xhr.addEventListener('abort', () => {
            if (timeoutId) {
              clearTimeout(timeoutId)
            }
            reject(new Error('Upload aborted'))
          })

          // 开始上传
          xhr.open('PUT', upload_url)
          xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream')
          xhr.send(file)
        })

        // 上传成功，返回结果
        return result
      } catch (uploadError: any) {
        // 检查是否是网络错误（可重试）
        if (
          uploadError.message?.includes('timeout') ||
          uploadError.message?.includes('Network error') ||
          uploadError.message?.includes('aborted')
        ) {
          lastError = uploadError
          if (attempt < maxRetries) {
            console.warn(`上传失败（尝试 ${attempt}/${maxRetries}），将在 ${attempt * 2} 秒后重试...`, uploadError.message)
            await new Promise(resolve => setTimeout(resolve, attempt * 2000)) // 指数退避
            continue
          }
        }
        throw uploadError
      }
    } catch (error: any) {
      lastError = error
      
      // 如果是非网络错误（如认证错误、文件不存在等），不重试
      if (
        error.message?.includes('401') ||
        error.message?.includes('403') ||
        error.message?.includes('404') ||
        error.message?.includes('Invalid session')
      ) {
        console.error('上传失败（不可重试的错误）:', error)
        throw error
      }

      // 如果是最后一次尝试，抛出错误
      if (attempt === maxRetries) {
        console.error(`上传失败（已重试 ${maxRetries} 次）:`, error)
        throw new Error(
          `文件上传失败: ${error.message || '网络连接错误'}。请检查网络连接后重试。`
        )
      }
    }
  }

  // 理论上不会到达这里，但为了类型安全
  throw lastError || new Error('上传失败：未知错误')
}

/**
 * 上传文件到OSS（直接上传，不使用后端代理）
 * 大文件（>100MB）自动使用分片上传
 */
export const uploadArtifact = async (
  sessionId: string,
  file: File,
  description?: string,
  onProgress?: (progress: number, info?: { currentChunk?: number; totalChunks?: number; uploadedBytes?: number }) => void,
): Promise<UploadArtifactResponse> => {
  return await uploadArtifactToOSS(sessionId, file, description, 3, onProgress)
}
