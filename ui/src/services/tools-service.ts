export interface ServiceInfo {
  id: string
  transport?: string
  url?: string
  timeout?: number
  about?: string
  host?: string
}

export async function listServices(): Promise<ServiceInfo[]> {
  const res = await fetch('/api/tools/services', {
    headers: { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` },
  })
  const data = await res.json()
  return data.services || []
}

export interface ToolInfo {
  name: string
  description?: string
  args_schema?: any
}

export async function listTools(serviceId: string, signal?: AbortSignal): Promise<ToolInfo[]> {
  const res = await fetch(`/api/tools/list?service_id=${encodeURIComponent(serviceId)}`, {
    headers: { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` },
    signal,
  })
  const rawBody = await res.text()

  if (!res.ok) {
    const snippet = rawBody.slice(0, 200)
    throw new Error(
      `Failed to load tools for service ${serviceId}: ${res.status} ${res.statusText}${snippet ? ` - ${snippet}` : ''}`,
    )
  }

  try {
    const data = rawBody ? JSON.parse(rawBody) : {}
    return data.tools || []
  } catch (error) {
    const snippet = rawBody.slice(0, 200)
    throw new Error(
      `Unexpected response format when loading tools for service ${serviceId}: ${snippet}`,
    )
  }
}

export async function invokeTool(serviceId: string, toolName: string, params: any): Promise<any> {
  const res = await fetch('/api/tools/invoke', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
    },
    body: JSON.stringify({ service_id: serviceId, tool_name: toolName, params }),
  })
  const data = await res.json()
  return data.result
}


