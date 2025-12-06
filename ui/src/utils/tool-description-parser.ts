/**
 * 工具描述解析器
 * 从工具的 description 中提取结构化信息
 */

export interface ParsedParameterDescription {
  name: string
  description: string
  tags: string[]
  isOptional: boolean
  defaultValue?: string
  examples?: string[]
  range?: string
}

export interface ParsedToolDescription {
  summary: string
  parameters: Map<string, ParsedParameterDescription>
  returns?: string
  notes?: string[]
}

/**
 * 解析工具描述
 * @param description 工具的完整描述文本
 * @returns 解析后的结构化信息
 */
export function parseToolDescription(description: string): ParsedToolDescription {
  const result: ParsedToolDescription = {
    summary: '',
    parameters: new Map(),
    returns: undefined,
    notes: [],
  }

  if (!description) {
    return result
  }

  // 分割描述为不同部分
  const lines = description.split('\n').map(line => line.trim())
  
  let currentSection = 'summary'
  let currentParamName: string | null = null
  let currentParamDesc: string[] = []
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // 检测 Args: 部分
    if (line.match(/^Args?:$/i)) {
      currentSection = 'args'
      continue
    }
    
    // 检测 Returns: 部分
    if (line.match(/^Returns?:$/i)) {
      // 保存之前的参数
      if (currentParamName) {
        saveParameter(result.parameters, currentParamName, currentParamDesc.join(' '))
      }
      currentSection = 'returns'
      currentParamName = null
      currentParamDesc = []
      continue
    }
    
    // 检测其他部分（如 Notes:, Examples: 等）
    if (line.match(/^(Notes?|Examples?|Warning|Note):$/i)) {
      if (currentParamName) {
        saveParameter(result.parameters, currentParamName, currentParamDesc.join(' '))
      }
      currentSection = line.toLowerCase().replace(':', '')
      currentParamName = null
      currentParamDesc = []
      continue
    }
    
    // 在 Args 部分，检测参数定义行
    if (currentSection === 'args') {
      // 参数定义行格式: "param_name: description" 或 "param_name (type): description"
      // 支持多种格式：
      // - "param_name: description"
      // - "    param_name: @tag@ description" (带缩进)
      // - "param_name (type): description"
      // - "param_name: @tag@ description"
      // - "param_name: [example] - description" (带示例)
      // - "param_name: value1, value2, value3" (枚举值列表)
      const paramMatch = line.match(/^\s*(\w+)(?:\s*\([^)]+\))?\s*[:：]\s*(.+)$/)
      
      if (paramMatch) {
        // 保存之前的参数
        if (currentParamName) {
          saveParameter(result.parameters, currentParamName, currentParamDesc.join(' '))
        }
        
        // 开始新参数
        currentParamName = paramMatch[1]
        const paramDescLine = paramMatch[2].trim()
        // 检查是否包含示例（如 [{"id": "seq1", ...}] - 描述）
        if (paramDescLine.includes(' - ') || paramDescLine.includes(' – ') || paramDescLine.includes(' — ')) {
          // 包含破折号，可能是"示例 - 描述"格式
          currentParamDesc = [paramDescLine]
        } else {
          // 普通格式
          currentParamDesc = [paramDescLine]
        }
      } else if (currentParamName && line && !line.match(/^[A-Z][a-z]+:/)) {
        // 参数描述的续行（但不是新的部分标题）
        // 检查是否是描述的一部分（有缩进或是普通文本）
        const trimmedLine = line.trim()
        if (trimmedLine.length > 0) {
          // 如果行首有缩进，或者是普通文本（不是大写字母开头的独立单词），则是续行
          if (line.startsWith(' ') || line.startsWith('\t') || 
              !trimmedLine.match(/^[A-Z][a-z]*\s*:/) || 
              trimmedLine.length < 20) {
            currentParamDesc.push(trimmedLine)
          } else {
            // 可能是新参数，但格式不标准，先保存当前参数
            saveParameter(result.parameters, currentParamName, currentParamDesc.join(' '))
            currentParamName = null
            currentParamDesc = []
          }
        }
      }
    }
    // 在 Returns 部分
    else if (currentSection === 'returns') {
      if (!result.returns) {
        result.returns = line
      } else {
        result.returns += ' ' + line
      }
    }
    // 在 summary 部分
    else if (currentSection === 'summary') {
      if (result.summary) {
        result.summary += ' ' + line
      } else {
        result.summary = line
      }
    }
    // 其他部分（Notes, Examples 等）
    else {
      if (line && !line.match(/^[A-Z][a-z]+:$/)) {
        if (!result.notes) {
          result.notes = []
        }
        result.notes.push(line)
      }
    }
  }
  
  // 保存最后一个参数
  if (currentParamName) {
    saveParameter(result.parameters, currentParamName, currentParamDesc.join(' '))
  }
  
  // 清理 summary
  result.summary = result.summary.trim()
  if (result.returns) {
    result.returns = result.returns.trim()
  }
  
  return result
}

/**
 * 保存参数信息
 */
function saveParameter(
  parameters: Map<string, ParsedParameterDescription>,
  paramName: string,
  description: string
) {
  if (!description) return
  
  // 提取标签
  const tagPattern = /@(\w+)@/g
  const tags: string[] = []
  let cleanDesc = description.replace(tagPattern, (_match, tag) => {
    tags.push(tag)
    return ''
  }).trim()
  
  // 提取可选性信息
  const optionalPatterns = [
    /(?:^|\.)\s*(?:Optional|optional|可选)\s*[.:]/i,
    /(?:^|\.)\s*If\s+not\s+provided/i,
    /(?:^|\.)\s*If\s+(?:not\s+)?specified/i,
    /(?:^|\.)\s*Defaults?\s+to/i,
    /(?:^|\.)\s*When\s+not\s+provided/i,
  ]
  const isOptional = optionalPatterns.some(pattern => pattern.test(cleanDesc))
  
  // 提取默认值（多种格式）
  const defaultPatterns = [
    /Defaults?\s+to\s+([^,\.]+)/i,
    /Default\s+is\s+([^,\.]+)/i,
    /Default:\s*([^,\.]+)/i,
    /默认值[：:]\s*([^,\.]+)/i,
  ]
  let defaultValue: string | undefined
  for (const pattern of defaultPatterns) {
    const match = cleanDesc.match(pattern)
    if (match && match[1]) {
      defaultValue = match[1].trim()
      break
    }
  }
  
  // 提取示例（多种格式）
  const examplePatterns = [
    /(?:e\.g\.|example|示例|例如)[：:\s]+([^,\.\n]+)/gi,
    /(?:such\s+as|like|比如)\s+([^,\.\n]+)/gi,
    /(?:for\s+example|例如)[：:\s]+([^,\.\n]+)/gi,
  ]
  const examples: string[] = []
  for (const pattern of examplePatterns) {
    const matches = cleanDesc.matchAll(pattern)
    for (const match of matches) {
      if (match[1]) {
        const example = match[1].trim().replace(/^["']|["']$/g, '') // 移除引号
        if (example && !examples.includes(example)) {
          examples.push(example)
        }
      }
    }
  }
  
  // 提取参数描述中的内联示例和枚举值
  // 格式1: sequences: [{"id": "seq1", "sequence": "ATGC..."}] - NUCLEOTIDE sequences!
  // 匹配格式：[...] - 或 {...} - （支持多行JSON）
  let hasInlineExample = false
  // 先尝试匹配简单的格式（非贪婪匹配，支持嵌套的JSON结构）
  // 匹配以 [ 或 { 开头，以 ] 或 } 结尾，后跟破折号的内容
  const jsonExamplePattern = /^(\[.*?\]|\{.*?\})\s*[-–—]\s*(.+)$/s
  const simpleMatch = cleanDesc.match(jsonExamplePattern)
  if (simpleMatch) {
    // 提取示例部分（破折号前的内容）
    const exampleValue = simpleMatch[1].trim()
    // 验证是否是有效的JSON格式（简单检查）
    try {
      JSON.parse(exampleValue)
      // 如果是有效的JSON，添加到示例中
      if (!examples.includes(exampleValue)) {
        examples.unshift(exampleValue) // 添加到开头，优先显示
      }
      // 使用破折号后的部分作为描述
      cleanDesc = simpleMatch[2].trim()
      hasInlineExample = true
    } catch {
      // 不是有效的JSON，可能是其他格式，继续处理
    }
  }
  
  // 格式2: organism: human, mouse, rabbit, rat, rhesus, pig (纯枚举值列表)
  // 检查是否是纯枚举值列表格式（没有破折号，直接是逗号分隔的值）
  if (!hasInlineExample) {
    // 检查是否像枚举值列表（逗号分隔的多个值，且每个值都是简单单词或大写字母组合）
    // 匹配格式：word1, word2, word3 或 IGH, IGK, IGL, TRA, TRB, TRG, TRD
    // 注意：枚举值应该是连续的，没有其他描述性文本
    const enumPattern = /^([A-Z_]+(?:\s*,\s*[A-Z_]+)+|[a-z_]+(?:\s*,\s*[a-z_]+)+)\s*$/i
    const enumMatch = cleanDesc.match(enumPattern)
    if (enumMatch) {
      const enumValues = enumMatch[1].split(',').map(v => v.trim()).filter(v => v)
      if (enumValues.length >= 2) {
        // 看起来像枚举值列表，添加到示例中
        examples.push(...enumValues)
        // 清空描述（因为这只是枚举值列表，没有其他描述）
        cleanDesc = ''
      }
    } else {
      // 格式3: 混合格式（枚举值 + 描述，如 "human, mouse, rabbit - description"）
      // 或者 "human, mouse, rabbit" 后面还有其他文本
      if (cleanDesc.includes(',')) {
        // 尝试匹配：枚举值列表 - 描述
        const mixedMatch = cleanDesc.match(/^([A-Z_]+(?:\s*,\s*[A-Z_]+)+|[a-z_]+(?:\s*,\s*[a-z_]+)+)\s*[-–—]\s*(.+)$/i)
        if (mixedMatch) {
          const enumPart = mixedMatch[1].trim()
          const descPart = mixedMatch[2].trim()
          const enumValues = enumPart.split(',').map(v => v.trim()).filter(v => v)
          if (enumValues.length >= 2) {
            examples.push(...enumValues)
            cleanDesc = descPart
          }
        } else {
          // 检查是否有其他模式：枚举值列表 + 简单描述（如 "human, mouse or TCR"）
          const orPattern = /^([A-Z_]+(?:\s*,\s*[A-Z_]+)+|[a-z_]+(?:\s*,\s*[a-z_]+)+)\s+(or|或)\s+([A-Z_]+|[a-z_]+)$/i
          const orMatch = cleanDesc.match(orPattern)
          if (orMatch) {
            const enumPart = orMatch[1].trim()
            const lastValue = orMatch[3].trim()
            const enumValues = enumPart.split(',').map(v => v.trim()).filter(v => v)
            if (enumValues.length >= 1) {
              examples.push(...enumValues, lastValue)
              cleanDesc = ''
            }
          }
        }
      }
    }
  }
  
  // 提取取值范围
  const rangePatterns = [
    /(?:range|范围|取值范围)[：:\s]+([^,\.\n]+)/i,
    /(?:between|between)\s+([^,\.\n]+)/i,
    /(?:from|from)\s+([^,\.\n]+)\s+to/i,
  ]
  let range: string | undefined
  for (const pattern of rangePatterns) {
    const match = cleanDesc.match(pattern)
    if (match && match[1]) {
      range = match[1].trim()
      break
    }
  }
  
  // 最后清理多余空格（cleanDesc 已经在上面处理过了）
  // 确保描述不为空（如果只有枚举值，描述可以为空）
  cleanDesc = cleanDesc.replace(/\s+/g, ' ').trim()
  
  parameters.set(paramName, {
    name: paramName,
    description: cleanDesc,
    tags,
    isOptional,
    defaultValue,
    examples: examples.length > 0 ? examples : undefined,
    range,
  })
}

/**
 * 格式化参数描述用于显示
 */
export function formatParameterDescription(param: ParsedParameterDescription): string {
  let result = param.description
  
  // 添加默认值信息
  if (param.defaultValue) {
    result += ` (默认值: ${param.defaultValue})`
  }
  
  // 添加取值范围
  if (param.range) {
    result += ` (范围: ${param.range})`
  }
  
  // 添加示例
  if (param.examples && param.examples.length > 0) {
    result += ` (示例: ${param.examples.join(', ')})`
  }
  
  return result
}

/**
 * 从工具描述中提取特定参数的信息
 */
export function getParameterInfoFromDescription(
  description: string,
  paramName: string
): ParsedParameterDescription | null {
  const parsed = parseToolDescription(description)
  return parsed.parameters.get(paramName) || null
}

