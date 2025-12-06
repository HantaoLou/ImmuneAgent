import React, {
  createContext,
  useState,
  useContext,
  useEffect,
} from 'react'
import type { ReactNode } from 'react'
import { type UsecaseInfo } from '../services/sessions-service'

interface UsecaseContextType {
  selectedUsecase: UsecaseInfo | null
  setSelectedUsecase: (usecase: UsecaseInfo | null) => void
  clearUsecase: () => void
  loadingUsecases: boolean
}

const UsecaseContext = createContext<UsecaseContextType | undefined>(undefined)

export const UsecaseProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [selectedUsecase, setSelectedUsecase] = useState<UsecaseInfo | null>(null)
  const [loadingUsecases] = useState(false)

  // 从localStorage加载持久化的usecase
  useEffect(() => {
    const savedUsecase = localStorage.getItem('selectedUsecase')
    if (savedUsecase) {
      try {
        const parsedUsecase = JSON.parse(savedUsecase)
        setSelectedUsecase(parsedUsecase)
      } catch (error) {
        console.error('Failed to parse saved usecase:', error)
        localStorage.removeItem('selectedUsecase')
      }
    }
  }, [])

  // 持久化usecase到localStorage
  const handleSetSelectedUsecase = (usecase: UsecaseInfo | null) => {
    setSelectedUsecase(usecase)
    if (usecase) {
      localStorage.setItem('selectedUsecase', JSON.stringify(usecase))
    } else {
      localStorage.removeItem('selectedUsecase')
    }
  }

  const clearUsecase = () => {
    setSelectedUsecase(null)
    localStorage.removeItem('selectedUsecase')
  }

  return (
    <UsecaseContext.Provider
      value={{ 
        selectedUsecase, 
        setSelectedUsecase: handleSetSelectedUsecase, 
        clearUsecase, 
        loadingUsecases
      }}
    >
      {children}
    </UsecaseContext.Provider>
  )
}

export const useUsecase = () => {
  const context = useContext(UsecaseContext)
  if (context === undefined) {
    throw new Error('useUsecase must be used within a UsecaseProvider')
  }
  return context
}