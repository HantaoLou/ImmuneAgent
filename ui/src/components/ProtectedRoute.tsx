import React from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const ProtectedRoute: React.FC = () => {
  const { isAuthenticated, checkAuth } = useAuth()

  // Check authentication status
  const isAuth = isAuthenticated || checkAuth()

  if (!isAuth) {
    // Redirect to auth page if not authenticated
    return <Navigate to="/auth" replace />
  }

  return <Outlet />
}

export default ProtectedRoute
