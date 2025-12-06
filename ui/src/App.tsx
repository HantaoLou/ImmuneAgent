import { ConfigProvider, App as AntApp, Empty } from 'antd'
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom'
import MainLayout from './layouts/main-layout-component'
import Layout from './layouts/layout-component'
import AuthPage from './pages/auth-page'
import HomePage from './pages/home-page'
import AgentsPage from './pages/agents-page'
import ToolsPage from './pages/tools-page'
import ProtectedRoute from './components/ProtectedRoute'
import { AuthProvider } from './contexts/AuthContext'
import { UsecaseProvider } from './contexts/UsecaseContext'
import { colors } from './styles/tokens'
import '@llamaindex/chat-ui/styles/markdown.css' // code, latex and custom markdown styling
import '@llamaindex/chat-ui/styles/pdf.css' // pdf styling
import '@llamaindex/chat-ui/styles/editor.css' // document editor styling

function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: colors.primary[500],
          colorBgContainer: colors.background.primary,
          colorText: colors.text.primary,
          colorTextSecondary: colors.text.secondary,
          colorBorder: colors.border.primary,
        },
      }}
    >
      <AntApp>
        <Router>
          <AuthProvider>
            <UsecaseProvider>
              <Routes>
              <Route path="/auth" element={<AuthPage />} />
              {/* Grouped agents routes to avoid duplication */}
              <Route path="/*" element={<Layout />}>
                <Route path="home" element={<HomePage />} />
                <Route path="*" element={<ProtectedRoute />}>
                  <Route path="agents" element={<AgentsPage />} />
                  <Route path="tools" element={<ToolsPage />} />
                  <Route path="console" element={<Empty />} />
                  <Route path="chat" element={<MainLayout />} />
                  <Route path="*" element={<Navigate to="/home" replace />} />
                </Route>
              </Route>
            </Routes>
            </UsecaseProvider>
          </AuthProvider>
        </Router>
      </AntApp>
    </ConfigProvider>
  )
}

export default App
