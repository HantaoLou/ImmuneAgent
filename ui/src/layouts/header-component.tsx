import {
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
} from '@ant-design/icons'
import { Dropdown } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import Logo from '../assets/Logo.png'

const Header: React.FC = () => {
  const navigate = useNavigate()
  const { logout } = useAuth()

  // User dropdown menu
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      danger: true,
    },
  ]

  const handleUserMenuClick = ({ key }: { key: string }) => {
    switch (key) {
      case 'logout':
        logout()
        break
      case 'settings':
        console.log('Settings clicked')
        break
      case 'profile':
        console.log('Profile clicked')
        break
    }
  }

  return (
    <div className="bg-gray-100 w-full h-16 flex items-center justify-between px-6">
      {/* Logo - 左侧手写风格 */}
      <div className="flex items-center cursor-pointer" onClick={() => navigate('/home')}>
        <img 
          src={Logo} 
          alt="Logo" 
          className="h-10 object-contain"
        />
      </div>

      {/* Navigation Links - 右侧导航链接 */}
      <div className="flex items-center space-x-12">
        <span
          className="text-black text-base font-normal cursor-pointer hover:text-gray-600 transition-all duration-200 hover:scale-110"
          onClick={() => navigate('/tools')}
        >
          Tools
        </span>
        <span
          className="text-black text-base font-normal cursor-pointer hover:text-gray-600 transition-all duration-200 hover:scale-110"
          onClick={() => navigate('/agents')}
        >
          Agents
        </span>
        <span
          className="text-gray-500 cursor-not-allowed"
          // className="text-black text-base font-normal cursor-pointer hover:text-gray-600 transition-all duration-200 hover:scale-110"
          // onClick={() => navigate('/console')}
        >
          Console
        </span>

        <Dropdown
          menu={{
            items: userMenuItems,
            onClick: handleUserMenuClick,
          }}
          placement="bottomRight"
          arrow
        >
          <div className="w-8 h-8 border-2 border-gray-400 rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-all duration-200">
            <UserOutlined style={{ fontSize: '16px', color: '#666' }} />
          </div>
        </Dropdown>
      </div>
    </div>
  )
}

export default Header
