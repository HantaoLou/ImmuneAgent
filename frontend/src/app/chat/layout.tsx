'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { AntdRegistry } from '@ant-design/nextjs-registry';
import { verifyToken } from '@/services/authService';

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const isValid = await verifyToken();
      if (!isValid) {
        router.push('/login');
        return;
      }

      setIsAuthenticated(true);
      setIsLoading(false);
    };

    checkAuth();
  }, [router]);

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: '#0a0a0a',
        color: '#00ff88'
      }}>
        Loading...
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return <AntdRegistry>{children}</AntdRegistry>;
}
