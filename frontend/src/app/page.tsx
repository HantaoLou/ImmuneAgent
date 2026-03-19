'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { verifyToken } from '@/services/authService';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const isValid = await verifyToken();
      if (isValid) {
        router.push('/chat');
      } else {
        router.push('/login');
      }
    };

    checkAuth();
  }, [router]);

  return (
    <div style={{ 
      display: 'flex', 
      height: '100vh', 
      width: '100vw', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: '#0a0a0a',
      color: '#86909C'
    }}>
      Redirecting...
    </div>
  );
}
