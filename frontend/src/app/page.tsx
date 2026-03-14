'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.push('/chat');
  }, [router]);

  return (
    <div className="flex h-screen w-screen items-center justify-center">
      <div className="text-[#86909C]">正在跳转到聊天页面...</div>
    </div>
  );
}
