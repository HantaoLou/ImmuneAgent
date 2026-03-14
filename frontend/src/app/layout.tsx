import type { Metadata } from "next";
import "./globals.css";
import "./theme.css";
import { AntdRegistry } from '@ant-design/nextjs-registry';
import { JetBrains_Mono, Space_Grotesk } from 'next/font/google';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
});

export const metadata: Metadata = {
  title: {
    default: "Agent Chat - 极简科技风 AI 聊天界面",
    template: "%s | Agent Chat"
  },
  description: "功能强大的 AI Agent 聊天前端，支持文件上传下载、会话管理、暗色科技风 UI。对标豆包 chat 页面的现代化聊天应用。",
  keywords: ["AI Chat", "Agent", "聊天界面", "文件上传", "Next.js", "TypeScript", "暗色主题"],
  authors: [{ name: "Agent Chat Team" }],
  creator: "Agent Chat",
  publisher: "Agent Chat",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "zh_CN",
    url: "https://agent-chat.example.com",
    siteName: "Agent Chat",
    title: "Agent Chat - 极简科技风 AI 聊天界面",
    description: "功能强大的 AI Agent 聊天前端，支持文件上传下载、会话管理、暗色科技风 UI",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Agent Chat - 暗色科技风聊天界面",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Agent Chat - 极简科技风 AI 聊天界面",
    description: "功能强大的 AI Agent 聊天前端，支持文件上传下载、会话管理",
    images: ["/og-image.png"],
    creator: "@agentchat",
  },
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 5,
  },
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#4080FF" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0f" },
  ],
  manifest: "/manifest.json",
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
    ],
    apple: [
      { url: "/apple-icon.png" },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className={`${jetbrainsMono.variable} ${spaceGrotesk.variable}`}>
      <body className="antialiased">
        <ErrorBoundary>
          <AntdRegistry>
            {children}
          </AntdRegistry>
        </ErrorBoundary>
      </body>
    </html>
  );
}
