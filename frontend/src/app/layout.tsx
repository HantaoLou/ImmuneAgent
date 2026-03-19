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
    default: "Agent Chat - Minimalist Tech-Style AI Chat Interface",
    template: "%s | Agent Chat"
  },
  description: "Powerful AI Agent chat frontend with file upload/download, session management, and dark tech-style UI. A modern chat application.",
  keywords: ["AI Chat", "Agent", "Chat Interface", "File Upload", "Next.js", "TypeScript", "Dark Theme"],
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
    locale: "en_US",
    url: "https://agent-chat.example.com",
    siteName: "Agent Chat",
    title: "Agent Chat - Minimalist Tech-Style AI Chat Interface",
    description: "Powerful AI Agent chat frontend with file upload/download, session management, and dark tech-style UI",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Agent Chat - Dark Tech-Style Chat Interface",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Agent Chat - Minimalist Tech-Style AI Chat Interface",
    description: "Powerful AI Agent chat frontend with file upload/download, session management",
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
    <html lang="en" className={`${jetbrainsMono.variable} ${spaceGrotesk.variable}`}>
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
