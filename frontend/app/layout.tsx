import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bio-Agent Demo",
  description: "Interactive demo for Bio-Agent - AI-powered bioinformatics assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
