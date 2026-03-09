# Bio-Agent Frontend

NextJS 15 frontend for Bio-Agent Demo System.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local with your API URL
```

## Running

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- Chat interface with real-time streaming
- Session management (create, switch, delete sessions)
- Template system for common tasks
- Responsive design for mobile and desktop
- Persistent storage using localStorage

## Tech Stack

- NextJS 15
- TypeScript
- TailwindCSS
- Lucide React (icons)
- clsx (class names)

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main chat page
│   └── globals.css         # Global styles
├── components/
│   ├── ui/
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   └── LoadingSpinner.tsx
│   ├── chat/
│   │   ├── ChatContainer.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── MessageInput.tsx
│   │   ├── MessageList.tsx
│   │   └── StreamingMessage.tsx
│   └── sidebar/
│       ├── SessionItem.tsx
│       ├── SessionList.tsx
│       └── TemplatePanel.tsx
└── lib/
    ├── api.ts              # API client
    ├── storage.ts          # Local storage utilities
    └── types.ts            # TypeScript types
```
