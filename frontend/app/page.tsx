'use client';

import React, { useState, useEffect } from 'react';
import { Plus, Menu, X } from 'lucide-react';
import { ChatContainer } from '@/components/chat/ChatContainer';
import { SessionList } from '@/components/sidebar/SessionList';
import { TemplatePanel } from '@/components/sidebar/TemplatePanel';
import { Session, Template } from '@/lib/types';
import { SessionStorage, createSession } from '@/lib/storage';
import { Button } from '@/components/ui/Button';

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);

  useEffect(() => {
    const loadedSessions = SessionStorage.getSessions();
    setSessions(loadedSessions);
    
    const loadedTemplates = SessionStorage.getTemplates();
    setTemplates(loadedTemplates);
    
    if (loadedSessions.length > 0) {
      setActiveSession(loadedSessions[0]);
    }
  }, []);

  const handleNewSession = () => {
    const newSession = createSession();
    const updatedSessions = [newSession, ...sessions];
    setSessions(updatedSessions);
    setActiveSession(newSession);
    SessionStorage.saveSession(newSession);
  };

  const handleSessionSelect = (sessionId: string) => {
    const session = sessions.find((s) => s.id === sessionId);
    if (session) {
      setActiveSession(session);
    }
  };

  const handleSessionDelete = (sessionId: string) => {
    const updatedSessions = sessions.filter((s) => s.id !== sessionId);
    setSessions(updatedSessions);
    SessionStorage.deleteSession(sessionId);
    
    if (activeSession?.id === sessionId) {
      setActiveSession(updatedSessions.length > 0 ? updatedSessions[0] : null);
    }
  };

  const handleSessionUpdate = (updatedSession: Session) => {
    setActiveSession(updatedSession);
    const updatedSessions = sessions.map((s) =>
      s.id === updatedSession.id ? updatedSession : s
    );
    setSessions(updatedSessions);
  };

  const handleTemplateSelect = (template: Template) => {
    if (!activeSession) {
      const newSession = createSession();
      const updatedSessions = [newSession, ...sessions];
      setSessions(updatedSessions);
      setActiveSession(newSession);
      SessionStorage.saveSession(newSession);
      setTimeout(() => {
        setPendingMessage(template.content);
      }, 100);
    } else {
      setPendingMessage(template.content);
    }
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Mobile menu button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed top-4 left-4 z-50 lg:hidden p-2 bg-white rounded-lg shadow-md"
      >
        {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } fixed lg:static inset-y-0 left-0 z-40 w-64 bg-gray-50 border-r border-gray-200 transform transition-transform duration-200 ease-in-out lg:translate-x-0 flex flex-col`}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">🧬 Bio-Agent</h1>
        </div>

        {/* New Session Button */}
        <div className="p-3">
          <Button
            onClick={handleNewSession}
            className="w-full"
            variant="primary"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>

        {/* Session List */}
        <SessionList
          sessions={sessions}
          activeSessionId={activeSession?.id}
          onSessionSelect={handleSessionSelect}
          onSessionDelete={handleSessionDelete}
        />

        {/* Templates Panel */}
        <TemplatePanel
          templates={templates}
          onTemplateSelect={handleTemplateSelect}
        />
      </div>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {activeSession ? (
          <>
            <div className="border-b border-gray-200 p-4 flex-shrink-0">
              <h2 className="text-lg font-semibold text-gray-900">
                {activeSession.title}
              </h2>
            </div>
            <ChatContainer
              session={activeSession}
              onSessionUpdate={handleSessionUpdate}
              pendingMessage={pendingMessage}
              onPendingMessageSent={() => setPendingMessage(null)}
            />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🧬</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Welcome to Bio-Agent
              </h2>
              <p className="text-gray-600 mb-6">
                Start a new chat to begin analyzing bioinformatics data
              </p>
              <Button onClick={handleNewSession} variant="primary">
                <Plus className="h-4 w-4 mr-2" />
                Start New Chat
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
