import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { InputPanel } from '@/components/chat/InputPanel';

describe('InputPanel Component', () => {
  const mockOnChange = vi.fn();
  const mockOnSend = vi.fn();
  const mockOnClear = vi.fn();
  const mockOnAttachmentsChange = vi.fn();

  const defaultProps = {
    value: '',
    onChange: mockOnChange,
    onSend: mockOnSend,
    onClear: mockOnClear,
    disabled: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render textarea', () => {
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toBeInTheDocument();
    });

    it('should render send button', () => {
      render(<InputPanel {...defaultProps} />);
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('should render clear button', () => {
      render(<InputPanel {...defaultProps} value="test" />);
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('should display current value', () => {
      render(<InputPanel {...defaultProps} value="Hello World" />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveValue('Hello World');
    });
  });

  describe('Input handling', () => {
    it('should call onChange when input changes', async () => {
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      
      fireEvent.change(textarea, { target: { value: 'Hello' } });
      
      expect(mockOnChange).toHaveBeenCalledWith('Hello');
    });

    it('should update value on change', () => {
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      
      fireEvent.change(textarea, { target: { value: 'Test message' } });
      
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  describe('Send button state', () => {
    it('should be disabled when input is empty', () => {
      render(<InputPanel {...defaultProps} value="" />);
      const sendButton = screen.getByRole('button', { name: /发送/i });
      expect(sendButton).toBeDisabled();
    });

    it('should be disabled when only whitespace', () => {
      render(<InputPanel {...defaultProps} value="   " />);
      const sendButton = screen.getByRole('button', { name: /发送/i });
      expect(sendButton).toBeDisabled();
    });

    it('should be enabled when input has content', () => {
      render(<InputPanel {...defaultProps} value="Hello" />);
      const sendButton = screen.getByRole('button', { name: /发送/i });
      expect(sendButton).not.toBeDisabled();
    });

    it('should call onSend when clicked', async () => {
      render(<InputPanel {...defaultProps} value="Test message" />);
      const sendButton = screen.getByRole('button', { name: /发送/i });
      
      await userEvent.click(sendButton);
      
      expect(mockOnSend).toHaveBeenCalled();
    });

    it('should be disabled when prop disabled=true', () => {
      render(<InputPanel {...defaultProps} value="Test" disabled={true} />);
      const sendButton = screen.getByRole('button', { name: /发送/i });
      expect(sendButton).toBeDisabled();
    });
  });

  describe('Clear button', () => {
    it('should call onClear when clicked', async () => {
      render(<InputPanel {...defaultProps} value="Test" />);
      const clearButton = screen.getByRole('button', { name: /清空/i });
      
      await userEvent.click(clearButton);
      
      expect(mockOnClear).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should have textbox role', () => {
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toBeInTheDocument();
    });

    it('should be focusable', () => {
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      textarea.focus();
      expect(textarea).toHaveFocus();
    });
  });

  describe('Edge cases', () => {
    it('should handle very long input', () => {
      const longText = 'a'.repeat(5000);
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      
      fireEvent.change(textarea, { target: { value: longText } });
      
      expect(mockOnChange).toHaveBeenCalled();
    });

    it('should handle special characters', () => {
      const specialText = '!@#$%^&*()_+-={}[]|\\:";\'<>?,./~`';
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      
      fireEvent.change(textarea, { target: { value: specialText } });
      
      expect(mockOnChange).toHaveBeenCalledWith(specialText);
    });

    it('should handle emoji input', () => {
      const emojiText = 'Hello 👋 World 🌍';
      render(<InputPanel {...defaultProps} />);
      const textarea = screen.getByRole('textbox');
      
      fireEvent.change(textarea, { target: { value: emojiText } });
      
      expect(mockOnChange).toHaveBeenCalledWith(emojiText);
    });
  });
});
