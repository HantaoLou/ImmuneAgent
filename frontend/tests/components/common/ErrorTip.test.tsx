import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorTip } from '@/components/common/ErrorTip';

describe('ErrorTip Component', () => {
  describe('Basic rendering', () => {
    it('should render error message', () => {
      render(<ErrorTip message="Test error message" />);

      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('should render with error icon', () => {
      const { container } = render(<ErrorTip message="Error occurred" />);

      expect(container.querySelector('.anticon-close-circle')).toBeInTheDocument();
    });

    it('should display as error type', () => {
      const { container } = render(<ErrorTip message="Error" />);

      expect(container.querySelector('.ant-alert-error')).toBeInTheDocument();
    });
  });

  describe('Message variations', () => {
    it('should handle long error messages', () => {
      const longMessage = 'This is a very long error message that might span multiple lines and should be displayed correctly by the component without any issues';
      render(<ErrorTip message={longMessage} />);

      expect(screen.getByText(longMessage)).toBeInTheDocument();
    });

    it('should handle special characters', () => {
      const specialMessage = 'Error: <>&"\'` special chars';
      render(<ErrorTip message={specialMessage} />);

      expect(screen.getByText(specialMessage)).toBeInTheDocument();
    });

    it('should handle emoji in message', () => {
      const emojiMessage = 'Error occurred 😢 Please try again 🔄';
      render(<ErrorTip message={emojiMessage} />);

      expect(screen.getByText(emojiMessage)).toBeInTheDocument();
    });

    it('should handle unicode characters', () => {
      const unicodeMessage = '错误：中文测试 日本語テスト 한글';
      render(<ErrorTip message={unicodeMessage} />);

      expect(screen.getByText(unicodeMessage)).toBeInTheDocument();
    });

    it('should handle empty message', () => {
      render(<ErrorTip message="" />);

      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
    });
  });

  describe('Close functionality', () => {
    it('should be closable by default', () => {
      const { container } = render(<ErrorTip message="Test error" />);

      expect(container.querySelector('.ant-alert-close-icon')).toBeInTheDocument();
    });

    it('should call onClose when closed', async () => {
      const user = userEvent.setup();
      const onClose = vi.fn();

      render(<ErrorTip message="Test error" onClose={onClose} />);

      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      expect(onClose).toHaveBeenCalled();
    });

    it('should work without onClose handler', async () => {
      const user = userEvent.setup();

      render(<ErrorTip message="Test error" />);

      const closeButton = screen.getByRole('button', { name: /close/i });

      await expect(user.click(closeButton)).resolves.not.toThrow();
    });
  });

  describe('Accessibility', () => {
    it('should have alert role', () => {
      render(<ErrorTip message="Error message" />);

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should be visible to screen readers', () => {
      render(<ErrorTip message="Important error" />);

      expect(screen.getByText('Important error')).toBeVisible();
    });
  });

  describe('Styling', () => {
    it('should have mb-4 class', () => {
      const { container } = render(<ErrorTip message="Error" />);

      expect(container.firstChild).toHaveClass('mb-4');
    });

    it('should show icon', () => {
      const { container } = render(<ErrorTip message="Error" />);

      expect(container.querySelector('.ant-alert-icon')).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle very long messages', () => {
      const veryLongMessage = 'A'.repeat(1000);
      render(<ErrorTip message={veryLongMessage} />);

      expect(screen.getByText(veryLongMessage)).toBeInTheDocument();
    });

    it('should handle multiline messages', () => {
      const multilineMessage = 'Error line 1\nError line 2\nError line 3';
      render(<ErrorTip message={multilineMessage} />);

      expect(screen.getByText(/Error line 1/)).toBeInTheDocument();
    });

    it('should handle HTML entities', () => {
      const htmlMessage = 'Error: &lt;script&gt;alert("xss")&lt;/script&gt;';
      render(<ErrorTip message={htmlMessage} />);

      expect(screen.getByText(htmlMessage)).toBeInTheDocument();
    });
  });
});
