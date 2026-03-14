import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EmptyState } from '@/components/common/EmptyState';

describe('EmptyState Component', () => {
  describe('Default rendering', () => {
    it('should render with default tip text', () => {
      render(<EmptyState />);

      expect(screen.getByText('暂无数据')).toBeInTheDocument();
    });

    it('should render title', () => {
      render(<EmptyState />);

      expect(screen.getByText('开始对话')).toBeInTheDocument();
    });

    it('should render icon', () => {
      render(<EmptyState />);

      expect(screen.getByText('💬')).toBeInTheDocument();
    });
  });

  describe('Custom tip', () => {
    it('should render custom tip text', () => {
      const customTip = '点击开始新对话';
      render(<EmptyState tip={customTip} />);

      expect(screen.getByText(customTip)).toBeInTheDocument();
    });

    it('should render long custom tip', () => {
      const longTip = '这是一个很长的提示文本，用于测试组件是否能正确显示长文本内容';
      render(<EmptyState tip={longTip} />);

      expect(screen.getByText(longTip)).toBeInTheDocument();
    });

    it('should render empty string tip', () => {
      render(<EmptyState tip="" />);

      const tipElement = screen.getByText('开始对话').parentElement;
      expect(tipElement).toHaveTextContent('');
    });
  });

  describe('Component structure', () => {
    it('should have correct structure', () => {
      const { container } = render(<EmptyState />);

      expect(container.firstChild).toBeInTheDocument();
    });

    it('should render icon and content', () => {
      render(<EmptyState tip="Test tip" />);

      expect(screen.getByText('💬')).toBeInTheDocument();
      expect(screen.getByText('开始对话')).toBeInTheDocument();
      expect(screen.getByText('Test tip')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should be visible to screen readers', () => {
      render(<EmptyState tip="No messages yet" />);

      expect(screen.getByText('No messages yet')).toBeVisible();
    });

    it('should have meaningful content', () => {
      render(<EmptyState />);

      const title = screen.getByText('开始对话');
      const tip = screen.getByText('暂无数据');

      expect(title).toBeInTheDocument();
      expect(tip).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle special characters in tip', () => {
      const specialTip = '特殊字符测试 <>&"\'';
      render(<EmptyState tip={specialTip} />);

      expect(screen.getByText(specialTip)).toBeInTheDocument();
    });

    it('should handle emoji in tip', () => {
      const emojiTip = '开始聊天 👋 发送消息 💬';
      render(<EmptyState tip={emojiTip} />);

      expect(screen.getByText(emojiTip)).toBeInTheDocument();
    });

    it('should handle unicode in tip', () => {
      const unicodeTip = '中文测试 日本語テスト 한글테스트';
      render(<EmptyState tip={unicodeTip} />);

      expect(screen.getByText(unicodeTip)).toBeInTheDocument();
    });
  });
});
