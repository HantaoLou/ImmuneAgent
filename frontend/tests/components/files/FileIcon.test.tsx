import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FileIcon } from '@/components/files/FileIcon';
import { FileCategory } from '@/types';

describe('FileIcon Component', () => {
  describe('Category icons', () => {
    it('should render image icon', () => {
      render(<FileIcon category="image" />);
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });

    it('should render document icon', () => {
      render(<FileIcon category="document" />);
      expect(screen.getByText('📄')).toBeInTheDocument();
    });

    it('should render code icon', () => {
      render(<FileIcon category="code" />);
      expect(screen.getByText('💻')).toBeInTheDocument();
    });

    it('should render data icon', () => {
      render(<FileIcon category="data" />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });

    it('should render other icon', () => {
      render(<FileIcon category="other" />);
      expect(screen.getByText('📎')).toBeInTheDocument();
    });
  });

  describe('Size variations', () => {
    it('should render small size', () => {
      const { container } = render(<FileIcon category="image" size="small" />);
      expect(container.firstChild).toHaveClass('small');
    });

    it('should render medium size by default', () => {
      const { container } = render(<FileIcon category="image" />);
      expect(container.firstChild).toHaveClass('medium');
    });

    it('should render large size', () => {
      const { container } = render(<FileIcon category="image" size="large" />);
      expect(container.firstChild).toHaveClass('large');
    });
  });

  describe('Color application', () => {
    it('should apply cyan color for image', () => {
      const { container } = render(<FileIcon category="image" />);
      const icon = container.firstChild as HTMLElement;
      expect(icon.style.color).toBe('var(--accent-cyan)');
    });

    it('should apply purple color for document', () => {
      const { container } = render(<FileIcon category="document" />);
      const icon = container.firstChild as HTMLElement;
      expect(icon.style.color).toBe('var(--accent-purple)');
    });

    it('should apply lime color for code', () => {
      const { container } = render(<FileIcon category="code" />);
      const icon = container.firstChild as HTMLElement;
      expect(icon.style.color).toBe('var(--accent-lime)');
    });

    it('should apply gold color for data', () => {
      const { container } = render(<FileIcon category="data" />);
      const icon = container.firstChild as HTMLElement;
      expect(icon.style.color).toBe('var(--accent-gold)');
    });

    it('should apply muted color for other', () => {
      const { container } = render(<FileIcon category="other" />);
      const icon = container.firstChild as HTMLElement;
      expect(icon.style.color).toBe('var(--text-muted)');
    });
  });

  describe('Custom className', () => {
    it('should apply custom className', () => {
      const { container } = render(<FileIcon category="image" className="custom-class" />);
      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('should combine custom className with size class', () => {
      const { container } = render(
        <FileIcon category="image" size="large" className="custom-class" />
      );
      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('Component structure', () => {
    it('should render as div element', () => {
      const { container } = render(<FileIcon category="image" />);
      expect(container.firstChild?.nodeName).toBe('DIV');
    });

    it('should have icon class', () => {
      const { container } = render(<FileIcon category="image" size="large" />);
      expect(container.firstChild).toHaveClass('icon');
    });
  });

  describe('All categories', () => {
    const categories: FileCategory[] = ['image', 'document', 'code', 'data', 'other'];

    it('should render all category types', () => {
      const icons = ['🖼️', '📄', '💻', '📊', '📎'];
      categories.forEach((category, index) => {
        const { unmount } = render(<FileIcon category={category} />);
        expect(screen.getByText(icons[index])).toBeInTheDocument();
        unmount();
      });
    });

    it('should apply correct colors for all categories', () => {
      const colorMap = {
        image: 'var(--accent-cyan)',
        document: 'var(--accent-purple)',
        code: 'var(--accent-lime)',
        data: 'var(--accent-gold)',
        other: 'var(--text-muted)',
      };

      categories.forEach((category) => {
        const { container, unmount } = render(<FileIcon category={category} />);
        const icon = container.firstChild as HTMLElement;
        expect(icon.style.color).toBe(colorMap[category]);
        unmount();
      });
    });
  });

  describe('Edge cases', () => {
    it('should render consistently for same props', () => {
      const { container: container1 } = render(<FileIcon category="image" size="medium" />);
      const { container: container2 } = render(<FileIcon category="image" size="medium" />);

      expect(container1.firstChild).toBeDefined();
      expect(container2.firstChild).toBeDefined();
    });

    it('should handle multiple size changes', () => {
      const { rerender, container } = render(<FileIcon category="image" size="small" />);
      expect(container.firstChild).toHaveClass('small');

      rerender(<FileIcon category="image" size="medium" />);
      expect(container.firstChild).toHaveClass('medium');

      rerender(<FileIcon category="image" size="large" />);
      expect(container.firstChild).toHaveClass('large');
    });

    it('should handle className changes', () => {
      const { rerender, container } = render(
        <FileIcon category="image" className="class-1" />
      );
      expect(container.firstChild).toHaveClass('class-1');

      rerender(<FileIcon category="image" className="class-2" />);
      expect(container.firstChild).toHaveClass('class-2');
      expect(container.firstChild).not.toHaveClass('class-1');
    });
  });

  describe('Accessibility', () => {
    it('should be visible', () => {
      render(<FileIcon category="image" />);
      expect(screen.getByText('🖼️')).toBeVisible();
    });
  });
});
