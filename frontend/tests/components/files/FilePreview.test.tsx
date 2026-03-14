import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FilePreview } from '@/components/files/FilePreview';
import { fixtures } from '../../setup/fixtures/files';

vi.mock('@/lib/fileUtils', () => ({
  fileUtils: {
    truncateFileName: vi.fn((name: string) => name.length > 20 ? name.slice(0, 20) + '...' : name),
    formatFileSize: vi.fn((size: number) => {
      if (size < 1024) return `${size} B`;
      if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    }),
  },
}));

const mockOnRemove = vi.fn();

describe('FilePreview Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render file name', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      expect(screen.getByText(fixtures.imageFile.name)).toBeInTheDocument();
    });

    it('should render file size', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      expect(screen.getByText(/2/)).toBeInTheDocument();
    });

    it('should have title attribute on file name', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      expect(screen.getByTitle(fixtures.imageFile.name)).toBeInTheDocument();
    });
  });

  describe('Image preview', () => {
    it('should render image preview for image files with url', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      const img = screen.getByRole('img');
      expect(img).toBeInTheDocument();
      expect(img).toHaveAttribute('src', fixtures.imageFile.url);
      expect(img).toHaveAttribute('alt', fixtures.imageFile.name);
    });

    it('should not render image preview for non-image files', () => {
      render(<FilePreview file={fixtures.documentFile} />);
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
    });

    it('should show icon instead of image for document files', () => {
      render(<FilePreview file={fixtures.documentFile} />);
      expect(screen.getByText('📄')).toBeInTheDocument();
    });

    it('should show icon for image files without url', () => {
      const imageWithoutUrl = { ...fixtures.imageFile, url: undefined };
      render(<FilePreview file={imageWithoutUrl} />);
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });
  });

  describe('File icons by category', () => {
    it('should render image icon for image category without url', () => {
      const file = { ...fixtures.documentFile, category: 'image' as const, url: undefined };
      render(<FilePreview file={file} />);
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });

    it('should render document icon for document category', () => {
      render(<FilePreview file={fixtures.documentFile} />);
      expect(screen.getByText('📄')).toBeInTheDocument();
    });

    it('should render code icon for code category', () => {
      render(<FilePreview file={fixtures.codeFile} />);
      expect(screen.getByText('💻')).toBeInTheDocument();
    });

    it('should render data icon for data category', () => {
      render(<FilePreview file={fixtures.dataFile} />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });
  });

  describe('Upload progress', () => {
    it('should show progress bar when uploadProgress is less than 100', () => {
      render(<FilePreview file={fixtures.uploadingFile} />);
      const progressBar = document.querySelector('[style*="width: 45%"]');
      expect(progressBar).toBeInTheDocument();
    });

    it('should not show progress bar when uploadProgress is 100', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      const progressBar = document.querySelector('[class*="progressBar"]');
      expect(progressBar).not.toBeInTheDocument();
    });

    it('should not show progress bar when uploadProgress is undefined', () => {
      const fileWithoutProgress = { ...fixtures.imageFile, uploadProgress: undefined };
      render(<FilePreview file={fileWithoutProgress} />);
      const progressBar = document.querySelector('[class*="progressBar"]');
      expect(progressBar).not.toBeInTheDocument();
    });

    it('should show correct progress width', () => {
      render(<FilePreview file={fixtures.uploadingFile} />);
      const progressBar = document.querySelector('[style*="width: 45%"]');
      expect(progressBar).toHaveStyle({ width: '45%' });
    });
  });

  describe('Remove button', () => {
    it('should not render remove button by default', () => {
      render(<FilePreview file={fixtures.imageFile} />);
      expect(screen.queryByRole('button', { name: '移除文件' })).not.toBeInTheDocument();
    });

    it('should render remove button when onRemove is provided', () => {
      render(<FilePreview file={fixtures.imageFile} onRemove={mockOnRemove} />);
      expect(screen.getByRole('button', { name: '移除文件' })).toBeInTheDocument();
    });

    it('should call onRemove when clicked', async () => {
      const user = userEvent.setup();
      render(<FilePreview file={fixtures.imageFile} onRemove={mockOnRemove} />);
      
      await user.click(screen.getByRole('button', { name: '移除文件' }));
      
      expect(mockOnRemove).toHaveBeenCalled();
    });

    it('should have aria-label for accessibility', () => {
      render(<FilePreview file={fixtures.imageFile} onRemove={mockOnRemove} />);
      expect(screen.getByLabelText('移除文件')).toBeInTheDocument();
    });
  });

  describe('Compact mode', () => {
    it('should not have compact class by default', () => {
      const { container } = render(<FilePreview file={fixtures.imageFile} />);
      expect(container.firstChild?.className).not.toMatch(/compact/i);
    });

    it('should have compact class when compact is true', () => {
      const { container } = render(<FilePreview file={fixtures.imageFile} compact={true} />);
      expect(container.firstChild?.className).toMatch(/compact/i);
    });
  });

  describe('Edge cases', () => {
    it('should handle file with zero size', () => {
      const zeroSizeFile = { ...fixtures.imageFile, size: 0 };
      render(<FilePreview file={zeroSizeFile} />);
      expect(screen.getByText(/0 B/)).toBeInTheDocument();
    });

    it('should handle file with very large size', () => {
      const largeFile = { ...fixtures.imageFile, size: 500 * 1024 * 1024 };
      render(<FilePreview file={largeFile} />);
      expect(screen.getByText(/500/)).toBeInTheDocument();
    });

    it('should handle progress at 0%', () => {
      const file = { ...fixtures.uploadingFile, uploadProgress: 0 };
      render(<FilePreview file={file} />);
      const progressBar = document.querySelector('[style*="width: 0%"]');
      expect(progressBar).toBeInTheDocument();
    });

    it('should handle progress at 99%', () => {
      const file = { ...fixtures.uploadingFile, uploadProgress: 99 };
      render(<FilePreview file={file} />);
      const progressBar = document.querySelector('[style*="width: 99%"]');
      expect(progressBar).toBeInTheDocument();
    });

    it('should handle all props combined', async () => {
      const user = userEvent.setup();
      render(
        <FilePreview
          file={fixtures.imageFile}
          onRemove={mockOnRemove}
          compact={true}
        />
      );
      
      expect(screen.getByRole('img')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '移除文件' })).toBeInTheDocument();
      
      await user.click(screen.getByRole('button', { name: '移除文件' }));
      expect(mockOnRemove).toHaveBeenCalled();
    });
  });
});
