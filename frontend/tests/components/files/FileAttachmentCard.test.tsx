import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileAttachmentCard } from '@/components/files/FileAttachmentCard';
import { fixtures } from '../../setup/fixtures/files';

vi.mock('@/services/fileService', () => ({
  fileService: {
    download: vi.fn().mockResolvedValue(new Blob(['test content'])),
    triggerDownload: vi.fn(),
  },
}));

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

const mockOnDelete = vi.fn();

describe('FileAttachmentCard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render file name', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByText(fixtures.imageFile.name)).toBeInTheDocument();
    });

    it('should render file size', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByText('2.0 MB')).toBeInTheDocument();
    });

    it('should render upload time', () => {
      render(<FileAttachmentCard file={fixtures.documentFile} />);
      const timeText = new Date(fixtures.documentFile.uploadTime).toLocaleString('zh-CN');
      expect(screen.getByText(new RegExp(timeText.split(' ')[0]))).toBeInTheDocument();
    });

    it('should render FileIcon with correct category', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });
  });

  describe('Download button', () => {
    it('should render download button by default', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument();
    });

    it('should hide download button when showDownload is false', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} showDownload={false} />);
      expect(screen.queryByRole('button', { name: /download/i })).not.toBeInTheDocument();
    });

    it('should call fileService.download when clicked', async () => {
      const { fileService } = await import('@/services/fileService');
      const user = userEvent.setup();
      
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      
      await user.click(screen.getByRole('button', { name: /download/i }));
      
      await waitFor(() => {
        expect(fileService.download).toHaveBeenCalledWith({
          fileId: fixtures.imageFile.id,
          sessionId: fixtures.imageFile.sessionId,
        });
      });
    });

    it('should call triggerDownload with correct filename', async () => {
      const { fileService } = await import('@/services/fileService');
      const user = userEvent.setup();
      
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      
      await user.click(screen.getByRole('button', { name: /download/i }));
      
      await waitFor(() => {
        expect(fileService.triggerDownload).toHaveBeenCalledWith(
          expect.any(Blob),
          fixtures.imageFile.name
        );
      });
    });

    it('should handle download error gracefully', async () => {
      const { fileService } = await import('@/services/fileService');
      vi.mocked(fileService.download).mockRejectedValueOnce(new Error('Download failed'));
      
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const user = userEvent.setup();
      
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      
      await user.click(screen.getByRole('button', { name: /download/i }));
      
      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('文件下载失败:', expect.any(Error));
      });
      
      consoleSpy.mockRestore();
    });
  });

  describe('Delete button', () => {
    it('should not render delete button by default', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
    });

    it('should render delete button when showDelete is true', () => {
      render(
        <FileAttachmentCard
          file={fixtures.imageFile}
          showDelete={true}
          onDelete={mockOnDelete}
        />
      );
      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
    });

    it('should not render delete button without onDelete handler', () => {
      render(
        <FileAttachmentCard
          file={fixtures.imageFile}
          showDelete={true}
        />
      );
      expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
    });

    it('should call onDelete when clicked', async () => {
      const user = userEvent.setup();
      
      render(
        <FileAttachmentCard
          file={fixtures.imageFile}
          showDelete={true}
          onDelete={mockOnDelete}
        />
      );
      
      await user.click(screen.getByRole('button', { name: /delete/i }));
      
      expect(mockOnDelete).toHaveBeenCalled();
    });
  });

  describe('Compact mode', () => {
    it('should not have compact class by default', () => {
      const { container } = render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(container.firstChild?.className).not.toMatch(/compact/i);
    });

    it('should have compact class when compact is true', () => {
      const { container } = render(<FileAttachmentCard file={fixtures.imageFile} compact={true} />);
      expect(container.firstChild?.className).toMatch(/compact/i);
    });
  });

  describe('Different file categories', () => {
    it('should render image file correctly', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });

    it('should render document file correctly', () => {
      render(<FileAttachmentCard file={fixtures.documentFile} />);
      expect(screen.getByText('📄')).toBeInTheDocument();
    });

    it('should render code file correctly', () => {
      render(<FileAttachmentCard file={fixtures.codeFile} />);
      expect(screen.getByText('💻')).toBeInTheDocument();
    });

    it('should render data file correctly', () => {
      render(<FileAttachmentCard file={fixtures.dataFile} />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have title attribute on file name', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      const nameElement = screen.getByTitle(fixtures.imageFile.name);
      expect(nameElement).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle very long file names', async () => {
      const { fileUtils } = await import('@/lib/fileUtils');
      const longName = 'a'.repeat(100) + '.txt';
      const longFile = { ...fixtures.documentFile, name: longName };
      
      render(<FileAttachmentCard file={longFile} />);
      
      expect(fileUtils.truncateFileName).toHaveBeenCalledWith(longName);
    });

    it('should handle file with zero size', () => {
      const zeroSizeFile = { ...fixtures.imageFile, size: 0 };
      
      render(<FileAttachmentCard file={zeroSizeFile} />);
      
      expect(screen.getByText(/0 B/)).toBeInTheDocument();
    });

    it('should handle file with very large size', async () => {
      const { fileUtils } = await import('@/lib/fileUtils');
      const largeFile = { ...fixtures.imageFile, size: 100 * 1024 * 1024 };
      
      render(<FileAttachmentCard file={largeFile} />);
      
      expect(fileUtils.formatFileSize).toHaveBeenCalledWith(largeFile.size);
    });

    it('should handle all props combined', async () => {
      const user = userEvent.setup();
      
      render(
        <FileAttachmentCard
          file={fixtures.imageFile}
          showDownload={true}
          showDelete={true}
          onDelete={mockOnDelete}
          compact={true}
        />
      );
      
      expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
      
      await user.click(screen.getByRole('button', { name: /delete/i }));
      expect(mockOnDelete).toHaveBeenCalled();
    });
  });
});

    it('should have compact class when compact is true', () => {
      const { container } = render(<FileAttachmentCard file={fixtures.imageFile} compact={true} />);
      expect(container.firstChild?.className).toMatch(/compact/i);
    });
  });

  describe('Different file categories', () => {
    it('should render image file correctly', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      expect(screen.getByText('🖼️')).toBeInTheDocument();
    });

    it('should render document file correctly', () => {
      render(<FileAttachmentCard file={fixtures.documentFile} />);
      expect(screen.getByText('📄')).toBeInTheDocument();
    });

    it('should render code file correctly', () => {
      render(<FileAttachmentCard file={fixtures.codeFile} />);
      expect(screen.getByText('💻')).toBeInTheDocument();
    });

    it('should render data file correctly', () => {
      render(<FileAttachmentCard file={fixtures.dataFile} />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have title attribute on file name', () => {
      render(<FileAttachmentCard file={fixtures.imageFile} />);
      const nameElement = screen.getByTitle(fixtures.imageFile.name);
      expect(nameElement).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle very long file names', async () => {
      const { fileUtils } = await import('@/lib/fileUtils');
      const longName = 'a'.repeat(100) + '.txt';
      const longFile = { ...fixtures.documentFile, name: longName };
      
      render(<FileAttachmentCard file={longFile} />);
      
      expect(fileUtils.truncateFileName).toHaveBeenCalledWith(longName);
    });

    it('should handle file with zero size', () => {
      const zeroSizeFile = { ...fixtures.imageFile, size: 0 };
      
      render(<FileAttachmentCard file={zeroSizeFile} />);
      
      expect(screen.getByText(/0 B/)).toBeInTheDocument();
    });

    it('should handle file with very large size', async () => {
      const { fileUtils } = await import('@/lib/fileUtils');
      const largeFile = { ...fixtures.imageFile, size: 100 * 1024 * 1024 };
      
      render(<FileAttachmentCard file={largeFile} />);
      
      expect(fileUtils.formatFileSize).toHaveBeenCalledWith(largeFile.size);
    });

    it('should handle all props combined', async () => {
      const user = userEvent.setup();
      
      render(
        <FileAttachmentCard
          file={fixtures.imageFile}
          showDownload={true}
          showDelete={true}
          onDelete={mockOnDelete}
          compact={true}
        />
      );
      
      expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
      
      await user.click(screen.getByRole('button', { name: /delete/i }));
      expect(mockOnDelete).toHaveBeenCalled();
    });
  });
});
