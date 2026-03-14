import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileManager } from '@/components/files/FileManager';
import { fixtures } from '../../setup/fixtures/files';

vi.mock('@/store/sessionStore', () => ({
  useSessionStore: vi.fn((selector) => {
    const state = {
      activeSessionId: 'session-active',
      sessionFiles: {
        'session-active': [
          fixtures.imageFile,
          fixtures.documentFile,
          fixtures.codeFile,
          fixtures.dataFile,
        ],
      },
      removeFile: vi.fn(),
    };
    return selector ? selector(state) : state;
  }),
}));

vi.mock('@/services/fileService', () => ({
  fileService: {
    download: vi.fn().mockResolvedValue(new Blob(['test'])),
    batchDownload: vi.fn().mockResolvedValue(new Blob(['test'])),
    triggerDownload: vi.fn(),
    delete: vi.fn().mockResolvedValue(undefined),
  },
}));

const mockOnClose = vi.fn();

describe('FileManager Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should not render when isOpen is false', () => {
      render(<FileManager isOpen={false} onClose={mockOnClose} />);
      expect(screen.queryByText('文件管理器')).not.toBeInTheDocument();
    });

    it('should render when isOpen is true', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText('文件管理器')).toBeInTheDocument();
    });

    it('should display file count', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/4 个文件/)).toBeInTheDocument();
    });

    it('should display total file size', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/11/)).toBeInTheDocument();
    });

    it('should render search input', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByPlaceholderText('搜索文件...')).toBeInTheDocument();
    });

    it('should render category buttons', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText('全部')).toBeInTheDocument();
      expect(screen.getByText('图片')).toBeInTheDocument();
      expect(screen.getByText('文档')).toBeInTheDocument();
      expect(screen.getByText('代码')).toBeInTheDocument();
      expect(screen.getByText('数据')).toBeInTheDocument();
    });

    it('should render close button', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      const closeBtn = screen.getByRole('button', { name: /close/i });
      expect(closeBtn).toBeInTheDocument();
    });
  });

  describe('Search functionality', () => {
    it('should filter files by search query', async () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText('搜索文件...');
      
      fireEvent.change(searchInput, { target: { value: 'image' } });
      
      await waitFor(() => {
        expect(screen.getByText('test-image.jpg')).toBeInTheDocument();
        expect(screen.queryByText('test-document.pdf')).not.toBeInTheDocument();
      });
    });

    it('should be case insensitive', async () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText('搜索文件...');
      
      fireEvent.change(searchInput, { target: { value: 'IMAGE' } });
      
      await waitFor(() => {
        expect(screen.getByText('test-image.jpg')).toBeInTheDocument();
      });
    });

    it('should show empty state when no matches', async () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText('搜索文件...');
      
      fireEvent.change(searchInput, { target: { value: 'nonexistent' } });
      
      await waitFor(() => {
        expect(screen.getByText('没有找到匹配的文件')).toBeInTheDocument();
      });
    });
  });

  describe('Category filtering', () => {
    it('should filter by image category', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('图片'));
      
      await waitFor(() => {
        expect(screen.getByText('test-image.jpg')).toBeInTheDocument();
        expect(screen.queryByText('test-document.pdf')).not.toBeInTheDocument();
      });
    });

    it('should filter by document category', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('文档'));
      
      await waitFor(() => {
        expect(screen.getByText('test-document.pdf')).toBeInTheDocument();
        expect(screen.queryByText('test-image.jpg')).not.toBeInTheDocument();
      });
    });

    it('should filter by code category', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('代码'));
      
      await waitFor(() => {
        expect(screen.getByText('test-code.ts')).toBeInTheDocument();
        expect(screen.queryByText('test-image.jpg')).not.toBeInTheDocument();
      });
    });

    it('should filter by data category', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('数据'));
      
      await waitFor(() => {
        expect(screen.getByText('test-data.json')).toBeInTheDocument();
        expect(screen.queryByText('test-image.jpg')).not.toBeInTheDocument();
      });
    });

    it('should show all files when "全部" is selected', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('代码'));
      await user.click(screen.getByText('全部'));
      
      await waitFor(() => {
        expect(screen.getByText('test-image.jpg')).toBeInTheDocument();
        expect(screen.getByText('test-document.pdf')).toBeInTheDocument();
        expect(screen.getByText('test-code.ts')).toBeInTheDocument();
        expect(screen.getByText('test-data.json')).toBeInTheDocument();
      });
    });
  });

  describe('Combined filtering', () => {
    it('should combine search and category filters', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('文档'));
      const searchInput = screen.getByPlaceholderText('搜索文件...');
      fireEvent.change(searchInput, { target: { value: 'pdf' } });
      
      await waitFor(() => {
        expect(screen.getByText('test-document.pdf')).toBeInTheDocument();
        expect(screen.queryByText('test-code.ts')).not.toBeInTheDocument();
      });
    });
  });

  describe('Batch download', () => {
    it('should render batch download button when files exist', () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText('批量下载')).toBeInTheDocument();
    });

    it('should call batchDownload when clicked', async () => {
      const { fileService } = await import('@/services/fileService');
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      await user.click(screen.getByText('批量下载'));
      
      await waitFor(() => {
        expect(fileService.batchDownload).toHaveBeenCalled();
      });
    });
  });

  describe('Close functionality', () => {
    it('should call onClose when close button is clicked', async () => {
      const user = userEvent.setup();
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      const closeButtons = screen.getAllByRole('button');
      await user.click(closeButtons[0]);
      
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('Empty state', () => {
    it('should show empty state when no session is active', async () => {
      const { useSessionStore } = await import('@/store/sessionStore');
      vi.mocked(useSessionStore).mockImplementation((selector: any) => {
        const state = {
          activeSessionId: null,
          sessionFiles: {},
          removeFile: vi.fn(),
        };
        return selector ? selector(state) : state;
      });

      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      expect(screen.getByText('暂无文件')).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    it('should handle special characters in search', async () => {
      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText('搜索文件...');
      
      fireEvent.change(searchInput, { target: { value: '!@#$%^&*()' } });
      
      expect(screen.getByText('没有找到匹配的文件')).toBeInTheDocument();
    });

    it('should handle files with same name', async () => {
      const { useSessionStore } = await import('@/store/sessionStore');
      vi.mocked(useSessionStore).mockImplementation((selector: any) => {
        const state = {
          activeSessionId: 'session-active',
          sessionFiles: {
            'session-active': [
              { ...fixtures.imageFile, id: 'file-1', name: 'same-name.jpg' },
              { ...fixtures.imageFile, id: 'file-2', name: 'same-name.jpg' },
            ],
          },
          removeFile: vi.fn(),
        };
        return selector ? selector(state) : state;
      });

      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      const sameNameElements = screen.getAllByText('same-name.jpg');
      expect(sameNameElements).toHaveLength(2);
    });

    it('should handle very long file names', async () => {
      const { useSessionStore } = await import('@/store/sessionStore');
      const longName = 'a'.repeat(200) + '.txt';
      vi.mocked(useSessionStore).mockImplementation((selector: any) => {
        const state = {
          activeSessionId: 'session-active',
          sessionFiles: {
            'session-active': [
              { ...fixtures.documentFile, id: 'file-long', name: longName },
            ],
          },
          removeFile: vi.fn(),
        };
        return selector ? selector(state) : state;
      });

      render(<FileManager isOpen={true} onClose={mockOnClose} />);
      
      expect(screen.getByTitle(longName)).toBeInTheDocument();
    });
  });
});
