import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileDropZone } from '@/components/files/FileDropZone';

vi.mock('@/components/files/FileUploadButton', () => ({
  FileUploadButton: ({ onFilesSelected, disabled }: any) => (
    <button
      data-testid="upload-button"
      disabled={disabled}
      onClick={() => {
        const file = new File(['test'], 'test.txt', { type: 'text/plain' });
        const fileList = {
          0: file,
          length: 1,
          item: (i: number) => (i === 0 ? file : null),
        } as FileList;
        onFilesSelected(fileList);
      }}
    >
      上传文件
    </button>
  ),
}));

const mockOnFilesSelected = vi.fn();

const createFileList = (files: File[]): FileList => {
  const fileList = {
    length: files.length,
    item: (index: number) => files[index] || null,
    ...files.reduce((acc, file, index) => {
      acc[index] = file;
      return acc;
    }, {} as Record<number, File>),
  };
  return fileList as FileList;
};

describe('FileDropZone Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render drop zone', () => {
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByText('拖拽文件到此处，或')).toBeInTheDocument();
    });

    it('should render file upload button', () => {
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByTestId('upload-button')).toBeInTheDocument();
    });

    it('should render hint text', () => {
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByText(/支持图片、文档、代码、数据文件/)).toBeInTheDocument();
    });

    it('should render custom children when provided', () => {
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        >
          <div>Custom content</div>
        </FileDropZone>
      );
      
      expect(screen.getByText('Custom content')).toBeInTheDocument();
      expect(screen.queryByText('拖拽文件到此处，或')).not.toBeInTheDocument();
    });
  });

  describe('Drag events', () => {
    it('should handle dragEnter event', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      fireEvent.dragEnter(dropZone);
      
      expect(dropZone.className).toMatch(/dragOver/i);
    });

    it('should handle dragLeave event', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      
      fireEvent.dragEnter(dropZone);
      expect(dropZone.className).toMatch(/dragOver/i);
      
      fireEvent.dragLeave(dropZone);
      expect(dropZone.className).not.toMatch(/dragOver/i);
    });

    it('should handle dragOver event', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      fireEvent.dragOver(dropZone);
      
      expect(dropZone).toBeInTheDocument();
    });

    it('should not add dragOver class when disabled', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={true}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      const initialClassName = dropZone.className;
      
      fireEvent.dragEnter(dropZone);
      
      expect(dropZone.className).toBe(initialClassName);
    });
  });

  describe('Drop event', () => {
    it('should call onFilesSelected when files are dropped', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const fileList = createFileList([file]);
      
      fireEvent.drop(dropZone, { dataTransfer: { files: fileList } });
      
      expect(mockOnFilesSelected).toHaveBeenCalledWith(fileList);
    });

    it('should remove dragOver class after drop', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      
      fireEvent.dragEnter(dropZone);
      expect(dropZone.className).toMatch(/dragOver/i);
      
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const fileList = createFileList([file]);
      
      fireEvent.drop(dropZone, { dataTransfer: { files: fileList } });
      expect(dropZone.className).not.toMatch(/dragOver/i);
    });

    it('should not call onFilesSelected when disabled', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={true}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const fileList = createFileList([file]);
      
      fireEvent.drop(dropZone, { dataTransfer: { files: fileList } });
      
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    it('should not call onFilesSelected when no files', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      
      fireEvent.drop(dropZone, { dataTransfer: { files: { length: 0 } as FileList } });
      
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    it('should handle multiple files dropped', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      const file1 = new File(['test1'], 'test1.txt', { type: 'text/plain' });
      const file2 = new File(['test2'], 'test2.txt', { type: 'text/plain' });
      const fileList = createFileList([file1, file2]);
      
      fireEvent.drop(dropZone, { dataTransfer: { files: fileList } });
      
      expect(mockOnFilesSelected).toHaveBeenCalledWith(fileList);
      expect(mockOnFilesSelected.mock.calls[0][0].length).toBe(2);
    });
  });

  describe('FileUploadButton integration', () => {
    it('should pass sessionId to FileUploadButton', () => {
      render(
        <FileDropZone
          sessionId="test-session-123"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByTestId('upload-button')).toBeInTheDocument();
    });

    it('should pass disabled prop to FileUploadButton', () => {
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={true}
        />
      );
      
      expect(screen.getByTestId('upload-button')).toBeDisabled();
    });

    it('should call onFilesSelected when upload button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      await user.click(screen.getByTestId('upload-button'));
      
      expect(mockOnFilesSelected).toHaveBeenCalled();
    });
  });

  describe('Edge cases', () => {
    it('should handle rapid drag enter/leave events', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      
      for (let i = 0; i < 10; i++) {
        fireEvent.dragEnter(dropZone);
        fireEvent.dragLeave(dropZone);
      }
      
      expect(dropZone.className).not.toMatch(/dragOver/i);
    });

    it('should handle drop with empty FileList', () => {
      const { container } = render(
        <FileDropZone
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const dropZone = container.firstChild as HTMLElement;
      const emptyFileList = createFileList([]);
      
      fireEvent.drop(dropZone, { dataTransfer: { files: emptyFileList } });
      
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });
  });
});
