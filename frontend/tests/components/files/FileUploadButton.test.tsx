import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileUploadButton } from '@/components/files/FileUploadButton';

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

describe('FileUploadButton Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render upload button', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByRole('button', { name: /上传文件/i })).toBeInTheDocument();
    });

    it('should render hidden file input', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]');
      expect(input).toBeInTheDocument();
    });

    it('should render UploadOutlined icon', () => {
      const { container } = render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(container.querySelector('.anticon-upload')).toBeInTheDocument();
    });
  });

  describe('File input configuration', () => {
    it('should have multiple attribute by default', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input.multiple).toBe(true);
    });

    it('should have multiple=false when multiple prop is false', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          multiple={false}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input.multiple).toBe(false);
    });

    it('should have accept attribute when provided', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          accept=".pdf,.doc,.docx"
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input.accept).toBe('.pdf,.doc,.docx');
    });

    it('should not have accept attribute when not provided', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input.accept).toBe('');
    });
  });

  describe('Button click behavior', () => {
    it('should trigger file input click when button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const clickSpy = vi.spyOn(input, 'click');
      
      await user.click(screen.getByRole('button', { name: /上传文件/i }));
      
      expect(clickSpy).toHaveBeenCalled();
    });

    it('should not trigger input click when disabled', async () => {
      const user = userEvent.setup();
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={true}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const clickSpy = vi.spyOn(input, 'click');
      
      await user.click(screen.getByRole('button', { name: /上传文件/i }));
      
      expect(clickSpy).not.toHaveBeenCalled();
    });

    it('should not trigger input click when no sessionId', async () => {
      const user = userEvent.setup();
      render(
        <FileUploadButton
          sessionId=""
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const clickSpy = vi.spyOn(input, 'click');
      
      await user.click(screen.getByRole('button', { name: /上传文件/i }));
      
      expect(clickSpy).not.toHaveBeenCalled();
    });
  });

  describe('File selection', () => {
    it('should call onFilesSelected when files are selected', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const fileList = createFileList([file]);
      
      fireEvent.change(input, { target: { files: fileList } });
      
      expect(mockOnFilesSelected).toHaveBeenCalledWith(fileList);
    });

    it('should not call onFilesSelected when no files', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      
      fireEvent.change(input, { target: { files: null } });
      
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    it('should handle multiple files', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file1 = new File(['test1'], 'test1.txt', { type: 'text/plain' });
      const file2 = new File(['test2'], 'test2.txt', { type: 'text/plain' });
      const fileList = createFileList([file1, file2]);
      
      fireEvent.change(input, { target: { files: fileList } });
      
      expect(mockOnFilesSelected).toHaveBeenCalledWith(fileList);
      expect(mockOnFilesSelected.mock.calls[0][0].length).toBe(2);
    });

    it('should reset input value after selection', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      const fileList = createFileList([file]);
      
      fireEvent.change(input, { target: { files: fileList } });
      
      expect(input.value).toBe('');
    });
  });

  describe('Disabled state', () => {
    it('should be disabled when disabled prop is true', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={true}
        />
      );
      
      expect(screen.getByRole('button', { name: /上传文件/i })).toBeDisabled();
    });

    it('should be disabled when sessionId is empty', () => {
      render(
        <FileUploadButton
          sessionId=""
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      expect(screen.getByRole('button', { name: /上传文件/i })).toBeDisabled();
    });

    it('should not be disabled when sessionId is valid and disabled is false', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
          disabled={false}
        />
      );
      
      expect(screen.getByRole('button', { name: /上传文件/i })).not.toBeDisabled();
    });
  });

  describe('Edge cases', () => {
    it('should handle rapid clicks', async () => {
      const user = userEvent.setup();
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const button = screen.getByRole('button', { name: /上传文件/i });
      
      await user.click(button);
      await user.click(button);
      await user.click(button);
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input).toBeInTheDocument();
    });

    it('should handle empty FileList', () => {
      render(
        <FileUploadButton
          sessionId="session-1"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      const emptyFileList = createFileList([]);
      
      fireEvent.change(input, { target: { files: emptyFileList } });
      
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    it('should handle sessionId with special characters', async () => {
      const user = userEvent.setup();
      render(
        <FileUploadButton
          sessionId="session-@#$%-123"
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      const button = screen.getByRole('button', { name: /上传文件/i });
      expect(button).not.toBeDisabled();
    });
  });
});
