import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { formatTime, formatDate } from '@/utils/format';

describe('format utilities', () => {
  describe('formatTime', () => {
    let originalDate: DateConstructor;

    beforeEach(() => {
      originalDate = global.Date;
    });

    afterEach(() => {
      global.Date = originalDate;
    });

    describe('same day formatting', () => {
      it('should format time as HH:mm:ss for same day', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-15T10:15:30').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        
        expect(result).toMatch(/^\d{2}:\d{2}:\d{2}$/);
        expect(result).toBe('10:15:30');
        
        vi.useRealTimers();
      });

      it('should format midnight correctly', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-15T00:00:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toBe('00:00:00');
        
        vi.useRealTimers();
      });

      it('should format 23:59:59 correctly', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-15T23:59:59').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toBe('23:59:59');
        
        vi.useRealTimers();
      });
    });

    describe('yesterday formatting', () => {
      it('should format as "昨天 HH:mm" for yesterday', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-14T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        
        expect(result).toMatch(/^昨天 \d{2}:\d{2}$/);
        expect(result).toBe('昨天 10:15');
        
        vi.useRealTimers();
      });

      it('should handle yesterday near midnight', () => {
        const now = new Date('2024-01-15T00:30:00');
        const timestamp = new Date('2024-01-14T23:45:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toMatch(/昨天|星期/);
        
        vi.useRealTimers();
      });
    });

    describe('within 7 days formatting', () => {
      it('should format as weekday HH:mm for 2-6 days ago', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-13T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toContain(':');
        expect(result).toMatch(/星期|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday/);
        
        vi.useRealTimers();
      });

      it('should format 3 days ago correctly', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-12T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toContain(':');
        
        vi.useRealTimers();
      });
    });

    describe('older than 7 days formatting', () => {
      it('should format as "MM-DD HH:mm" for dates older than 7 days', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2024-01-07T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        
        expect(result).toMatch(/^\d{2}-\d{2} \d{2}:\d{2}$/);
        
        vi.useRealTimers();
      });

      it('should format last month correctly', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2023-12-25T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toBe('12-25 10:15');
        
        vi.useRealTimers();
      });

      it('should format last year correctly', () => {
        const now = new Date('2024-01-15T14:30:00');
        const timestamp = new Date('2023-01-15T10:15:00').getTime();
        
        vi.useFakeTimers();
        vi.setSystemTime(now);
        
        const result = formatTime(timestamp);
        expect(result).toBe('01-15 10:15');
        
        vi.useRealTimers();
      });
    });
  });

  describe('formatDate', () => {
    it('should format timestamp as YYYY-MM-DD HH:mm:ss', () => {
      const timestamp = new Date('2024-01-15T14:30:45').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-01-15 14:30:45');
    });

    it('should format January 1st correctly', () => {
      const timestamp = new Date('2024-01-01T00:00:00').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-01-01 00:00:00');
    });

    it('should format December 31st correctly', () => {
      const timestamp = new Date('2024-12-31T23:59:59').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-12-31 23:59:59');
    });

    it('should pad single digits with zeros', () => {
      const timestamp = new Date('2024-02-03T05:07:09').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-02-03 05:07:09');
    });

    it('should handle midnight correctly', () => {
      const timestamp = new Date('2024-06-15T00:00:00').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-06-15 00:00:00');
    });

    it('should handle noon correctly', () => {
      const timestamp = new Date('2024-06-15T12:00:00').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-06-15 12:00:00');
    });

    it('should format different years correctly', () => {
      const timestamp2023 = new Date('2023-06-15T12:00:00').getTime();
      const timestamp2025 = new Date('2025-06-15T12:00:00').getTime();
      
      expect(formatDate(timestamp2023)).toBe('2023-06-15 12:00:00');
      expect(formatDate(timestamp2025)).toBe('2025-06-15 12:00:00');
    });

    it('should handle leap year dates', () => {
      const timestamp = new Date('2024-02-29T12:00:00').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2024-02-29 12:00:00');
    });
  });

  describe('edge cases', () => {
    it('should handle very old timestamps', () => {
      const timestamp = new Date('2000-01-01T00:00:00').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2000-01-01 00:00:00');
    });

    it('should handle future timestamps', () => {
      const timestamp = new Date('2030-12-31T23:59:59').getTime();
      
      const result = formatDate(timestamp);
      
      expect(result).toBe('2030-12-31 23:59:59');
    });

    it('should return consistent format for formatDate', () => {
      const timestamp = Date.now();
      const result = formatDate(timestamp);
      
      expect(result).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
    });
  });
});
