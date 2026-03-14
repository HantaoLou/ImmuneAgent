import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);

export const formatTime = (timestamp: number): string => {
  const date = dayjs(timestamp);
  const now = dayjs();
  const diffDays = now.diff(date, 'day');

  if (diffDays === 0) {
    return date.format('HH:mm:ss');
  } else if (diffDays === 1) {
    return '昨天 ' + date.format('HH:mm');
  } else if (diffDays < 7) {
    return date.format('dddd HH:mm');
  } else {
    return date.format('MM-DD HH:mm');
  }
};

export const formatDate = (timestamp: number): string => {
  return dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss');
};
