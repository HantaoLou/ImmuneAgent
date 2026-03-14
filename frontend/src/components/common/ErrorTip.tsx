'use client';

import React from 'react';
import { Alert } from 'antd';
import { CloseCircleOutlined } from '@ant-design/icons';

interface ErrorTipProps {
  message: string;
  onClose?: () => void;
}

export const ErrorTip: React.FC<ErrorTipProps> = ({ message, onClose }) => {
  return (
    <Alert
      message={message}
      type="error"
      icon={<CloseCircleOutlined />}
      showIcon
      closable
      onClose={onClose}
      className="mb-4"
    />
  );
};
