import React from 'react';
import clsx from 'clsx';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'bordered' | 'elevated';
}

export function Card({ children, variant = 'default', className, ...props }: CardProps) {
  const variants = {
    default: 'bg-white',
    bordered: 'bg-white border border-gray-200',
    elevated: 'bg-white shadow-lg',
  };

  return (
    <div className={clsx('rounded-lg p-4', variants[variant], className)} {...props}>
      {children}
    </div>
  );
}
