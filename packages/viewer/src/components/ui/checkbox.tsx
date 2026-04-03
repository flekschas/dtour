import type { ComponentPropsWithoutRef } from 'react';
import { cn } from '../../lib/utils.ts';

const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 14 14" fill="none" aria-hidden="true">
    <path
      d="M11.5 3.5L5.5 9.5L2.5 6.5"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export const Checkbox = ({
  checked,
  onCheckedChange,
  className,
  ...props
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
} & Omit<ComponentPropsWithoutRef<'button'>, 'onClick' | 'role'>) => (
  <button
    type="button"
    // biome-ignore lint/a11y/useSemanticElements: <explanation>
    role="checkbox"
    aria-checked={checked}
    onClick={() => onCheckedChange(!checked)}
    className={cn(
      'inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border transition-colors',
      checked
        ? 'border-dtour-highlight bg-dtour-highlight text-dtour-bg'
        : 'border-dtour-text-muted bg-transparent text-transparent',
      className,
    )}
    {...props}
  >
    <CheckIcon />
  </button>
);
