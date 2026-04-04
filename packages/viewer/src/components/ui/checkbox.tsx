import { CheckIcon } from '@phosphor-icons/react';
import type { ComponentPropsWithoutRef } from 'react';
import { cn } from '../../lib/utils.ts';

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
    onClick={(e) => {
      e.stopPropagation();
      onCheckedChange(!checked);
    }}
    className={cn(
      'inline-flex h-4 w-4 shrink-0 items-center justify-center transition-colors',
      checked ? 'text-dtour-text' : 'text-transparent',
      className,
    )}
    {...props}
  >
    <CheckIcon size={14} weight="bold" />
  </button>
);
