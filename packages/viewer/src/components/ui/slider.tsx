import * as SliderPrimitive from '@radix-ui/react-slider';
import type { ComponentPropsWithoutRef, ElementRef } from 'react';
import { forwardRef } from 'react';
import { cn } from '../../lib/utils.ts';

const Slider = forwardRef<
  ElementRef<typeof SliderPrimitive.Root>,
  ComponentPropsWithoutRef<typeof SliderPrimitive.Root> & { ticks?: number }
>(({ className, orientation, ticks, value, ...props }, ref) => {
  const current = value?.[0] ?? 0;
  const min = props.min ?? 0;
  const max = props.max ?? 100;
  return (
    <SliderPrimitive.Root
      ref={ref}
      orientation={orientation}
      {...(value ? { value } : {})}
      className={cn(
        'relative flex touch-none select-none',
        orientation === 'vertical' ? 'h-full flex-col items-center' : 'w-full items-center',
        className,
      )}
      {...props}
    >
      {ticks != null && ticks > 1 && (
        <div
          className={cn(
            'pointer-events-none absolute',
            orientation === 'vertical'
              ? 'inset-y-0 left-1/2 flex -translate-x-1/2 flex-col justify-between'
              : 'inset-x-0 top-1/2 flex -translate-y-1/2 justify-between',
          )}
        >
          {Array.from({ length: ticks }, (_, i) => {
            const tickValue =
              orientation === 'vertical'
                ? max - (i * (max - min)) / (ticks - 1)
                : min + (i * (max - min)) / (ticks - 1);
            const filled = tickValue <= current;
            return (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: static tick marks
                key={i}
                className={cn(
                  'flex h-[9px] w-[9px] items-center justify-center rounded-full',
                  filled ? 'bg-dtour-accent' : 'bg-dtour-border',
                )}
              />
            );
          })}
        </div>
      )}
      <SliderPrimitive.Track
        className={cn(
          'relative grow overflow-hidden rounded-full bg-dtour-border',
          orientation === 'vertical' ? 'w-1.5' : 'h-1.5 w-full',
        )}
      >
        <SliderPrimitive.Range
          className={cn(
            'absolute rounded-full bg-dtour-accent',
            orientation === 'vertical' ? 'w-full' : 'h-full',
          )}
        />
      </SliderPrimitive.Track>
      {ticks != null && ticks > 1 && (
        <div
          className={cn(
            'pointer-events-none absolute',
            orientation === 'vertical'
              ? 'inset-y-0 left-1/2 flex -translate-x-1/2 flex-col justify-between'
              : 'inset-x-0 top-1/2 flex -translate-y-1/2 justify-between',
          )}
        >
          {Array.from({ length: ticks }, (_, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
            <span key={i} className="h-[9px] w-[9px] flex items-center justify-center">
              <span className="h-[3px] w-[3px] rounded-full bg-white/30" />
            </span>
          ))}
        </div>
      )}
      <SliderPrimitive.Thumb className="block h-4 w-4 rounded-full border-2 border-dtour-accent bg-white shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-dtour-accent disabled:pointer-events-none disabled:opacity-50" />
    </SliderPrimitive.Root>
  );
});
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
