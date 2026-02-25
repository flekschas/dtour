import * as SliderPrimitive from '@radix-ui/react-slider';
import type { ComponentPropsWithoutRef, ElementRef } from 'react';
import { forwardRef } from 'react';
import { cn } from '../../lib/utils.ts';

const Slider = forwardRef<
  ElementRef<typeof SliderPrimitive.Root>,
  ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, orientation, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    orientation={orientation}
    className={cn(
      'relative flex touch-none select-none',
      orientation === 'vertical'
        ? 'h-full flex-col items-center'
        : 'w-full items-center',
      className,
    )}
    {...props}
  >
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
    <SliderPrimitive.Thumb className="block h-4 w-4 rounded-full border-2 border-dtour-accent bg-white shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-dtour-accent disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
