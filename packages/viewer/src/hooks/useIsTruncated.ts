import { type RefObject, useEffect, useRef, useState } from 'react';
import { throttleAndDebounce } from '../lib/throttle-debounce.ts';

/**
 * Track whether a single-line, `truncate`-styled element is actually overflowing
 * (i.e. its text is being clipped with an ellipsis). Useful for conditionally
 * showing a tooltip with the full content only when it's truncated.
 *
 * Re-checks on element resize (debounced) and whenever `content` changes, so
 * callers should pass the rendered text/content as the trigger.
 */
export function useIsTruncated<T extends HTMLElement>(
  content: unknown,
): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [isTruncated, setIsTruncated] = useState(false);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `content` is an intentional re-check trigger, not read inside the effect
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const check = () => setIsTruncated(el.scrollWidth > el.clientWidth);
    const debouncedCheck = throttleAndDebounce(check, 100, 150);

    // Initial measurement is synchronous so the tooltip is ready on first hover.
    check();

    const observer = new ResizeObserver(debouncedCheck);
    observer.observe(el);

    return () => {
      observer.disconnect();
      debouncedCheck.cancel();
    };
  }, [content]);

  return [ref, isTruncated];
}
