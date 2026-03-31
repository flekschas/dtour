/**
 * Throttle and debounce a function call.
 *
 * Throttling ensures the function is called at most every `throttleTime` ms.
 * Debouncing ensures a final call happens after `debounceTime` ms of silence.
 * Combined, this gives periodic updates during rapid firing AND a guaranteed
 * final call with the latest arguments.
 *
 * Example with throttleTime=3 and debounceTime=3:
 *   1. call(args1) => fn(args1) called immediately
 *   2. call(args2) => ignored (throttled)
 *   3. call(args3) => ignored (throttled)
 *   4. call(args4) => fn(args4) called (throttle window expired)
 *   5. call(args5) => ignored (throttled)
 *   6. (silence)
 *   7. fn(args5) called (debounce fires)
 */
export type ThrottledAndDebouncedFunction<Input extends unknown[]> = {
  (...args: Input): void;
  /** Cancel the pending debounce timer. */
  cancel: () => void;
  /** Reset throttle state so the next call fires immediately. */
  reset: () => void;
  /** Bypass throttle/debounce and call immediately. */
  now: (...args: Input) => void;
};

export const throttleAndDebounce = <Input extends unknown[]>(
  fn: (...args: Input) => void,
  throttleTime: number,
  debounceTime?: number,
): ThrottledAndDebouncedFunction<Input> => {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let blockedCalls = 0;

  const finalWait = debounceTime ?? throttleTime;

  const debounced = (...args: Input) => {
    const later = () => {
      if (blockedCalls > 0) {
        fn(...args);
        blockedCalls = 0;
      }
    };

    clearTimeout(timeout);
    timeout = setTimeout(later, finalWait);
  };

  let isWaiting = false;
  const throttledAndDebounced = (...args: Input) => {
    if (isWaiting) {
      blockedCalls++;
      debounced(...args);
    } else {
      fn(...args);
      debounced(...args);

      isWaiting = true;
      blockedCalls = 0;

      setTimeout(() => {
        isWaiting = false;
      }, throttleTime);
    }
  };

  throttledAndDebounced.reset = () => {
    isWaiting = false;
  };

  throttledAndDebounced.cancel = () => {
    clearTimeout(timeout);
  };

  throttledAndDebounced.now = (...args: Input) => fn(...args);

  return throttledAndDebounced;
};
