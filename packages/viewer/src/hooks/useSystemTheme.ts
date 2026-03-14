import { useSetAtom } from 'jotai';
import { useEffect } from 'react';
import { systemThemeAtom } from '../state/atoms.ts';

/**
 * Subscribes to the OS-level color scheme preference and updates
 * `systemThemeAtom`. Called once in DtourInner.
 */
export const useSystemTheme = () => {
  const setSystemTheme = useSetAtom(systemThemeAtom);

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const update = () => setSystemTheme(mql.matches ? 'dark' : 'light');
    update();
    mql.addEventListener('change', update);
    return () => mql.removeEventListener('change', update);
  }, [setSystemTheme]);
};
