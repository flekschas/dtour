import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import { dataNameAtom, pointColorAtom } from '../state/atoms.ts';
import { loadSettings, saveSettings } from '../state/settings-persistence.ts';

/**
 * Always-on persistence: syncs point color to/from localStorage keyed by dataName.
 *
 * - On dataName change: restores saved color (if any)
 * - On pointColor change: saves to localStorage
 */
export const useSettingsPersistence = () => {
  const dataName = useAtomValue(dataNameAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const setPointColor = useSetAtom(pointColorAtom);

  // Track whether restore has fired for this dataName to avoid
  // save-on-initial-restore loops.
  const restoredRef = useRef<string | null>(null);

  // Restore settings when dataName changes
  useEffect(() => {
    if (!dataName) return;
    if (restoredRef.current === dataName) return;

    restoredRef.current = dataName;
    const saved = loadSettings(dataName);
    if (saved) {
      setPointColor(saved.pointColor);
    }
  }, [dataName, setPointColor]);

  // Save settings when pointColor changes
  useEffect(() => {
    if (!dataName) return;
    // Skip saving until restore has completed for this dataName
    if (restoredRef.current !== dataName) return;

    saveSettings(dataName, { pointColor });
  }, [pointColor, dataName]);
};
