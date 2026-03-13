import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import { activeColumnsAtom, dataNameAtom, pointColorAtom, showLegendAtom } from '../state/atoms.ts';
import { loadSettings, saveSettings } from '../state/settings-persistence.ts';

/**
 * Always-on persistence: syncs point color and active columns
 * to/from localStorage keyed by dataName.
 *
 * - On dataName change: restores saved settings (if any)
 * - On pointColor or activeColumns change: saves to localStorage
 */
export const useSettingsPersistence = () => {
  const dataName = useAtomValue(dataNameAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const setPointColor = useSetAtom(pointColorAtom);
  const activeColumns = useAtomValue(activeColumnsAtom);
  const setActiveColumns = useSetAtom(activeColumnsAtom);
  const showLegend = useAtomValue(showLegendAtom);
  const setShowLegend = useSetAtom(showLegendAtom);

  // Track whether restore has fired for this dataName to avoid
  // save-on-initial-restore loops.
  const restoredRef = useRef<string | null>(null);
  // Flag to skip the save effect that fires in the same commit as a restore.
  // Without this, the save effect captures stale (default) values from the
  // pre-restore render and overwrites localStorage before the restored
  // values have propagated through React's state update batching.
  const skipNextSaveRef = useRef(false);

  // Restore settings when dataName changes
  useEffect(() => {
    if (!dataName) return;
    if (restoredRef.current === dataName) return;

    restoredRef.current = dataName;
    const saved = loadSettings(dataName);
    if (saved) {
      skipNextSaveRef.current = true;
      setPointColor(saved.pointColor);
      if (saved.activeColumns !== undefined) {
        setActiveColumns(saved.activeColumns === null ? null : new Set(saved.activeColumns));
      }
      if (saved.showLegend !== undefined) {
        setShowLegend(saved.showLegend);
      }
    }
  }, [dataName, setPointColor, setActiveColumns, setShowLegend]);

  // Save settings when pointColor or activeColumns changes
  useEffect(() => {
    if (!dataName) return;
    // Skip saving until restore has completed for this dataName
    if (restoredRef.current !== dataName) return;
    // Skip the stale save that fires in the same commit as restore
    if (skipNextSaveRef.current) {
      skipNextSaveRef.current = false;
      return;
    }

    saveSettings(dataName, {
      pointColor,
      activeColumns:
        activeColumns === null ? null : Array.from(activeColumns).sort((a, b) => a - b),
      showLegend,
    });
  }, [pointColor, activeColumns, showLegend, dataName]);
};
