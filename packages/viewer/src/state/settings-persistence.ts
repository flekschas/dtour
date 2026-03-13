const STORAGE_PREFIX = 'dtour:';

type PersistedSettings = {
  pointColor: [number, number, number] | string;
  activeColumns?: number[] | null;
  showLegend?: boolean;
};

export const saveSettings = (dataName: string, settings: PersistedSettings): void => {
  try {
    localStorage.setItem(STORAGE_PREFIX + dataName, JSON.stringify(settings));
  } catch {
    // localStorage may be full or unavailable
  }
};

export const loadSettings = (dataName: string): PersistedSettings | null => {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + dataName);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedSettings;
  } catch {
    return null;
  }
};
