import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  CompassIcon,
  CursorIcon,
  GaugeIcon,
  MagnifyingGlassMinusIcon,
  PaletteIcon,
  PathIcon,
  PauseIcon,
  PlayIcon,
} from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useRef } from 'react';
import { useAnimatePosition } from '../hooks/useAnimatePosition.ts';
import { usePlayback } from '../hooks/usePlayback.ts';
import {
  cameraZoomAtom,
  grandExitTargetAtom,
  guidedSuspendedAtom,
  metadataAtom,
  pointColorAtom,
  selectedKeyframeAtom,
  tourPlayingAtom,
  tourSpeedAtom,
  viewModeAtom,
} from '../state/atoms.ts';
import { Logo } from './Logo.tsx';
import { Button } from './ui/button.tsx';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu.tsx';
import { Slider } from './ui/slider.tsx';

type ViewMode = 'guided' | 'manual' | 'grand';

const MODE_CONFIG: { mode: ViewMode; label: string; icon: typeof PathIcon }[] = [
  { mode: 'guided', label: 'Guided', icon: PathIcon },
  { mode: 'manual', label: 'Manual', icon: CursorIcon },
  { mode: 'grand', label: 'Grand', icon: CompassIcon },
];

const DEFAULT_COLOR: [number, number, number] = [0.25, 0.5, 0.9];

export type DtourToolbarProps = {
  onLoadData?: ((data: ArrayBuffer, fileName: string) => void) | undefined;
};

export const DtourToolbar = ({ onLoadData }: DtourToolbarProps) => {
  const [playing, setPlaying] = useAtom(tourPlayingAtom);
  const [speed, setSpeed] = useAtom(tourSpeedAtom);
  const [zoom, setZoom] = useAtom(cameraZoomAtom);
  const metadata = useAtomValue(metadataAtom);
  const [viewMode, setViewMode] = useAtom(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setSelectedKeyframe = useSetAtom(selectedKeyframeAtom);
  const [pointColor, setPointColor] = useAtom(pointColorAtom);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Activate the rAF playback loop
  usePlayback();

  const { animateTo, cancelAnimation } = useAnimatePosition();

  const handlePlayPause = useCallback(() => {
    cancelAnimation();
    setGuidedSuspended(false);
    if (!playing) setSelectedKeyframe(null);
    setPlaying((p) => !p);
  }, [playing, setPlaying, setGuidedSuspended, setSelectedKeyframe, cancelAnimation]);

  const handleReset = useCallback(() => {
    setGuidedSuspended(false);
    setPlaying(false);
    animateTo(0);
  }, [setPlaying, setGuidedSuspended, animateTo]);

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !onLoadData) return;
      onLoadData(await file.arrayBuffer(), file.name);
      e.target.value = '';
    },
    [onLoadData],
  );

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Determine active color-by column (string = column name, array = uniform)
  const activeColorColumn = typeof pointColor === 'string' ? pointColor : null;

  const toggleColorBy = useCallback(
    (columnName: string) => {
      setPointColor((prev) => (prev === columnName ? DEFAULT_COLOR : columnName));
    },
    [setPointColor],
  );

  return (
    <div className="grid h-10 grid-cols-[1fr_auto_1fr] items-center border-b border-dtour-surface bg-dtour-bg px-3 text-dtour-text">
      {/* Hidden file input */}
      {onLoadData && (
        <input
          ref={fileInputRef}
          type="file"
          accept=".parquet,.pq,.arrow"
          className="hidden"
          onChange={handleFileSelect}
        />
      )}

      {/* Left: branding + mode switcher */}
      <div className="flex items-center gap-2">
        <div className="relative text-sm font-semibold tracking-wide text-white">
          <div className="opacity-0 pointer-events-none">dtour</div>
          <div className="absolute inset-0">
            <Logo />
          </div>
        </div>
        <div className="ml-2 flex items-center overflow-hidden rounded-md border border-dtour-surface">
          {MODE_CONFIG.map(({ mode, label, icon: Icon }) => (
            <Button
              key={mode}
              variant="ghost"
              size="sm"
              className={`rounded-none ${viewMode === mode ? 'bg-dtour-surface text-white' : 'text-dtour-text-muted'}`}
              onClick={() => {
                if (viewMode === 'grand') {
                  if (mode === 'grand') {
                    setGrandExitTarget(null);
                    return;
                  }
                  setGrandExitTarget(mode);
                } else {
                  if (mode === 'guided' && viewMode !== 'guided') setGuidedSuspended(true);
                  if (mode !== 'guided' && viewMode === 'guided') setPlaying(false);
                  if (mode === 'grand') setGrandExitTarget(null);
                  setViewMode(mode);
                }
              }}
              title={label}
            >
              <Icon size={14} weight={viewMode === mode ? 'fill' : 'regular'} />
              <span className="ml-1 text-xs">{label}</span>
            </Button>
          ))}
        </div>
      </div>

      {/* Center: playback controls (guided mode) / speed (grand mode) */}
      <div className="flex items-center gap-1">
        {viewMode === 'guided' && (
          <>
            <Button variant="ghost" size="icon" onClick={handleReset} title="Reset to start">
              <ArrowCounterClockwiseIcon size={16} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handlePlayPause}
              title={playing ? 'Pause' : 'Play'}
            >
              {playing ? (
                <PauseIcon size={16} weight="fill" />
              ) : (
                <PlayIcon size={16} weight="fill" />
              )}
            </Button>
          </>
        )}

        {(viewMode === 'guided' || viewMode === 'grand') && (
          <Popover.Root>
            <Popover.Trigger asChild>
              <Button variant="ghost" size="icon" title={`Speed: ${speed}x`}>
                <GaugeIcon size={16} />
              </Button>
            </Popover.Trigger>
            <Popover.Portal>
              <Popover.Content
                side="bottom"
                align="center"
                sideOffset={4}
                className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-surface p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
              >
                <span className="text-xs text-dtour-text-muted">Speed</span>
                <Slider
                  orientation="vertical"
                  min={0}
                  max={SPEED_STEPS.length - 1}
                  step={1}
                  value={[speedToStep(speed)]}
                  onValueChange={([step]: number[]) => {
                    if (step !== undefined) setSpeed(stepToSpeed(step));
                  }}
                  className="h-[120px]"
                />
                <span className="text-xs font-medium text-white">{speed}x</span>
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
        )}

        <Popover.Root>
          <Popover.Trigger asChild>
            <Button variant="ghost" size="icon" title={`Zoom: ${zoomToDistance(zoom)}x`}>
              <MagnifyingGlassMinusIcon size={16} />
            </Button>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              side="bottom"
              align="center"
              sideOffset={4}
              className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-surface p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
            >
              <span className="text-xs text-dtour-text-muted">Distance</span>
              <Slider
                orientation="vertical"
                min={0}
                max={DISTANCE_STEPS.length - 1}
                step={1}
                value={[distanceToStep(zoomToDistance(zoom))]}
                onValueChange={([step]: number[]) => {
                  if (step !== undefined) setZoom(1 / stepToDistance(step));
                }}
                className="h-[120px]"
              />
              <span className="text-xs font-medium text-white">{zoomToDistance(zoom)}x</span>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      </div>

      {/* Right: data info + settings */}
      <div className="flex items-center justify-end gap-2">
        {metadata ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                {metadata.rowCount.toLocaleString()} pts &times; {metadata.dimCount} dims
                <CaretDownIcon size={12} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="max-h-[60vh] w-64 overflow-y-auto">
              {/* Numeric columns */}
              {metadata.columnNames.length > 0 && (
                <>
                  <DropdownMenuLabel className="text-xs font-semibold">Numerical</DropdownMenuLabel>
                  {metadata.columnNames.map((col) => (
                    <ColumnRow
                      checked
                      key={col}
                      name={col}
                      dtype="num"
                      isColorActive={activeColorColumn === col}
                      onToggleColor={() => toggleColorBy(col)}
                    />
                  ))}
                </>
              )}

              {/* Categorical columns */}
              {metadata.categoricalColumnNames.length > 0 && (
                <>
                  <DropdownMenuLabel className="text-xs font-semibold">
                    Categorical
                  </DropdownMenuLabel>
                  {metadata.categoricalColumnNames.map((col) => (
                    <ColumnRow
                      key={col}
                      name={col}
                      dtype="cat"
                      isColorActive={activeColorColumn === col}
                      onToggleColor={() => toggleColorBy(col)}
                    />
                  ))}
                </>
              )}

              {onLoadData && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-xs active:scale-[0.97] transition-transform"
                    onSelect={openFilePicker}
                  >
                    Load new data
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : onLoadData ? (
          <Button variant="ghost" size="sm" onClick={openFilePicker}>
            Load data
          </Button>
        ) : (
          <Button variant="ghost" size="sm">
            No data
          </Button>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Column row — a single column entry in the settings dropdown
// ---------------------------------------------------------------------------

const ColumnRow = ({
  name,
  dtype,
  isColorActive,
  onToggleColor,
  checked,
}: {
  name: string;
  dtype: 'num' | 'cat';
  isColorActive: boolean;
  onToggleColor: () => void;
  checked?: boolean;
}) => (
  <DropdownMenuCheckboxItem
    onSelect={(e) => e.preventDefault()}
    className="flex items-center gap-2 pr-1"
    checked={checked ?? false}
  >
    <span className="flex-1 truncate text-xs">{name}</span>
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onToggleColor();
      }}
      className={`shrink-0 cursor-pointer rounded p-1 transition-[color,transform] active:scale-[0.85] ${
        isColorActive ? 'bg-dtour-highlight text-black' : 'text-dtour-text-muted hover:text-white'
      }`}
      title={isColorActive ? `Stop coloring by ${name}` : `Color by ${name}`}
    >
      <PaletteIcon size={12} weight={isColorActive ? 'fill' : 'regular'} />
    </button>
  </DropdownMenuCheckboxItem>
);

// ---------------------------------------------------------------------------
// Speed / distance step helpers
// ---------------------------------------------------------------------------

const SPEED_STEPS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5] as const;

const speedToStep = (speed: number): number => {
  let best = 0;
  let bestDist = Math.abs(speed - SPEED_STEPS[0]!);
  for (let i = 1; i < SPEED_STEPS.length; i++) {
    const dist = Math.abs(speed - SPEED_STEPS[i]!);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
};

const stepToSpeed = (step: number): number => SPEED_STEPS[step] ?? 1;

const DISTANCE_STEPS = [1, 1.25, 1.5, 2, 2.5, 3, 4] as const;

const zoomToDistance = (zoom: number): number => {
  const d = 1 / zoom;
  let best = 0;
  let bestDist = Math.abs(d - DISTANCE_STEPS[0]!);
  for (let i = 1; i < DISTANCE_STEPS.length; i++) {
    const dist = Math.abs(d - DISTANCE_STEPS[i]!);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return DISTANCE_STEPS[best]!;
};

const distanceToStep = (distance: number): number => {
  let best = 0;
  let bestDist = Math.abs(distance - DISTANCE_STEPS[0]!);
  for (let i = 1; i < DISTANCE_STEPS.length; i++) {
    const dist = Math.abs(distance - DISTANCE_STEPS[i]!);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
};

const stepToDistance = (step: number): number => DISTANCE_STEPS[step] ?? 1.25;
