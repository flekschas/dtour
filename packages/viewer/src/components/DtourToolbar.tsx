import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  ChartScatterIcon,
  CompassIcon,
  CursorIcon,
  GaugeIcon,
  ImageSquareIcon,
  MagnifyingGlassMinusIcon,
  MonitorIcon,
  MoonIcon,
  PaintBrushIcon,
  PathIcon,
  PauseIcon,
  PlayIcon,
  SidebarSimpleIcon,
  SlidersHorizontalIcon,
  SunIcon,
  TagIcon,
} from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAnimatePosition } from '../hooks/useAnimatePosition.ts';
import { usePortalContainer } from '../portal-container.tsx';
import {
  activeColumnsAtom,
  cameraZoomAtom,
  frameLoadingsAtom,
  grandExitTargetAtom,
  guidedSuspendedAtom,
  legendVisibleAtom,
  metadataAtom,
  pointColorAtom,
  previewCountAtom,
  previewScaleAtom,
  selectedKeyframeAtom,
  showAxesAtom,
  showFrameLoadingsAtom,
  showFrameNumbersAtom,
  showLegendAtom,
  showTourDescriptionAtom,
  sliderSpacingAtom,
  themeModeAtom,
  tourByAtom,
  tourPlayingAtom,
  tourSpeedAtom,
  viewModeAtom,
} from '../state/atoms.ts';
import { Logo } from './Logo.tsx';
import { Button } from './ui/button.tsx';
import { Checkbox } from './ui/checkbox.tsx';
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
  onLogoClick?: (() => void) | undefined;
};

export const DtourToolbar = ({ onLoadData, onLogoClick }: DtourToolbarProps) => {
  const [playing, setPlaying] = useAtom(tourPlayingAtom);
  const [speed, setSpeed] = useAtom(tourSpeedAtom);
  const [zoom, setZoom] = useAtom(cameraZoomAtom);
  const metadata = useAtomValue(metadataAtom);
  const [viewMode, setViewMode] = useAtom(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setSelectedKeyframe = useSetAtom(selectedKeyframeAtom);
  const [pointColor, setPointColor] = useAtom(pointColorAtom);
  const [activeColumns, setActiveColumns] = useAtom(activeColumnsAtom);
  const [previewCount, setPreviewCount] = useAtom(previewCountAtom);
  const [previewScale, setPreviewScale] = useAtom(previewScaleAtom);
  const [showLegend, setShowLegend] = useAtom(showLegendAtom);
  const legendVisible = useAtomValue(legendVisibleAtom);
  const [themeMode, setThemeMode] = useAtom(themeModeAtom);
  const [showAxes, setShowAxes] = useAtom(showAxesAtom);
  const [showFrameNumbers, setShowFrameNumbers] = useAtom(showFrameNumbersAtom);
  const [showFrameLoadings, setShowFrameLoadings] = useAtom(showFrameLoadingsAtom);
  const hasFrameLoadings = useAtomValue(frameLoadingsAtom) !== null;
  const [tourBy, setTourBy] = useAtom(tourByAtom);
  const [sliderSpacing, setSliderSpacing] = useAtom(sliderSpacingAtom);
  const [showTourDescription, setShowTourDescription] = useAtom(showTourDescriptionAtom);

  const portalContainer = usePortalContainer();
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const activeCount =
    activeColumns === null ? (metadata?.columnNames.length ?? 0) : activeColumns.size;

  const handleToggleColumn = useCallback(
    (dimIndex: number) => {
      setActiveColumns((prev) => {
        const current =
          prev ?? new Set(Array.from({ length: metadata?.dimCount ?? 0 }, (_, i) => i));
        const next = new Set(current);
        if (next.has(dimIndex)) {
          if (next.size <= 2) return prev;
          next.delete(dimIndex);
        } else {
          next.add(dimIndex);
        }
        // Optimize: return null when all columns are active
        if (metadata && next.size === metadata.dimCount) return null;
        return next;
      });
    },
    [metadata, setActiveColumns],
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
        {onLogoClick ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onLogoClick}
            className="-ml-1 -mr-1   relative font-semibold tracking-wide text-dtour-highlight"
          >
            <div className="opacity-0 px-2 pointer-events-none">dtour</div>
            <div
              className="absolute top-0 left-2 bottom-0 right-2 flex items-center justify-center"
              data-logo-target
            >
              <Logo />
            </div>
          </Button>
        ) : (
          <div className="relative text-sm font-semibold tracking-wide text-dtour-highlight">
            <div className="opacity-0 pointer-events-none">dtour</div>
            <div className="absolute inset-0" data-logo-target>
              <Logo />
            </div>
          </div>
        )}
        <div className="ml-2 flex items-center overflow-hidden rounded-md border border-dtour-surface">
          {/* Guided button — expands to include Dims/PCA sub-toggle when active */}
          <div
            className={`flex gap-0 items-center ${viewMode === 'guided' ? 'bg-dtour-surface text-dtour-highlight' : 'text-dtour-text-muted'}`}
          >
            <Button
              variant="ghost"
              size="sm"
              className={`rounded-none ${viewMode === 'guided' ? 'text-dtour-highlight' : ''}`}
              onClick={() => {
                if (viewMode === 'grand') {
                  setGrandExitTarget('guided');
                } else if (viewMode !== 'guided') {
                  setGuidedSuspended(true);
                  setViewMode('guided');
                }
              }}
              title="Guided"
            >
              <PathIcon size={14} weight={viewMode === 'guided' ? 'fill' : 'regular'} />
              <span className="ml-1 text-xs">Guided{viewMode === 'guided' ? ':' : ''}</span>
            </Button>
            {viewMode === 'guided' && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  className={`rounded-none px-0 ${tourBy === 'dimensions' ? 'text-dtour-highlight' : 'text-dtour-text-muted'}`}
                  onClick={() => setTourBy('dimensions')}
                  title="Tour by dimensions"
                >
                  <span className="text-xs">Dims</span>
                </Button>
                <span className="text-[10px] text-dtour-text-muted select-none px-1.5">/</span>
                <Button
                  variant="ghost"
                  size="sm"
                  className={`rounded-none px-0 ${tourBy === 'pca' ? 'text-dtour-highlight' : 'text-dtour-text-muted'}`}
                  onClick={() => setTourBy('pca')}
                  title="Tour by principal components"
                >
                  <span className="text-xs">PCA</span>
                </Button>
                <div className="w-1.5 h-full text-[10px] text-dtour-text-muted select-none" />
              </>
            )}
          </div>
          {/* Manual + Grand buttons */}
          {MODE_CONFIG.filter(({ mode }) => mode !== 'guided').map(
            ({ mode, label, icon: Icon }) => (
              <Button
                key={mode}
                variant="ghost"
                size="sm"
                className={`rounded-none ${viewMode === mode ? 'bg-dtour-surface text-dtour-highlight' : 'text-dtour-text-muted'}`}
                onClick={() => {
                  if (viewMode === 'grand') {
                    if (mode === 'grand') {
                      setGrandExitTarget(null);
                      return;
                    }
                    setGrandExitTarget(mode);
                  } else {
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
            ),
          )}
        </div>
      </div>

      {/* Center: playback controls (guided mode) / speed (grand mode) */}
      <div className="flex items-center gap-1">
        {viewMode === 'guided' && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" title="Tour settings">
                <SlidersHorizontalIcon size={16} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="center">
              <DropdownMenuItem
                onSelect={(e) => {
                  e.preventDefault();
                  setSliderSpacing(sliderSpacing === 'equal' ? 'geodesic' : 'equal');
                }}
              >
                <Checkbox
                  checked={sliderSpacing === 'geodesic'}
                  onCheckedChange={() =>
                    setSliderSpacing(sliderSpacing === 'equal' ? 'geodesic' : 'equal')
                  }
                />
                <span className="text-xs">Geodesic spacing</span>
              </DropdownMenuItem>
              {hasFrameLoadings && (
                <DropdownMenuItem
                  onSelect={(e) => {
                    e.preventDefault();
                    setShowTourDescription((v) => !v);
                  }}
                >
                  <Checkbox
                    checked={showTourDescription}
                    onCheckedChange={() => setShowTourDescription((v) => !v)}
                  />
                  <span className="text-xs">Tour description</span>
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        <Popover.Root>
          <Popover.Trigger asChild>
            <Button variant="ghost" size="icon" title={`Zoom: ${zoomToDistance(zoom)}x`}>
              <MagnifyingGlassMinusIcon size={16} />
            </Button>
          </Popover.Trigger>
          <Popover.Portal container={portalContainer}>
            <Popover.Content
              side="bottom"
              align="center"
              sideOffset={4}
              className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-surface p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
            >
              <span className="text-xs text-center font-semibold text-dtour-text-muted">Zoom</span>
              <Slider
                orientation="vertical"
                min={0}
                max={DISTANCE_STEPS.length - 1}
                step={1}
                ticks={DISTANCE_STEPS.length}
                value={[distanceToStep(zoomToDistance(zoom))]}
                onValueChange={([step]: number[]) => {
                  if (step !== undefined) setZoom(1 / stepToDistance(step));
                }}
                className="h-[120px]"
              />
              <span className="text-xs font-medium text-dtour-highlight">
                {zoomToDistance(zoom)}x
              </span>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>

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
            <Popover.Portal container={portalContainer}>
              <Popover.Content
                side="bottom"
                align="center"
                sideOffset={4}
                className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-surface p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
              >
                <div className="text-xs text-center font-semibold text-dtour-text-muted">Speed</div>
                <Slider
                  orientation="vertical"
                  min={0}
                  max={SPEED_STEPS.length - 1}
                  step={1}
                  ticks={SPEED_STEPS.length}
                  value={[speedToStep(speed)]}
                  onValueChange={([step]: number[]) => {
                    if (step !== undefined) setSpeed(stepToSpeed(step));
                  }}
                  className="h-[120px]"
                />
                <span className="text-xs font-medium text-dtour-highlight">{speed}x</span>
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
        )}

        {viewMode === 'guided' && (
          <Popover.Root>
            <Popover.Trigger asChild>
              <Button variant="ghost" size="icon" title={`Previews: ${previewCount}`}>
                <ImageSquareIcon size={16} />
              </Button>
            </Popover.Trigger>
            <Popover.Portal container={portalContainer}>
              <Popover.Content
                side="bottom"
                align="center"
                sideOffset={4}
                className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-surface p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
              >
                <div className="flex flex-col gap-2">
                  <div className="text-xs text-center font-semibold text-dtour-text-muted">
                    Preview
                  </div>
                  <div className="flex gap-4">
                    <div className="flex flex-col items-center gap-2">
                      <span className="text-xs text-dtour-text-muted">Count</span>
                      <PreviewStepSlider
                        steps={PREVIEW_COUNT_STEPS}
                        value={previewCount}
                        onCommit={setPreviewCount}
                      />
                    </div>
                    <div className="flex flex-col items-center gap-2">
                      <span className="text-xs text-dtour-text-muted">Size</span>
                      <PreviewStepSlider
                        steps={PREVIEW_SCALE_STEPS}
                        value={previewScale}
                        onCommit={setPreviewScale}
                        formatLabel={SCALE_LABELS}
                      />
                    </div>
                  </div>
                  <div
                    className="flex items-center gap-1.5 cursor-pointer select-none pt-1"
                    onClick={() => setShowFrameNumbers((v) => !v)}
                    onKeyDown={undefined}
                  >
                    <Checkbox
                      checked={showFrameNumbers}
                      onCheckedChange={() => setShowFrameNumbers((v) => !v)}
                    />
                    <span className="text-xs text-dtour-text-muted">Numbers</span>
                  </div>
                </div>
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
        )}

        {viewMode === 'guided' && (
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowAxes((v) => !v)}
              title={showAxes ? 'Hide axes' : 'Show axes'}
              className={showAxes ? '' : 'opacity-40'}
            >
              <ChartScatterIcon size={16} weight={showAxes ? 'fill' : 'regular'} />
            </Button>
            {hasFrameLoadings && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowFrameLoadings((v) => !v)}
                title={showFrameLoadings ? 'Hide feature loadings' : 'Show feature loadings'}
                className={showFrameLoadings ? '' : 'opacity-40'}
              >
                <TagIcon size={16} weight={showFrameLoadings ? 'fill' : 'regular'} />
              </Button>
            )}
          </>
        )}
      </div>

      {/* Right: data info + settings */}
      <div className="flex items-center justify-end gap-2">
        {metadata ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                {metadata.rowCount.toLocaleString()} pts &times;{' '}
                {activeCount === metadata.dimCount
                  ? `${metadata.dimCount} dims`
                  : `${activeCount}/${metadata.dimCount} dims`}
                <CaretDownIcon size={12} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="max-h-[60vh] w-64 overflow-y-auto">
              {/* Numeric columns */}
              {metadata.columnNames.length > 0 && (
                <>
                  <DropdownMenuLabel className="text-xs font-semibold">Numerical</DropdownMenuLabel>
                  {metadata.columnNames.map((col, index) => {
                    const isActive = activeColumns === null || activeColumns.has(index);
                    return (
                      <ColumnRow
                        key={col}
                        name={col}
                        dtype="num"
                        checked={isActive}
                        onCheckedChange={() => handleToggleColumn(index)}
                        disabled={isActive && activeCount <= 2}
                        isColorActive={activeColorColumn === col}
                        onToggleColor={() => toggleColorBy(col)}
                      />
                    );
                  })}
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
        <Button
          variant="ghost"
          size="icon"
          onClick={() =>
            setThemeMode((m) => (m === 'dark' ? 'light' : m === 'light' ? 'system' : 'dark'))
          }
          title={`Theme: ${themeMode === 'dark' ? 'Dark' : themeMode === 'light' ? 'Light' : 'System'}`}
        >
          {themeMode === 'dark' ? (
            <MoonIcon size={16} weight="fill" />
          ) : themeMode === 'light' ? (
            <SunIcon size={16} weight="fill" />
          ) : (
            <MonitorIcon size={16} weight="fill" />
          )}
        </Button>
        {activeColorColumn && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setShowLegend((v) => !v)}
            title={showLegend ? 'Hide legend' : 'Show legend'}
            className={legendVisible || showLegend ? '' : 'opacity-40'}
          >
            <SidebarSimpleIcon size={16} weight={showLegend ? 'fill' : 'regular'} />
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
  onCheckedChange,
  disabled,
}: {
  name: string;
  dtype: 'num' | 'cat';
  isColorActive: boolean;
  onToggleColor: () => void;
  checked?: boolean;
  onCheckedChange?: () => void;
  disabled?: boolean;
}) => (
  <DropdownMenuCheckboxItem
    onSelect={(e) => e.preventDefault()}
    className="flex items-center gap-2 pr-1"
    checked={checked ?? false}
    {...(onCheckedChange ? { onCheckedChange } : {})}
    {...(disabled ? { disabled } : {})}
  >
    <span className="flex-1 truncate text-xs">{name}</span>
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onToggleColor();
      }}
      className={`shrink-0 cursor-pointer rounded p-1 transition-[color,transform] active:scale-[0.85] ${
        isColorActive
          ? 'bg-dtour-highlight text-dtour-bg'
          : 'text-dtour-text-muted hover:text-dtour-highlight'
      }`}
      title={isColorActive ? `Stop coloring by ${name}` : `Color by ${name}`}
    >
      <PaintBrushIcon size={12} weight={isColorActive ? 'fill' : 'regular'} />
    </button>
  </DropdownMenuCheckboxItem>
);

// ---------------------------------------------------------------------------
// Preview step slider — generic discrete slider with local drag state
// ---------------------------------------------------------------------------

const PREVIEW_COUNT_STEPS: (4 | 8 | 12 | 16)[] = [4, 8, 12, 16];
const PREVIEW_SCALE_STEPS: (1 | 0.75 | 0.5)[] = [0.5, 0.75, 1];
const SCALE_LABELS: Record<number, string> = { 1: 'L', 0.75: 'M', 0.5: 'S' };

function PreviewStepSlider<T extends number>({
  steps,
  value,
  onCommit,
  formatLabel,
}: {
  steps: T[];
  value: T;
  onCommit: (v: T) => void;
  formatLabel?: Record<number, string>;
}) {
  const [localStep, setLocalStep] = useState(() => steps.indexOf(value));

  // Resync local state when value changes externally (e.g. restored settings, spec updates)
  useEffect(() => {
    const idx = steps.indexOf(value);
    if (idx !== -1) setLocalStep(idx);
  }, [value, steps]);

  const display = formatLabel?.[steps[localStep]!] ?? String(steps[localStep] ?? value);

  return (
    <>
      <Slider
        orientation="vertical"
        min={0}
        max={steps.length - 1}
        step={1}
        ticks={steps.length}
        value={[localStep]}
        onValueChange={([step]: number[]) => {
          if (step !== undefined) setLocalStep(step);
        }}
        onValueCommit={([step]: number[]) => {
          if (step !== undefined) onCommit(steps[step]!);
        }}
        className="h-[120px]"
      />
      <span className="text-xs font-medium text-dtour-highlight">{display}</span>
    </>
  );
}

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
