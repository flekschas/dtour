import {
  ArrowsCounterClockwiseIcon,
  CaretDownIcon,
  ChartScatterIcon,
  CompassIcon,
  CursorIcon,
  DatabaseIcon,
  GaugeIcon,
  MagnifyingGlassIcon,
  MonitorIcon,
  MoonIcon,
  PaintBrushIcon,
  PathIcon,
  PauseIcon,
  PlayIcon,
  SidebarSimpleIcon,
  SlidersHorizontalIcon,
  SunIcon,
} from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import { useAtom, useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAnimatePosition } from '../hooks/useAnimatePosition.ts';
import type { PreviewScaleSetting } from '../layout/gallery-positions.ts';
import { usePortalContainer } from '../portal-container.tsx';
import type { PreviewCount } from '../spec.ts';
import { DTOUR_DEFAULTS } from '../spec.ts';
import {
  activeColumnsAtom,
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  centeringAtom,
  color2dColumnsAtom,
  color2dEnabledAtom,
  grandExitTargetAtom,
  guidedSuspendedAtom,
  keyframeLoadingsAtom,
  legendVisibleAtom,
  metadataAtom,
  minPointSizeAtom,
  panZoomModeAtom,
  pointColorByAtom,
  pointOpacityAtom,
  predefinedTourAtom,
  previewCountAtom,
  previewScaleAtom,
  resolvedPreviewScaleAtom,
  resumeGuidedAtom,
  selectedKeyframeAtom,
  showAxesAtom,
  showKeyframeLoadingsAtom,
  showKeyframeNumbersAtom,
  showLegendAtom,
  showTourDescriptionAtom,
  sliderVisibilityAtom,
  themeModeAtom,
  tourByAtom,
  tourDescriptionAtom,
  tourPlayingAtom,
  tourSliderSpacingAtom,
  tourSpeedAtom,
  tourTraversalAtom,
} from '../state/atoms.ts';
import { applySpecToStore } from '../state/spec-sync.ts';
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
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip.tsx';

type ViewMode = 'guided' | 'manual' | 'grand';

const MODE_CONFIG: { mode: ViewMode; label: string; icon: typeof PathIcon }[] = [
  { mode: 'guided', label: 'Guided', icon: PathIcon },
  { mode: 'manual', label: 'Manual', icon: CursorIcon },
  { mode: 'grand', label: 'Grand', icon: CompassIcon },
];

export type DtourToolbarProps = {
  onLoadData?: ((data: ArrayBuffer, fileName: string) => void) | undefined;
  onLogoClick?: (() => void) | undefined;
};

export const DtourToolbar = ({ onLoadData, onLogoClick }: DtourToolbarProps) => {
  const [playing, setPlaying] = useAtom(tourPlayingAtom);
  const [speed, setSpeed] = useAtom(tourSpeedAtom);
  const [zoom, setZoom] = useAtom(cameraZoomAtom);
  const [panX, setPanX] = useAtom(cameraPanXAtom);
  const [panY, setPanY] = useAtom(cameraPanYAtom);
  const [panZoomMode, setPanZoomMode] = useAtom(panZoomModeAtom);
  const metadata = useAtomValue(metadataAtom);
  const [tourTraversal, setTourTraversal] = useAtom(tourTraversalAtom);
  const resumeGuided = useAtomValue(resumeGuidedAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setSelectedKeyframe = useSetAtom(selectedKeyframeAtom);
  const [pointColorBy, setPointColorBy] = useAtom(pointColorByAtom);
  const [activeColumns, setActiveColumns] = useAtom(activeColumnsAtom);
  const [previewCount, setPreviewCount] = useAtom(previewCountAtom);
  const [previewScale, setPreviewScale] = useAtom(previewScaleAtom);
  const resolvedPreviewScale = useAtomValue(resolvedPreviewScaleAtom);
  const [showLegend, setShowLegend] = useAtom(showLegendAtom);
  const legendVisible = useAtomValue(legendVisibleAtom);
  const [themeMode, setThemeMode] = useAtom(themeModeAtom);
  const [showAxes, setShowAxes] = useAtom(showAxesAtom);
  const [showKeyframeNumbers, setShowKeyframeNumbers] = useAtom(showKeyframeNumbersAtom);
  const [showKeyframeLoadings, setShowKeyframeLoadings] = useAtom(showKeyframeLoadingsAtom);
  const hasKeyframeLoadings = useAtomValue(keyframeLoadingsAtom) !== null;
  const hasTourDescription = useAtomValue(tourDescriptionAtom) !== null;
  const [tourBy, setTourBy] = useAtom(tourByAtom);
  const predefinedTour = useAtomValue(predefinedTourAtom);
  const isPredefinedTour = predefinedTour !== null;
  const predefinedViewCount = predefinedTour?.keyframeCount ?? null;
  const [tourSliderSpacing, setTourSliderSpacing] = useAtom(tourSliderSpacingAtom);
  const [showTourDescription, setShowTourDescription] = useAtom(showTourDescriptionAtom);
  const [sliderVisibility, setSliderVisibility] = useAtom(sliderVisibilityAtom);
  const [color2dEnabled, setColor2dEnabled] = useAtom(color2dEnabledAtom);
  const [color2dColumns, setColor2dColumns] = useAtom(color2dColumnsAtom);
  const [minPointSize, setMinPointSize] = useAtom(minPointSizeAtom);
  const [pointOpacity, setPointOpacity] = useAtom(pointOpacityAtom);
  const [centering, setCentering] = useAtom(centeringAtom);

  const store = useStore();
  const portalContainer = usePortalContainer();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const toolbarRef = useRef<HTMLDivElement>(null);
  const [isMedium, setIsMedium] = useState(false);
  const [isWide, setIsWide] = useState(false);
  const [isCompact, setIsCompact] = useState(false);
  useEffect(() => {
    const el = toolbarRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      setIsMedium(w >= 720);
      setIsWide(w >= 960);
      setIsCompact(w < 480);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { cancelAnimation } = useAnimatePosition();

  const handlePlayPause = useCallback(() => {
    cancelAnimation();
    resumeGuided?.fn(300);
    if (!playing) setSelectedKeyframe(null);
    setPlaying((p) => !p);
  }, [playing, setPlaying, resumeGuided, setSelectedKeyframe, cancelAnimation]);

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

  // Determine active color-by column
  const activeColorColumn = pointColorBy;

  const toggleColorBy = useCallback(
    (columnName: string, isCategorical?: boolean) => {
      if (isCategorical && color2dEnabled) {
        // Clicking a categorical column exits 2D mode and applies 1D coloring
        setColor2dEnabled(false);
        setColor2dColumns(null);
        setPointColorBy(columnName);
        return;
      }
      if (color2dEnabled) {
        // In 2D mode: toggle column in/out of the pair (max 2, FIFO eviction).
        // Only produces a valid [colA, colB] when two distinct columns are selected.
        setColor2dColumns((prev) => {
          if (!prev) return [columnName, ''] as unknown as [string, string]; // partial: 1 selected
          if (prev[1] === '') {
            // Had 1 selected — this click completes the pair or deselects
            if (prev[0] === columnName) return null; // deselect only selected column
            return [prev[0], columnName]; // pair complete
          }
          // Already have 2 distinct columns
          if (prev[0] === columnName) return [prev[1], ''] as unknown as [string, string]; // deselect first
          if (prev[1] === columnName) return [prev[0], ''] as unknown as [string, string]; // deselect second
          // Evict oldest, add new
          return [prev[1], columnName];
        });
      } else {
        setPointColorBy((prev) => (prev === columnName ? null : columnName));
      }
    },
    [setPointColorBy, color2dEnabled, setColor2dEnabled, setColor2dColumns],
  );

  const toggle2dMode = useCallback(() => {
    setColor2dEnabled((prev) => {
      if (!prev) {
        // Entering 2D mode: clear 1D color and auto-select first two numerical columns
        setPointColorBy(null);
        const cols = metadata?.columnNames;
        if (cols && cols.length >= 2) {
          setColor2dColumns([cols[0]!, cols[1]!]);
        }
      } else {
        // Leaving 2D mode: clear 2D columns
        setColor2dColumns(null);
      }
      return !prev;
    });
  }, [setColor2dEnabled, setPointColorBy, setColor2dColumns, metadata]);

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

  // Single mode-switch handler shared by the segmented control and the
  // collapsed dropdown. Preserves the grand-exit choreography: switching
  // away from grand sets an exit target (animated) rather than snapping.
  const switchToMode = (mode: ViewMode) => {
    if (mode === 'guided') {
      if (tourTraversal === 'grand') {
        setGrandExitTarget('guided');
      } else if (tourTraversal !== 'guided') {
        setGuidedSuspended(true);
        setTourTraversal('guided');
      }
      return;
    }
    if (tourTraversal === 'grand') {
      if (mode === 'grand') {
        setGrandExitTarget(null);
        return;
      }
      setGrandExitTarget(mode);
    } else {
      if (tourTraversal === 'guided') setPlaying(false);
      if (mode === 'grand') setGrandExitTarget(null);
      setTourTraversal(mode);
    }
  };

  const activeMode = MODE_CONFIG.find((m) => m.mode === tourTraversal) ?? MODE_CONFIG[0]!;
  const ActiveModeIcon = activeMode.icon;

  return (
    <div
      ref={toolbarRef}
      className="grid h-10 grid-cols-[1fr_auto_1fr] items-center border-b border-dtour-surface bg-dtour-bg px-3 text-dtour-text"
    >
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
      <div className="flex items-center gap-1">
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
        {isWide ? (
          <div className="group/modes ml-2 flex items-center overflow-hidden rounded-md border border-dtour-surface">
            {/* Guided button — expands to include Dims/PCA sub-toggle when active */}
            <div
              className={`group flex gap-0 items-center ${tourTraversal === 'guided' ? 'bg-dtour-surface text-dtour-highlight' : 'text-dtour-text-muted'} ${!isMedium && tourTraversal !== 'guided' ? 'overflow-hidden max-w-0 group-hover/modes:max-w-24 transition-[max-width] duration-200 ease-in-out' : ''}`}
            >
              <Button
                variant="ghost"
                size="sm"
                className={`rounded-none ${tourTraversal === 'guided' ? 'text-dtour-highlight' : ''}`}
                onClick={() => {
                  if (tourTraversal === 'grand') {
                    setGrandExitTarget('guided');
                  } else if (tourTraversal !== 'guided') {
                    setGuidedSuspended(true);
                    setTourTraversal('guided');
                  }
                }}
                title="Guided"
              >
                {isWide && (
                  <PathIcon size={14} weight={tourTraversal === 'guided' ? 'fill' : 'regular'} />
                )}
                <span className="text-xs">
                  Guided
                  {tourTraversal === 'guided' ? (
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 ease-out">
                      :
                    </span>
                  ) : (
                    ''
                  )}
                </span>
              </Button>
              {tourTraversal === 'guided' && tourBy === 'parameter' && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="text-xs text-dtour-highlight cursor-default px-1">
                        Params
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      In parameter touring, only guided tour is available
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              {tourTraversal === 'guided' && isPredefinedTour && tourBy !== 'parameter' && (
                <div
                  className={`-ml-1 flex items-center overflow-hidden max-w-0 group-hover:max-w-24 transition-[max-width] duration-200 ease-in-out ${!isMedium ? 'group-hover/modes:max-w-24' : ''}`}
                >
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-xs text-dtour-highlight cursor-default px-1">
                          Planned
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="bottom">This tour was precomputed</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              )}
              {tourTraversal === 'guided' && !isPredefinedTour && tourBy !== 'parameter' && (
                <div
                  className={`-ml-1 flex items-center overflow-hidden max-w-0 group-hover:max-w-24 transition-[max-width] duration-200 ease-in-out ${!isMedium ? 'group-hover/modes:max-w-24' : ''}`}
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`rounded-none px-0 ${tourBy === 'dimensions' ? 'text-dtour-highlight' : 'text-dtour-text-muted'}`}
                    onClick={() => setTourBy('dimensions')}
                    title="Tour by dimensions"
                  >
                    <span className="text-xs">Dims</span>
                  </Button>
                  <span className="text-[10px] text-dtour-text-muted select-none px-1">/</span>
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
                </div>
              )}
            </div>
            {/* Manual + Grand buttons (hidden for parameter tours) */}
            {tourBy !== 'parameter' &&
              MODE_CONFIG.filter(({ mode }) => mode !== 'guided').map(
                ({ mode, label, icon: Icon }) => (
                  <div
                    key={mode}
                    className={
                      !isMedium && tourTraversal !== mode
                        ? 'overflow-hidden max-w-0 group-hover/modes:max-w-24 transition-[max-width] duration-200 ease-in-out'
                        : ''
                    }
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      className={`rounded-none ${tourTraversal === mode ? 'bg-dtour-surface text-dtour-highlight' : 'text-dtour-text-muted'}`}
                      onClick={() => {
                        if (tourTraversal === 'grand') {
                          if (mode === 'grand') {
                            setGrandExitTarget(null);
                            return;
                          }
                          setGrandExitTarget(mode);
                        } else {
                          if (mode !== 'guided' && tourTraversal === 'guided') setPlaying(false);
                          if (mode === 'grand') setGrandExitTarget(null);
                          setTourTraversal(mode);
                        }
                      }}
                      title={label}
                    >
                      {isWide && (
                        <Icon size={14} weight={tourTraversal === mode ? 'fill' : 'regular'} />
                      )}
                      <span className="text-xs">{label}</span>
                    </Button>
                  </div>
                ),
              )}
          </div>
        ) : tourBy === 'parameter' ? (
          <div className="ml-2 flex items-center gap-1 rounded-md border border-dtour-surface bg-dtour-surface px-2 py-1 text-dtour-highlight">
            <PathIcon size={14} weight="fill" />
            {!isCompact && <span className="text-xs">Guided</span>}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="cursor-default text-xs text-dtour-text-muted">Params</span>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  In parameter touring, only guided tour is available
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        ) : (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="ml-2 rounded-md border border-dtour-surface text-dtour-highlight"
                title={`Mode: ${activeMode.label}`}
              >
                <ActiveModeIcon size={14} weight="fill" />
                {!isCompact && <span className="text-xs">{activeMode.label}</span>}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-44">
              {MODE_CONFIG.map(({ mode, label, icon: Icon }) => (
                <DropdownMenuItem
                  key={mode}
                  className={`gap-2 text-xs ${tourTraversal === mode ? 'text-dtour-highlight' : ''}`}
                  onSelect={() => switchToMode(mode)}
                >
                  <Icon size={14} weight={tourTraversal === mode ? 'fill' : 'regular'} />
                  <span className="flex-1">{label}</span>
                </DropdownMenuItem>
              ))}
              {tourTraversal === 'guided' &&
                (isPredefinedTour ? (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel className="text-xs font-normal text-dtour-text-muted">
                      Planned tour (precomputed)
                    </DropdownMenuLabel>
                  </>
                ) : (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="gap-2" onSelect={(e) => e.preventDefault()}>
                      <span className="flex-1 text-xs">Tour by</span>
                      <div className="flex overflow-hidden rounded-md border border-dtour-border text-[10px] font-medium">
                        <button
                          type="button"
                          onClick={() => setTourBy('dimensions')}
                          className={`cursor-pointer px-1.5 py-0.5 transition-colors ${
                            tourBy === 'dimensions'
                              ? 'bg-dtour-border text-dtour-text'
                              : 'text-dtour-text-muted hover:text-dtour-highlight'
                          }`}
                          title="Tour by dimensions"
                        >
                          Dims
                        </button>
                        <button
                          type="button"
                          onClick={() => setTourBy('pca')}
                          className={`cursor-pointer border-l border-dtour-border px-1.5 py-0.5 transition-colors ${
                            tourBy === 'pca'
                              ? 'bg-dtour-border text-dtour-text'
                              : 'text-dtour-text-muted hover:text-dtour-highlight'
                          }`}
                          title="Tour by principal components"
                        >
                          PCA
                        </button>
                      </div>
                    </DropdownMenuItem>
                  </>
                ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Center: Settings | Play | Zoom (+ Speed & Axes at ≥960px) */}
      <div className="flex items-center gap-1">
        {/* Settings dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" title="Settings">
              <SlidersHorizontalIcon size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center" className="w-56">
            <DropdownMenuLabel className="text-xs font-semibold">Rendering</DropdownMenuLabel>
            <DropdownMenuItem
              className="flex flex-col items-start gap-1"
              onSelect={(e) => e.preventDefault()}
            >
              <div className="flex w-full items-center justify-between">
                <span className="text-xs">Min point size</span>
                <span className="text-xs font-medium text-dtour-highlight">{minPointSize}px</span>
              </div>
              <Slider
                min={1}
                max={20}
                step={1}
                value={[minPointSize]}
                onValueChange={([v]: number[]) => {
                  if (v !== undefined) setMinPointSize(v);
                }}
                className="w-full"
              />
            </DropdownMenuItem>
            <DropdownMenuItem
              className="flex flex-col items-start gap-1"
              onSelect={(e) => e.preventDefault()}
            >
              <div className="flex w-full items-center justify-between">
                <span className="text-xs">Point opacity</span>
                <span className="text-xs font-medium text-dtour-highlight">
                  {pointOpacity === 'auto' ? 'Auto' : `${Math.round(pointOpacity * 100)}%`}
                </span>
              </div>
              <Slider
                min={0}
                max={20}
                step={1}
                value={[pointOpacity === 'auto' ? 0 : Math.round(pointOpacity * 20)]}
                onValueChange={([v]: number[]) => {
                  if (v !== undefined) setPointOpacity(v === 0 ? 'auto' : v / 20);
                }}
                className="w-full"
              />
            </DropdownMenuItem>
            <DropdownMenuItem
              className="gap-4"
              onSelect={(e) => {
                e.preventDefault();
                setCentering(centering === 'midrange' ? 'mean' : 'midrange');
              }}
            >
              <span className="flex-1 text-xs">Mean centering</span>
              <Checkbox
                checked={centering === 'mean'}
                onCheckedChange={() => setCentering(centering === 'midrange' ? 'mean' : 'midrange')}
              />
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className="text-xs font-semibold">Camera</DropdownMenuLabel>
            <DropdownMenuItem
              className="flex flex-col items-start gap-1"
              onSelect={(e) => e.preventDefault()}
            >
              <div className="flex w-full items-center justify-between">
                <span className="text-xs">Zoom</span>
                <span className="text-xs font-medium text-dtour-highlight">
                  {Math.round(zoom * 100)}%
                </span>
              </div>
              <Slider
                min={0}
                max={ZOOM_STEPS.length - 1}
                step={1}
                ticks={ZOOM_STEPS.length}
                value={[zoomToStep(zoom)]}
                onValueChange={([step]: number[]) => {
                  if (step !== undefined) setZoom(stepToZoom(step));
                }}
                className="w-full"
              />
            </DropdownMenuItem>
            {!isWide && tourTraversal === 'guided' && (
              <DropdownMenuItem
                className="gap-4"
                onSelect={(e) => {
                  e.preventDefault();
                  setShowAxes((v) => !v);
                }}
              >
                <span className="flex-1 text-xs">Show axes</span>
                <Checkbox checked={showAxes} onCheckedChange={() => setShowAxes((v) => !v)} />
              </DropdownMenuItem>
            )}

            {!isWide && (tourTraversal === 'guided' || tourTraversal === 'grand') && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs font-semibold">Playback</DropdownMenuLabel>
                <DropdownMenuItem
                  className="flex flex-col items-start gap-1"
                  onSelect={(e) => e.preventDefault()}
                >
                  <div className="flex w-full items-center justify-between">
                    <span className="text-xs">Speed</span>
                    <span className="text-xs font-medium text-dtour-highlight">{speed}x</span>
                  </div>
                  <Slider
                    min={0}
                    max={SPEED_STEPS.length - 1}
                    step={1}
                    ticks={SPEED_STEPS.length}
                    value={[speedToStep(speed)]}
                    onValueChange={([step]: number[]) => {
                      if (step !== undefined) setSpeed(stepToSpeed(step));
                    }}
                    className="w-full"
                  />
                </DropdownMenuItem>
              </>
            )}

            {tourTraversal === 'guided' && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs font-semibold">Tour</DropdownMenuLabel>
                <DropdownMenuItem
                  className="gap-4"
                  onSelect={(e) => {
                    e.preventDefault();
                    setTourSliderSpacing(tourSliderSpacing === 'equal' ? 'geodesic' : 'equal');
                  }}
                >
                  <span className="flex-1 text-xs">Geodesic spacing</span>
                  <Checkbox
                    checked={tourSliderSpacing === 'geodesic'}
                    onCheckedChange={() =>
                      setTourSliderSpacing(tourSliderSpacing === 'equal' ? 'geodesic' : 'equal')
                    }
                  />
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="flex flex-col items-start gap-1"
                  onSelect={(e) => e.preventDefault()}
                >
                  <div className="flex w-full items-center justify-between">
                    <span className="text-xs">Slider Visibility</span>
                    <span className="text-xs font-medium text-dtour-highlight">
                      {SLIDER_VIS_LABELS[sliderVisibility]}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={2}
                    step={1}
                    ticks={3}
                    value={[SLIDER_VIS_STEPS.indexOf(sliderVisibility)]}
                    onValueChange={([v]: number[]) => {
                      if (v !== undefined) setSliderVisibility(SLIDER_VIS_STEPS[v]!);
                    }}
                    className="w-full"
                  />
                </DropdownMenuItem>
                {hasKeyframeLoadings && (
                  <DropdownMenuItem
                    className="gap-4"
                    onSelect={(e) => {
                      e.preventDefault();
                      setShowKeyframeLoadings((v) => !v);
                    }}
                  >
                    <span className="flex-1 text-xs">Feature correlations</span>
                    <Checkbox
                      checked={showKeyframeLoadings}
                      onCheckedChange={() => setShowKeyframeLoadings((v) => !v)}
                    />
                  </DropdownMenuItem>
                )}
                {hasTourDescription && (
                  <DropdownMenuItem
                    className="gap-4"
                    onSelect={(e) => {
                      e.preventDefault();
                      setShowTourDescription((v) => !(v ?? hasTourDescription));
                    }}
                  >
                    <span className="flex-1 text-xs">Tour description</span>
                    <Checkbox
                      checked={showTourDescription ?? hasTourDescription}
                      onCheckedChange={() =>
                        setShowTourDescription((v) => !(v ?? hasTourDescription))
                      }
                    />
                  </DropdownMenuItem>
                )}
              </>
            )}

            {tourTraversal === 'guided' && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs font-semibold">Previews</DropdownMenuLabel>
                <DropdownMenuItem
                  className="flex flex-col items-start gap-1"
                  onSelect={(e) => e.preventDefault()}
                >
                  <div className="flex w-full items-center justify-between">
                    <span className="text-xs">Count</span>
                    <span className="text-xs font-medium text-dtour-highlight">
                      {isPredefinedTour ? predefinedViewCount : previewCount}
                    </span>
                  </div>
                  {!isPredefinedTour && (
                    <Slider
                      min={0}
                      max={PREVIEW_COUNT_STEPS.length - 1}
                      step={1}
                      ticks={PREVIEW_COUNT_STEPS.length}
                      value={[PREVIEW_COUNT_STEPS.indexOf(previewCount)]}
                      onValueChange={([step]: number[]) => {
                        if (step !== undefined) setPreviewCount(PREVIEW_COUNT_STEPS[step]!);
                      }}
                      className="w-full"
                    />
                  )}
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="flex flex-col items-start gap-1"
                  onSelect={(e) => e.preventDefault()}
                >
                  <div className="flex w-full items-center justify-between">
                    <span className="text-xs">Size</span>
                    <span className="text-xs font-medium text-dtour-highlight">
                      {previewScale === 'auto'
                        ? `Auto · ${SCALE_LABELS[String(resolvedPreviewScale)]}`
                        : (SCALE_LABELS[String(previewScale)] ?? previewScale)}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={PREVIEW_SCALE_STEPS.length - 1}
                    step={1}
                    ticks={PREVIEW_SCALE_STEPS.length}
                    value={[PREVIEW_SCALE_STEPS.indexOf(previewScale)]}
                    onValueChange={([step]: number[]) => {
                      if (step !== undefined) setPreviewScale(PREVIEW_SCALE_STEPS[step]!);
                    }}
                    className="w-full"
                  />
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="gap-4"
                  onSelect={(e) => {
                    e.preventDefault();
                    setShowKeyframeNumbers((v) => !v);
                  }}
                >
                  <span className="flex-1 text-xs">Show Numbers</span>
                  <Checkbox
                    checked={showKeyframeNumbers}
                    onCheckedChange={() => setShowKeyframeNumbers((v) => !v)}
                  />
                </DropdownMenuItem>
              </>
            )}
            {isCompact && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs font-semibold">Appearance</DropdownMenuLabel>
                <DropdownMenuItem className="gap-4" onSelect={(e) => e.preventDefault()}>
                  <span className="flex-1 text-xs">Theme</span>
                  <div className="flex overflow-hidden rounded-md border border-dtour-border">
                    {(['light', 'dark', 'system'] as const).map((m) => {
                      const Icon = m === 'light' ? SunIcon : m === 'dark' ? MoonIcon : MonitorIcon;
                      return (
                        <button
                          key={m}
                          type="button"
                          onClick={() => setThemeMode(m)}
                          className={`cursor-pointer px-2 py-1 transition-colors ${
                            themeMode === m
                              ? 'bg-dtour-border text-dtour-text'
                              : 'text-dtour-text-muted hover:text-dtour-highlight'
                          }`}
                          title={`${m.charAt(0).toUpperCase()}${m.slice(1)} theme`}
                        >
                          <Icon size={14} weight="fill" />
                        </button>
                      );
                    })}
                  </div>
                </DropdownMenuItem>
              </>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-xs"
              onSelect={() => applySpecToStore(store, DTOUR_DEFAULTS)}
            >
              Reset settings
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Speed popover — standalone at ≥960px */}
        {isWide && (tourTraversal === 'guided' || tourTraversal === 'grand') && (
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
                className="z-50 flex flex-col items-center gap-2 rounded border border-dtour-border bg-dtour-bg p-3 shadow-md origin-(--radix-popover-content-transform-origin) data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 animate-ease-out"
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

        {/* Play/Pause — guided only */}
        {tourTraversal === 'guided' && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handlePlayPause}
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? <PauseIcon size={16} weight="fill" /> : <PlayIcon size={16} weight="fill" />}
          </Button>
        )}

        {/* Axes toggle — standalone at ≥960px, guided only */}
        {isWide && tourTraversal === 'guided' && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setShowAxes((v) => !v)}
            title={showAxes ? 'Hide axes' : 'Show axes'}
            className={showAxes ? '' : 'opacity-40'}
          >
            <ChartScatterIcon size={16} weight={showAxes ? 'fill' : 'regular'} />
          </Button>
        )}

        {/* Pan/zoom mode toggle — only shown in guided mode (always active in manual/grand) */}
        {tourTraversal === 'guided' && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setPanZoomMode((v) => !v)}
            title={panZoomMode ? 'Pan/zoom scroll (click to disable)' : 'Enable pan/zoom scroll'}
            className={panZoomMode ? '' : 'opacity-40'}
          >
            <MagnifyingGlassIcon size={16} weight={panZoomMode ? 'fill' : 'regular'} />
          </Button>
        )}

        {/* Camera reset — appears when camera is not at default */}
        {(panX !== 0 || panY !== 0 || zoom !== DTOUR_DEFAULTS.cameraZoom) && (
          <Button
            variant="ghost"
            size="icon"
            title="Reset camera"
            onClick={() => {
              const startPanX = panX;
              const startPanY = panY;
              const startZoom = zoom;
              const targetZoom = DTOUR_DEFAULTS.cameraZoom;
              const startTime = performance.now();
              const duration = 250;
              const tick = (now: number) => {
                const t = Math.min(1, (now - startTime) / duration);
                // ease-in-out cubic
                const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
                setPanX(startPanX * (1 - e));
                setPanY(startPanY * (1 - e));
                setZoom(startZoom + (targetZoom - startZoom) * e);
                if (t < 1) requestAnimationFrame(tick);
              };
              requestAnimationFrame(tick);
            }}
          >
            <ArrowsCounterClockwiseIcon size={16} />
          </Button>
        )}
      </div>

      {/* Right: data info + settings */}
      <div className="flex items-center justify-end gap-1">
        {metadata ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              {isWide ? (
                <Button variant="ghost" size="sm">
                  {metadata.rowCount.toLocaleString()} pts &times;{' '}
                  {activeCount === metadata.dimCount
                    ? `${metadata.dimCount} dims`
                    : `${activeCount}/${metadata.dimCount} dims`}
                  <CaretDownIcon size={12} />
                </Button>
              ) : (
                <Button variant="ghost" size="icon" title="Data">
                  <DatabaseIcon size={16} />
                </Button>
              )}
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="max-h-[60vh] w-64 overflow-y-auto">
              {/* Numeric columns */}
              {metadata.columnNames.length > 0 && (
                <>
                  <DropdownMenuLabel className="flex items-center justify-between text-xs font-semibold">
                    <span>Numerical Dims</span>
                    {/* biome-ignore lint/a11y/useKeyWithClickEvents: inner buttons handle keyboard */}
                    <div
                      className="flex rounded-md border border-dtour-border text-[10px] font-medium overflow-hidden opacity-50"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          if (color2dEnabled) toggle2dMode();
                        }}
                        className={`cursor-pointer px-1.5 py-0.5 transition-colors ${
                          !color2dEnabled
                            ? 'bg-dtour-border text-dtour-text'
                            : 'text-dtour-text-muted hover:text-dtour-highlight'
                        }`}
                        title="1D coloring (single column)"
                      >
                        1D
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (!color2dEnabled) toggle2dMode();
                        }}
                        className={`cursor-pointer px-1.5 py-0.5 transition-colors border-l border-dtour-border ${
                          color2dEnabled
                            ? 'bg-dtour-border text-dtour-text'
                            : 'text-dtour-text-muted hover:text-dtour-highlight'
                        }`}
                        title="2D coloring (two columns)"
                      >
                        2D
                      </button>
                    </div>
                  </DropdownMenuLabel>
                  {metadata.columnNames.map((col, index) => {
                    const isActive = activeColumns === null || activeColumns.has(index);
                    const isColor2d = color2dEnabled && color2dColumns?.includes(col);
                    return (
                      <ColumnRow
                        key={col}
                        name={col}
                        dtype="num"
                        checked={isActive}
                        onCheckedChange={
                          isPredefinedTour ? undefined : () => handleToggleColumn(index)
                        }
                        disabled={isPredefinedTour || (isActive && activeCount <= 2)}
                        isColorActive={color2dEnabled ? !!isColor2d : activeColorColumn === col}
                        onToggleColor={() => toggleColorBy(col)}
                      />
                    );
                  })}
                </>
              )}

              {/* Categorical columns */}
              {metadata.categoricalColumnNames.length > 0 && (
                <>
                  <DropdownMenuLabel className="flex items-center justify-between text-xs font-semibold">
                    <span>Categorical Dims</span>
                    <div className="flex rounded-md border border-dtour-border text-[10px] font-medium overflow-hidden opacity-50">
                      <span className="bg-dtour-highlight text-dtour-bg px-1.5 py-0.5">1D</span>
                    </div>
                  </DropdownMenuLabel>
                  {metadata.categoricalColumnNames.map((col) => (
                    <ColumnRow
                      key={col}
                      name={col}
                      dtype="cat"
                      isColorActive={activeColorColumn === col}
                      onToggleColor={() => toggleColorBy(col, true)}
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
        {!isCompact && (
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
        )}
        {(activeColorColumn || (color2dEnabled && color2dColumns?.[1])) && (
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
  dtype: _dtype,
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
    className={`flex items-center gap-2 pr-1${disabled ? ' cursor-default opacity-60 data-highlighted:bg-transparent' : ''}`}
    checked={checked ?? false}
    onCheckedChange={disabled ? undefined : onCheckedChange}
  >
    <span className="flex-1 truncate text-xs">{name}</span>
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onToggleColor();
      }}
      className={`shrink-0 cursor-pointer rounded p-1 opacity-100 transition-[color,transform] active:scale-[0.85] ${
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

const SLIDER_VIS_STEPS: ('visible' | 'subtle' | 'hidden')[] = ['hidden', 'subtle', 'visible'];
const SLIDER_VIS_LABELS: Record<string, string> = {
  visible: 'Visible',
  subtle: 'Subtle',
  hidden: 'Hidden',
};

const PREVIEW_COUNT_STEPS: PreviewCount[] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const PREVIEW_SCALE_STEPS: PreviewScaleSetting[] = ['auto', 0.5, 0.75, 1];
const SCALE_LABELS: Record<string, string> = { auto: 'Auto', 1: 'L', 0.75: 'M', 0.5: 'S' };

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

// Sorted ascending: zoom < 1 = zoom out, zoom > 1 = zoom in
const ZOOM_STEPS = [0.25, 0.4, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 2.5, 4] as const;

const zoomToStep = (zoom: number): number => {
  let best = 0;
  let bestDist = Math.abs(zoom - ZOOM_STEPS[0]!);
  for (let i = 1; i < ZOOM_STEPS.length; i++) {
    const dist = Math.abs(zoom - ZOOM_STEPS[i]!);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
};

const stepToZoom = (step: number): number => ZOOM_STEPS[step] ?? 1;
