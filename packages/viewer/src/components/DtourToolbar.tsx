import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  CompassIcon,
  CursorIcon,
  GaugeIcon,
  PathIcon,
  PauseIcon,
  PlayIcon,
} from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback } from 'react';
import { usePlayback } from '../hooks/usePlayback.ts';
import {
  grandExitTargetAtom,
  guidedSuspendedAtom,
  metadataAtom,
  selectedKeyframeAtom,
  tourPlayingAtom,
  tourPositionAtom,
  tourSpeedAtom,
  viewModeAtom,
} from '../state/atoms.ts';
import { Button } from './ui/button.tsx';
import {
  DropdownMenu,
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

export const DtourToolbar = () => {
  const [playing, setPlaying] = useAtom(tourPlayingAtom);
  const setPosition = useSetAtom(tourPositionAtom);
  const [speed, setSpeed] = useAtom(tourSpeedAtom);
  const metadata = useAtomValue(metadataAtom);
  const [viewMode, setViewMode] = useAtom(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setSelectedKeyframe = useSetAtom(selectedKeyframeAtom);

  // Activate the rAF playback loop
  usePlayback();

  const handlePlayPause = useCallback(() => {
    setGuidedSuspended(false);
    if (!playing) setSelectedKeyframe(null);
    setPlaying((p) => !p);
  }, [playing, setPlaying, setGuidedSuspended, setSelectedKeyframe]);

  const handleReset = useCallback(() => {
    setGuidedSuspended(false);
    setPlaying(false);
    setPosition(0);
  }, [setPlaying, setPosition, setGuidedSuspended]);

  return (
    <div className="grid h-10 grid-cols-[1fr_auto_1fr] items-center border-b border-dtour-border bg-dtour-bg px-3 text-dtour-text">
      {/* Left: branding + mode switcher */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold tracking-wide text-white">dtour</span>
        <div className="ml-2 flex items-center rounded-md border border-dtour-border">
          {MODE_CONFIG.map(({ mode, label, icon: Icon }) => (
            <Button
              key={mode}
              variant="ghost"
              size="sm"
              className={`rounded-none first:rounded-l-md last:rounded-r-md ${viewMode === mode ? 'bg-dtour-surface text-white' : 'text-dtour-text-muted'}`}
              onClick={() => {
                if (viewMode === 'grand') {
                  if (mode === 'grand') {
                    // Cancel any in-progress exit
                    setGrandExitTarget(null);
                    return;
                  }
                  // Request ease-out exit from grand
                  setGrandExitTarget(mode);
                } else {
                  // Instant switch (not exiting grand)
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
            {/* Reset */}
            <Button variant="ghost" size="icon" onClick={handleReset} title="Reset to start">
              <ArrowCounterClockwiseIcon size={16} />
            </Button>

            {/* Play / Pause */}
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

        {/* Speed — shown in guided and grand modes */}
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
                className="z-50 flex flex-col items-center gap-2 rounded-md border border-dtour-border bg-dtour-surface p-3 shadow-md"
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
      </div>

      {/* Right: data info + settings */}
      <div className="flex items-center justify-end gap-2">
        <span className="text-xs text-dtour-text-muted">
          {metadata
            ? `${metadata.rowCount.toLocaleString()} pts \u00d7 ${metadata.dimCount} dims`
            : 'No data'}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm">
              Settings
              <CaretDownIcon size={12} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuLabel>Settings</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem disabled>Column selection (coming soon)</DropdownMenuItem>
            <DropdownMenuItem disabled>Color mapping (coming soon)</DropdownMenuItem>
            <DropdownMenuItem disabled>Labels (coming soon)</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
};

// Map between continuous speed values and discrete slider steps
const SPEED_STEPS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5] as const;

const speedToStep = (speed: number): number => {
  // Find closest step
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
