import { ArrowCounterClockwise, CaretDown, Gauge, Pause, Play } from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import * as Slider from '@radix-ui/react-slider';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback } from 'react';
import { usePlayback } from '../hooks/usePlayback.ts';
import { metadataAtom, tourPlayingAtom, tourPositionAtom, tourSpeedAtom } from '../state/atoms.ts';
import { Button } from './ui/button.tsx';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu.tsx';

export const DtourToolbar = () => {
  const [playing, setPlaying] = useAtom(tourPlayingAtom);
  const setPosition = useSetAtom(tourPositionAtom);
  const [speed, setSpeed] = useAtom(tourSpeedAtom);
  const metadata = useAtomValue(metadataAtom);

  // Activate the rAF playback loop
  usePlayback();

  const handlePlayPause = useCallback(() => {
    setPlaying((p) => !p);
  }, [setPlaying]);

  const handleReset = useCallback(() => {
    setPlaying(false);
    setPosition(0);
  }, [setPlaying, setPosition]);

  return (
    <div className="grid h-10 grid-cols-[1fr_auto_1fr] items-center border-b border-dtour-border bg-dtour-bg px-3 text-dtour-text">
      {/* Left: branding */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold tracking-wide text-white">dtour</span>
      </div>

      {/* Center: playback controls — grid column guarantees true centering */}
      <div className="flex items-center gap-1">
        {/* Reset */}
        <Button variant="ghost" size="icon" onClick={handleReset} title="Reset to start">
          <ArrowCounterClockwise size={16} />
        </Button>

        {/* Play / Pause */}
        <Button
          variant="ghost"
          size="icon"
          onClick={handlePlayPause}
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? <Pause size={16} weight="fill" /> : <Play size={16} weight="fill" />}
        </Button>

        {/* Speed — icon button with vertical slider popover */}
        <Popover.Root>
          <Popover.Trigger asChild>
            <Button variant="ghost" size="icon" title={`Speed: ${speed}x`}>
              <Gauge size={16} />
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
              <Slider.Root
                orientation="vertical"
                min={0}
                max={SPEED_STEPS.length - 1}
                step={1}
                value={[speedToStep(speed)]}
                onValueChange={([step]) => {
                  if (step !== undefined) setSpeed(stepToSpeed(step));
                }}
                style={{
                  position: 'relative',
                  display: 'flex',
                  width: 20,
                  height: 120,
                  touchAction: 'none',
                  userSelect: 'none',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Slider.Track
                  style={{
                    position: 'relative',
                    width: 6,
                    flexGrow: 1,
                    borderRadius: 9999,
                    backgroundColor: '#2a2a3a',
                  }}
                >
                  <Slider.Range
                    style={{
                      position: 'absolute',
                      width: '100%',
                      borderRadius: 9999,
                      backgroundColor: '#4080e8',
                    }}
                  />
                </Slider.Track>
                <Slider.Thumb
                  style={{
                    display: 'block',
                    width: 16,
                    height: 16,
                    borderRadius: '50%',
                    border: '2px solid #4080e8',
                    backgroundColor: 'white',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                    outline: 'none',
                  }}
                />
              </Slider.Root>
              <span className="text-xs font-medium text-white">{speed}x</span>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
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
              <CaretDown size={12} />
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
