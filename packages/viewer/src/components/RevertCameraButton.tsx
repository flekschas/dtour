import { ArrowCounterClockwiseIcon } from '@phosphor-icons/react';
import { useAtomValue } from 'jotai';
import { useEffect, useState } from 'react';
import { cn } from '../lib/utils.ts';
import { is3dRotatedAtom } from '../state/atoms.ts';
import { Button } from './ui/button.tsx';

type RevertCameraButtonProps = {
  onRevert: () => void;
};

export const RevertCameraButton = ({ onRevert }: RevertCameraButtonProps) => {
  const is3dRotated = useAtomValue(is3dRotatedAtom);
  // Delay visibility by a frame so the CSS transition triggers
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (is3dRotated) {
      const id = requestAnimationFrame(() => setVisible(true));
      return () => cancelAnimationFrame(id);
    }
    setVisible(false);
  }, [is3dRotated]);

  if (!is3dRotated) return null;

  return (
    <Button
      variant="ghost"
      className={cn(
        'flex items-center gap-2 absolute bottom-8 left-1/2 -translate-x-1/2 text-xs cursor-pointer px-3 py-2 bg-dtour-surface/60 hover:bg-dtour-surface backdrop-blur-sm transition-opacity ease-out duration-250',
        visible ? 'opacity-100' : 'opacity-0',
      )}
      onClick={onRevert}
    >
      <ArrowCounterClockwiseIcon size={12} />
      Revert camera to adjust projection
    </Button>
  );
};
