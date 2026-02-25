import { atom } from 'jotai';
import { canvasSizeAtom, metadataAtom, pointOpacityAtom, pointSizeAtom } from './atoms.ts';

const COLOR_BUDGET = 0.5;

/**
 * Compute ideal point size (NDC) and opacity using the Reusser color budget algorithm.
 *
 * Each point gets a share of the total "color budget" (fraction of canvas area).
 * The ideal radius is derived from that share. When the ideal radius falls below
 * the minimum visible size (1 CSS pixel = DPR physical pixels), the radius is
 * clamped and opacity is reduced instead.
 */
export const computeAutoStyle = (
  rowCount: number,
  canvasWidth: number,
  canvasHeight: number,
  dpr: number,
): { pointSize: number; opacity: number } => {
  if (rowCount === 0 || canvasWidth === 0 || canvasHeight === 0) {
    return { pointSize: 0.012, opacity: 0.7 };
  }

  const physW = canvasWidth * dpr;
  const physH = canvasHeight * dpr;
  const totalBudget = physW * physH * COLOR_BUDGET;
  const perPoint = totalBudget / rowCount;
  const idealRadius = Math.sqrt(perPoint / Math.PI);
  const minRadius = dpr; // 1 CSS pixel

  let radius: number;
  let opacity: number;

  if (idealRadius >= minRadius) {
    radius = idealRadius;
    opacity = 1.0;
  } else {
    radius = minRadius;
    opacity = Math.max(0.01, perPoint / (Math.PI * minRadius * minRadius));
  }

  // Convert physical-pixel radius to NDC point size.
  // The shader interprets point_size as NDC height units (range [-1, 1] = 2 units).
  // pointSize = diameter_ndc = 2 * radius / canvasHeight_physical
  const pointSize = (2 * radius) / physH;

  return { pointSize, opacity };
};

/** Resolved point size — auto-computed or user-specified numeric value. */
export const resolvedPointSizeAtom = atom((get) => {
  const raw = get(pointSizeAtom);
  if (raw !== 'auto') return raw;
  const meta = get(metadataAtom);
  const canvas = get(canvasSizeAtom);
  if (!meta || canvas.width === 0) return 0.012;
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  return computeAutoStyle(meta.rowCount, canvas.width, canvas.height, dpr).pointSize;
});

/** Resolved point opacity — auto-computed or user-specified numeric value. */
export const resolvedPointOpacityAtom = atom((get) => {
  const raw = get(pointOpacityAtom);
  if (raw !== 'auto') return raw;
  const meta = get(metadataAtom);
  const canvas = get(canvasSizeAtom);
  if (!meta || canvas.width === 0) return 0.7;
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  return computeAutoStyle(meta.rowCount, canvas.width, canvas.height, dpr).opacity;
});
