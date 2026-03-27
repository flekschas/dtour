import { useCallback, useEffect, useRef } from 'react';

// --- Constants (from regl-scatterplot) ---
const LONG_PRESS_TIME = 750;
const LONG_PRESS_AFTER_EFFECT_TIME = 500;
const LONG_PRESS_EFFECT_DELAY = 100;
const LONG_PRESS_REVERT_EFFECT_TIME = 250;

const INDICATOR_COLOR = '#4f8ff7';
const INDICATOR_ACTIVE_COLOR = '#4f8ff7';

// --- getComputedStyle helpers ---
const getCurrentTransform = (node: HTMLElement, hasRotated = false) => {
  const cs = getComputedStyle(node);
  const opacity = +cs.opacity;
  const m = cs.transform.match(/([0-9.-]+)+/g);

  if (!m) return { opacity, scale: 0, rotate: 0 };

  const a = +m[0]!;
  const b = +m[1]!;
  const scale = Math.sqrt(a * a + b * b);
  let rotate = Math.atan2(b, a) * (180 / Math.PI);
  if (hasRotated && rotate <= 0) rotate = 360 + rotate;

  return { opacity, scale, rotate };
};

// --- DOM element creation (from create-long-press-elements.js) ---
const createElements = () => {
  const root = document.createElement('div');
  root.style.position = 'fixed';
  root.style.width = '1.25rem';
  root.style.height = '1.25rem';
  root.style.pointerEvents = 'none';
  root.style.transform = 'translate(-50%,-50%)';
  root.style.zIndex = '100';

  const circle = document.createElement('div');
  circle.style.position = 'absolute';
  circle.style.top = '0';
  circle.style.left = '0';
  circle.style.width = '1.25rem';
  circle.style.height = '1.25rem';
  circle.style.clipPath = 'inset(0px 0px 0px 50%)';
  circle.style.opacity = '0';
  root.appendChild(circle);

  const circleLeft = document.createElement('div');
  circleLeft.style.boxSizing = 'content-box';
  circleLeft.style.position = 'absolute';
  circleLeft.style.top = '0';
  circleLeft.style.left = '0';
  circleLeft.style.width = '0.8rem';
  circleLeft.style.height = '0.8rem';
  circleLeft.style.border = '0.2rem solid currentcolor';
  circleLeft.style.borderRadius = '0.8rem';
  circleLeft.style.clipPath = 'inset(0px 50% 0px 0px)';
  circleLeft.style.transform = 'rotate(0deg)';
  circle.appendChild(circleLeft);

  const circleRight = document.createElement('div');
  circleRight.style.boxSizing = 'content-box';
  circleRight.style.position = 'absolute';
  circleRight.style.top = '0';
  circleRight.style.left = '0';
  circleRight.style.width = '0.8rem';
  circleRight.style.height = '0.8rem';
  circleRight.style.border = '0.2rem solid currentcolor';
  circleRight.style.borderRadius = '0.8rem';
  circleRight.style.clipPath = 'inset(0px 50% 0px 0px)';
  circleRight.style.transform = 'rotate(0deg)';
  circle.appendChild(circleRight);

  const effect = document.createElement('div');
  effect.style.position = 'absolute';
  effect.style.top = '0';
  effect.style.left = '0';
  effect.style.width = '1.25rem';
  effect.style.height = '1.25rem';
  effect.style.borderRadius = '1.25rem';
  effect.style.background = 'currentcolor';
  effect.style.transform = 'scale(0)';
  effect.style.opacity = '0';
  root.appendChild(effect);

  return { root, circle, circleLeft, circleRight, effect };
};

// --- The hook ---
export const useLongPressIndicator = () => {
  const elementsRef = useRef<ReturnType<typeof createElements> | null>(null);
  const animationsRef = useRef<Animation[]>([]);
  const isStarting = useRef(false);

  // Create / destroy DOM elements
  useEffect(() => {
    const created = createElements();
    created.root.style.color = INDICATOR_COLOR;
    elementsRef.current = created;

    // Always append to document.body so `position: fixed` is relative to
    // the viewport — a transformed ancestor would break fixed positioning.
    document.body.appendChild(created.root);

    return () => {
      for (const a of animationsRef.current) a.cancel();
      animationsRef.current = [];
      created.root.remove();
      elementsRef.current = null;
    };
  }, []);

  const show = useCallback((x: number, y: number) => {
    const el = elementsRef.current;
    if (!el) return;

    isStarting.current = true;

    // Capture current animated state before canceling
    const mainColor = getComputedStyle(el.root).color || 'currentcolor';
    const circleCs = getComputedStyle(el.circle);
    const circleClipPath = circleCs.clipPath || 'inset(0 0 0 50%)';
    const circleOpacity = Number(circleCs.opacity) || 0;
    const effectState = getCurrentTransform(el.effect);
    const leftState = getCurrentTransform(el.circleLeft);
    const rightState = getCurrentTransform(el.circleRight);

    // Cancel running animations, then set inline styles so state persists
    for (const a of animationsRef.current) a.cancel();

    el.root.style.color = INDICATOR_COLOR;
    el.root.style.top = `${y}px`;
    el.root.style.left = `${x}px`;
    el.circle.style.clipPath = circleClipPath;
    el.circle.style.opacity = String(circleOpacity);
    el.effect.style.opacity = String(effectState.opacity);
    el.effect.style.transform = `scale(${effectState.scale})`;
    el.circleLeft.style.transform = `rotate(${leftState.rotate}deg)`;
    el.circleRight.style.transform = `rotate(${rightState.rotate}deg)`;

    // Compute timing based on how far the previous animation progressed
    const progress = leftState.rotate / 360;
    const duration = (1 - progress) * LONG_PRESS_TIME + LONG_PRESS_AFTER_EFFECT_TIME;
    const lp = ((1 - progress) * LONG_PRESS_TIME) / duration;
    const half = lp / 2;
    const afterEffect = lp + (1 - lp) / 4;
    const opts = { duration, delay: LONG_PRESS_EFFECT_DELAY, fill: 'forwards' as const };
    // CSS animation-timing-function applies per-segment; in Web Animations API
    // that maps to per-keyframe easing (not the effect-level easing option).
    const eo = 'ease-out';

    const anims: Animation[] = [];

    // Root: color + opacity
    anims.push(
      el.root.animate(
        [
          { color: mainColor, opacity: 1, offset: 0, easing: eo },
          { color: mainColor, opacity: 1, offset: lp, easing: eo },
          { color: INDICATOR_ACTIVE_COLOR, opacity: 0.8, offset: 1 },
        ],
        opts,
      ),
    );

    // Effect circle: scale + fade
    anims.push(
      el.effect.animate(
        [
          {
            opacity: effectState.opacity,
            transform: `scale(${effectState.scale})`,
            offset: 0,
            easing: eo,
          },
          {
            opacity: effectState.opacity,
            transform: `scale(${effectState.scale})`,
            offset: lp,
            easing: eo,
          },
          { opacity: 0.66, transform: 'scale(1.5)', offset: afterEffect, easing: eo },
          { opacity: 0, transform: 'scale(2)', offset: 0.99, easing: eo },
          { opacity: 0, transform: 'scale(0)', offset: 1 },
        ],
        opts,
      ),
    );

    // Circle left half: rotation
    anims.push(
      el.circleLeft.animate(
        [
          { transform: `rotate(${leftState.rotate}deg)`, offset: 0 },
          { transform: 'rotate(360deg)', offset: lp },
          { transform: 'rotate(360deg)', offset: 1 },
        ],
        opts,
      ),
    );

    // Circle right half: rotation
    anims.push(
      el.circleRight.animate(
        [
          { transform: `rotate(${rightState.rotate}deg)`, offset: 0 },
          { transform: 'rotate(180deg)', offset: half },
          { transform: 'rotate(180deg)', offset: 1 },
        ],
        opts,
      ),
    );

    // Circle container: clip-path reveal
    anims.push(
      el.circle.animate(
        [
          { clipPath: circleClipPath, opacity: circleOpacity, offset: 0 },
          { clipPath: circleClipPath, opacity: 1, offset: half },
          { clipPath: 'inset(0)', opacity: 1, offset: half + 0.0001 },
          { clipPath: 'inset(0)', opacity: 1, offset: 1 },
        ],
        opts,
      ),
    );

    animationsRef.current = anims;
  }, []);

  const hide = useCallback(() => {
    const el = elementsRef.current;
    if (!el || !isStarting.current) return;

    isStarting.current = false;

    // Capture current animated state before canceling
    const mainColor = getComputedStyle(el.root).color || 'currentcolor';
    const circleCs = getComputedStyle(el.circle);
    const circleClipPath = circleCs.clipPath || 'inset(0px)';
    const circleOpacity = Number(circleCs.opacity) || 1;
    const effectState = getCurrentTransform(el.effect);

    // Detect if past the 50% mark of the circle animation
    const pastHalf = circleCs.clipPath.slice(-2, -1) === 'x';

    const leftState = getCurrentTransform(el.circleLeft, pastHalf);
    const rightState = getCurrentTransform(el.circleRight);

    // Cancel running animations, then set inline styles so state persists
    for (const a of animationsRef.current) a.cancel();

    el.root.style.color = mainColor;
    el.circle.style.clipPath = circleClipPath;
    el.circle.style.opacity = String(circleOpacity);
    el.effect.style.opacity = String(effectState.opacity);
    el.effect.style.transform = `scale(${effectState.scale})`;
    el.circleLeft.style.transform = `rotate(${leftState.rotate}deg)`;
    el.circleRight.style.transform = `rotate(${rightState.rotate}deg)`;

    // Compute timing
    const progress = leftState.rotate / 360;
    const duration = progress * LONG_PRESS_REVERT_EFFECT_TIME;

    if (duration < 1) {
      animationsRef.current = [];
      return;
    }

    const rotated = Math.min(1, progress);
    const half = rotated > 0.5 ? 1 - 0.5 / rotated : 0;
    const opts = { duration, fill: 'forwards' as const };

    const anims: Animation[] = [];

    // Root: color revert
    anims.push(el.root.animate([{ color: mainColor }, { color: INDICATOR_COLOR }], opts));

    // Effect: fade out
    anims.push(
      el.effect.animate(
        [
          { opacity: effectState.opacity, transform: `scale(${effectState.scale})`, offset: 0 },
          { opacity: 0, transform: `scale(${effectState.scale + 0.5})`, offset: 0.99 },
          { opacity: 0, transform: 'scale(0)', offset: 1 },
        ],
        opts,
      ),
    );

    // Circle left: uses circleRight's rotation (intentionally swapped, matches regl-scatterplot)
    anims.push(
      el.circleLeft.animate(
        half > 0
          ? [
              { transform: `rotate(${rightState.rotate}deg)`, offset: 0 },
              { transform: `rotate(${rightState.rotate}deg)`, offset: half },
              { transform: 'rotate(0deg)', offset: 1 },
            ]
          : [
              { transform: `rotate(${rightState.rotate}deg)`, offset: 0 },
              { transform: 'rotate(0deg)', offset: 1 },
            ],
        opts,
      ),
    );

    // Circle right: uses circleLeft's rotation (intentionally swapped)
    anims.push(
      el.circleRight.animate(
        [
          { transform: `rotate(${leftState.rotate}deg)`, offset: 0 },
          { transform: 'rotate(0deg)', offset: 1 },
        ],
        opts,
      ),
    );

    // Circle container: clip-path hide
    anims.push(
      el.circle.animate(
        half > 0
          ? [
              { clipPath: circleClipPath, opacity: circleOpacity, offset: 0 },
              { clipPath: circleClipPath, opacity: circleOpacity, offset: half },
              { clipPath: 'inset(0 0 0 50%)', opacity: circleOpacity, offset: half + 0.0001 },
              { clipPath: 'inset(0 0 0 50%)', opacity: 0, offset: 1 },
            ]
          : [
              { clipPath: circleClipPath, opacity: circleOpacity, offset: 0 },
              { clipPath: 'inset(0 0 0 50%)', opacity: circleOpacity, offset: 0.0001 },
              { clipPath: 'inset(0 0 0 50%)', opacity: 0, offset: 1 },
            ],
        opts,
      ),
    );

    animationsRef.current = anims;
  }, []);

  return { show, hide };
};
