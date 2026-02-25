import { useCallback, useEffect, useRef } from 'react';

// --- Constants (from regl-scatterplot) ---
const LONG_PRESS_TIME = 750;
const LONG_PRESS_AFTER_EFFECT_TIME = 500;
const LONG_PRESS_EFFECT_DELAY = 100;
const LONG_PRESS_REVERT_EFFECT_TIME = 250;

const INDICATOR_COLOR = '#4f8ff7';
const INDICATOR_ACTIVE_COLOR = '#4f8ff7';

// --- Stylesheet management ---
let cachedSheet: CSSStyleSheet | null = null;

const getSheet = (): CSSStyleSheet => {
  if (!cachedSheet) {
    const el = document.createElement('style');
    document.head.appendChild(el);
    cachedSheet = el.sheet!;
  }
  return cachedSheet;
};

const addRule = (rule: string): number => {
  const sheet = getSheet();
  const idx = sheet.cssRules.length;
  sheet.insertRule(rule, idx);
  return idx;
};

const removeRule = (index: number) => {
  getSheet().deleteRule(index);
};

// --- Animation name generators ---
const mainInAnim = (t: number, d: number) => `${t}ms ease-out mainIn ${d}ms 1 normal forwards`;
const effectInAnim = (t: number, d: number) => `${t}ms ease-out effectIn ${d}ms 1 normal forwards`;
const circleLeftInAnim = (t: number, d: number) =>
  `${t}ms linear leftSpinIn ${d}ms 1 normal forwards`;
const circleRightInAnim = (t: number, d: number) =>
  `${t}ms linear rightSpinIn ${d}ms 1 normal forwards`;
const circleInAnim = (t: number, d: number) => `${t}ms linear circleIn ${d}ms 1 normal forwards`;

const mainOutAnim = (t: number) => `${t}ms linear mainOut 0s 1 normal forwards`;
const effectOutAnim = (t: number) => `${t}ms linear effectOut 0s 1 normal forwards`;
const circleLeftOutAnim = (t: number) => `${t}ms linear leftSpinOut 0s 1 normal forwards`;
const circleRightOutAnim = (t: number) => `${t}ms linear rightSpinOut 0s 1 normal forwards`;
const circleOutAnim = (t: number) => `${t}ms linear circleOut 0s 1 normal forwards`;

// --- Keyframe rule generators (in) ---
const mainInRule = (percent: number, currentColor: string, targetColor: string) => `
  @keyframes mainIn {
    0% { color: ${currentColor}; opacity: 0; }
    0%, ${percent}% { color: ${currentColor}; opacity: 1; }
    100% { color: ${targetColor}; opacity: 0.8; }
  }`;

const effectInRule = (percent: number, afterPercent: number, opacity: number, scale: number) => `
  @keyframes effectIn {
    0%, ${percent}% { opacity: ${opacity}; transform: scale(${scale}); }
    ${afterPercent}% { opacity: 0.66; transform: scale(1.5); }
    99% { opacity: 0; transform: scale(2); }
    100% { opacity: 0; transform: scale(0); }
  }`;

const circleInRule = (halfPercent: number, clipPath: string, opacity: number) => `
  @keyframes circleIn {
    0% { clip-path: ${clipPath}; opacity: ${opacity}; }
    ${halfPercent}% { clip-path: ${clipPath}; opacity: 1; }
    ${halfPercent + 0.01}%, 100% { clip-path: inset(0); opacity: 1; }
  }`;

const circleLeftInRule = (pct: number, angle: number) => `
  @keyframes leftSpinIn {
    0% { transform: rotate(${angle}deg); }
    ${pct}%, 100% { transform: rotate(360deg); }
  }`;

const circleRightInRule = (halfPct: number, angle: number) => `
  @keyframes rightSpinIn {
    0% { transform: rotate(${angle}deg); }
    ${halfPct}%, 100% { transform: rotate(180deg); }
  }`;

// --- Keyframe rule generators (out) ---
const mainOutRule = (curColor: string, tgtColor: string) => `
  @keyframes mainOut {
    0% { color: ${curColor}; }
    100% { color: ${tgtColor}; }
  }`;

const effectOutRule = (opacity: number, scale: number) => `
  @keyframes effectOut {
    0% { opacity: ${opacity}; transform: scale(${scale}); }
    99% { opacity: 0; transform: scale(${scale + 0.5}); }
    100% { opacity: 0; transform: scale(0); }
  }`;

const circleRightOutRule = (halfPct: number, angle: number) => `
  @keyframes rightSpinOut {
    0%, ${halfPct}% { transform: rotate(${angle}deg); }
    100% { transform: rotate(0deg); }
  }`;

const circleLeftOutRule = (angle: number) => `
  @keyframes leftSpinOut {
    0% { transform: rotate(${angle}deg); }
    100% { transform: rotate(0deg); }
  }`;

const circleOutRule = (halfPct: number, clipPath: string, opacity: number) => `
  @keyframes circleOut {
    0%, ${halfPct}% { clip-path: ${clipPath}; opacity: ${opacity}; }
    ${halfPct + 0.01}% { clip-path: inset(0 0 0 50%); opacity: ${opacity}; }
    100% { clip-path: inset(0 0 0 50%); opacity: 0; }
  }`;

// --- Animation creation helpers ---
type AnimKeys = 'main' | 'effect' | 'circleLeft' | 'circleRight' | 'circle';
type Animations = {
  rules: Record<AnimKeys, string>;
  names: Record<AnimKeys, string>;
};

const createInAnimations = ({
  time = LONG_PRESS_TIME,
  extraTime = LONG_PRESS_AFTER_EFFECT_TIME,
  delay = LONG_PRESS_EFFECT_DELAY,
  currentColor,
  targetColor,
  effectOpacity,
  effectScale,
  circleLeftRotation,
  circleRightRotation,
  circleClipPath,
  circleOpacity,
}: {
  time?: number;
  extraTime?: number;
  delay?: number;
  currentColor: string;
  targetColor: string;
  effectOpacity: number;
  effectScale: number;
  circleLeftRotation: number;
  circleRightRotation: number;
  circleClipPath: string;
  circleOpacity: number;
}): Animations => {
  const p = circleLeftRotation / 360;
  const actualTime = (1 - p) * time + extraTime;
  const longPressPct = Math.round((((1 - p) * time) / actualTime) * 100);
  const halfPct = Math.round(longPressPct / 2);
  const afterEffectPct = longPressPct + (100 - longPressPct) / 4;

  return {
    rules: {
      main: mainInRule(longPressPct, currentColor, targetColor),
      effect: effectInRule(longPressPct, afterEffectPct, effectOpacity, effectScale),
      circleRight: circleRightInRule(halfPct, circleRightRotation),
      circleLeft: circleLeftInRule(longPressPct, circleLeftRotation),
      circle: circleInRule(halfPct, circleClipPath, circleOpacity),
    },
    names: {
      main: mainInAnim(actualTime, delay),
      effect: effectInAnim(actualTime, delay),
      circleLeft: circleLeftInAnim(actualTime, delay),
      circleRight: circleRightInAnim(actualTime, delay),
      circle: circleInAnim(actualTime, delay),
    },
  };
};

const createOutAnimations = ({
  time = LONG_PRESS_REVERT_EFFECT_TIME,
  currentColor,
  targetColor,
  effectOpacity,
  effectScale,
  circleLeftRotation,
  circleRightRotation,
  circleClipPath,
  circleOpacity,
}: {
  time?: number;
  currentColor: string;
  targetColor: string;
  effectOpacity: number;
  effectScale: number;
  circleLeftRotation: number;
  circleRightRotation: number;
  circleClipPath: string;
  circleOpacity: number;
}): Animations => {
  const p = circleLeftRotation / 360;
  const actualTime = p * time;
  const rotatedPct = Math.min(100, p * 100);
  const halfPct = rotatedPct > 50 ? Math.round((1 - 50 / rotatedPct) * 100) : 0;

  return {
    rules: {
      main: mainOutRule(currentColor, targetColor),
      effect: effectOutRule(effectOpacity, effectScale),
      circleRight: circleRightOutRule(halfPct, circleRightRotation),
      circleLeft: circleLeftOutRule(circleLeftRotation),
      circle: circleOutRule(halfPct, circleClipPath, circleOpacity),
    },
    names: {
      main: mainOutAnim(actualTime),
      effect: effectOutAnim(actualTime),
      // Note: intentionally swapped (matches regl-scatterplot)
      circleRight: circleLeftOutAnim(actualTime),
      circleLeft: circleRightOutAnim(actualTime),
      circle: circleOutAnim(actualTime),
    },
  };
};

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

// --- Rule index tracker ---
type RuleIndices = {
  mainIn: number | null;
  effectIn: number | null;
  circleLeftIn: number | null;
  circleRightIn: number | null;
  circleIn: number | null;
  mainOut: number | null;
  effectOut: number | null;
  circleLeftOut: number | null;
  circleRightOut: number | null;
  circleOut: number | null;
};

const emptyIndices = (): RuleIndices => ({
  mainIn: null,
  effectIn: null,
  circleLeftIn: null,
  circleRightIn: null,
  circleIn: null,
  mainOut: null,
  effectOut: null,
  circleLeftOut: null,
  circleRightOut: null,
  circleOut: null,
});

const safeRemove = (indices: RuleIndices, key: keyof RuleIndices) => {
  const val = indices[key];
  if (val !== null) {
    removeRule(val);
    indices[key] = null;
  }
};

// --- The hook ---
export const useLongPressIndicator = (parentRef: React.RefObject<HTMLElement | null>) => {
  const els = useRef<ReturnType<typeof createElements> | null>(null);
  const idx = useRef<RuleIndices>(emptyIndices());
  const isStarting = useRef(false);

  // Create / destroy DOM elements
  useEffect(() => {
    const elements = createElements();
    elements.root.style.color = INDICATOR_COLOR;
    els.current = elements;

    const parent = parentRef.current ?? document.body;
    parent.appendChild(elements.root);

    return () => {
      elements.root.remove();
      els.current = null;
    };
  }, [parentRef]);

  const show = useCallback((x: number, y: number) => {
    const e = els.current;
    if (!e) return;

    isStarting.current = true;

    const mainStyle = getComputedStyle(e.root);
    e.root.style.color = INDICATOR_COLOR;
    e.root.style.top = `${y}px`;
    e.root.style.left = `${x}px`;
    e.root.style.animation = 'none';

    const circleStyle = getComputedStyle(e.circle);
    e.circle.style.clipPath = circleStyle.clipPath;
    e.circle.style.opacity = circleStyle.opacity;
    e.circle.style.animation = 'none';

    const effectStyle = getCurrentTransform(e.effect);
    e.effect.style.opacity = String(effectStyle.opacity);
    e.effect.style.transform = `scale(${effectStyle.scale})`;
    e.effect.style.animation = 'none';

    const leftStyle = getCurrentTransform(e.circleLeft);
    e.circleLeft.style.transform = `rotate(${leftStyle.rotate}deg)`;
    e.circleLeft.style.animation = 'none';

    const rightStyle = getCurrentTransform(e.circleRight);
    e.circleRight.style.transform = `rotate(${rightStyle.rotate}deg)`;
    e.circleRight.style.animation = 'none';

    requestAnimationFrame(() => {
      if (!isStarting.current || !els.current) return;

      const i = idx.current;
      safeRemove(i, 'circleIn');
      safeRemove(i, 'circleRightIn');
      safeRemove(i, 'circleLeftIn');
      safeRemove(i, 'effectIn');
      safeRemove(i, 'mainIn');

      const { rules, names } = createInAnimations({
        currentColor: mainStyle.color || 'currentcolor',
        targetColor: INDICATOR_ACTIVE_COLOR,
        effectOpacity: effectStyle.opacity || 0,
        effectScale: effectStyle.scale || 0,
        circleLeftRotation: leftStyle.rotate || 0,
        circleRightRotation: rightStyle.rotate || 0,
        circleClipPath: circleStyle.clipPath || 'inset(0 0 0 50%)',
        circleOpacity: Number(circleStyle.opacity) || 0,
      });

      i.mainIn = addRule(rules.main);
      i.effectIn = addRule(rules.effect);
      i.circleLeftIn = addRule(rules.circleLeft);
      i.circleRightIn = addRule(rules.circleRight);
      i.circleIn = addRule(rules.circle);

      e.root.style.animation = names.main;
      e.effect.style.animation = names.effect;
      e.circleLeft.style.animation = names.circleLeft;
      e.circleRight.style.animation = names.circleRight;
      e.circle.style.animation = names.circle;
    });
  }, []);

  const hide = useCallback(() => {
    const e = els.current;
    if (!e || !isStarting.current) return;

    isStarting.current = false;

    const mainStyle = getComputedStyle(e.root);
    e.root.style.color = mainStyle.color;
    e.root.style.animation = 'none';

    const circleStyle = getComputedStyle(e.circle);
    e.circle.style.clipPath = circleStyle.clipPath;
    e.circle.style.opacity = circleStyle.opacity;
    e.circle.style.animation = 'none';

    const effectStyle = getCurrentTransform(e.effect);
    e.effect.style.opacity = String(effectStyle.opacity);
    e.effect.style.transform = `scale(${effectStyle.scale})`;
    e.effect.style.animation = 'none';

    // Detect if past the 50% mark of the circle animation
    const pastHalf = circleStyle.clipPath.slice(-2, -1) === 'x';

    const leftStyle = getCurrentTransform(e.circleLeft, pastHalf);
    e.circleLeft.style.transform = `rotate(${leftStyle.rotate}deg)`;
    e.circleLeft.style.animation = 'none';

    const rightStyle = getCurrentTransform(e.circleRight);
    e.circleRight.style.transform = `rotate(${rightStyle.rotate}deg)`;
    e.circleRight.style.animation = 'none';

    requestAnimationFrame(() => {
      const i = idx.current;
      safeRemove(i, 'circleOut');
      safeRemove(i, 'circleRightOut');
      safeRemove(i, 'circleLeftOut');
      safeRemove(i, 'effectOut');
      safeRemove(i, 'mainOut');

      const { rules, names } = createOutAnimations({
        currentColor: mainStyle.color || 'currentcolor',
        targetColor: INDICATOR_COLOR,
        effectOpacity: effectStyle.opacity || 0,
        effectScale: effectStyle.scale || 0,
        circleLeftRotation: leftStyle.rotate || 0,
        circleRightRotation: rightStyle.rotate || 0,
        circleClipPath: circleStyle.clipPath || 'inset(0px)',
        circleOpacity: Number(circleStyle.opacity) || 1,
      });

      i.mainOut = addRule(rules.main);
      i.effectOut = addRule(rules.effect);
      i.circleLeftOut = addRule(rules.circleLeft);
      i.circleRightOut = addRule(rules.circleRight);
      i.circleOut = addRule(rules.circle);

      e.root.style.animation = names.main;
      e.effect.style.animation = names.effect;
      e.circleLeft.style.animation = names.circleLeft;
      e.circleRight.style.animation = names.circleRight;
      e.circle.style.animation = names.circle;
    });
  }, []);

  return { show, hide };
};
