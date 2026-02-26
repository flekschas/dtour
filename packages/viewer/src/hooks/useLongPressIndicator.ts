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

const circleLeftInRule = (percent: number, angle: number) => `
  @keyframes leftSpinIn {
    0% { transform: rotate(${angle}deg); }
    ${percent}%, 100% { transform: rotate(360deg); }
  }`;

const circleRightInRule = (halfPercent: number, angle: number) => `
  @keyframes rightSpinIn {
    0% { transform: rotate(${angle}deg); }
    ${halfPercent}%, 100% { transform: rotate(180deg); }
  }`;

// --- Keyframe rule generators (out) ---
const mainOutRule = (currentColor: string, targetColor: string) => `
  @keyframes mainOut {
    0% { color: ${currentColor}; }
    100% { color: ${targetColor}; }
  }`;

const effectOutRule = (opacity: number, scale: number) => `
  @keyframes effectOut {
    0% { opacity: ${opacity}; transform: scale(${scale}); }
    99% { opacity: 0; transform: scale(${scale + 0.5}); }
    100% { opacity: 0; transform: scale(0); }
  }`;

const circleRightOutRule = (halfPercent: number, angle: number) => `
  @keyframes rightSpinOut {
    0%, ${halfPercent}% { transform: rotate(${angle}deg); }
    100% { transform: rotate(0deg); }
  }`;

const circleLeftOutRule = (angle: number) => `
  @keyframes leftSpinOut {
    0% { transform: rotate(${angle}deg); }
    100% { transform: rotate(0deg); }
  }`;

const circleOutRule = (halfPercent: number, clipPath: string, opacity: number) => `
  @keyframes circleOut {
    0%, ${halfPercent}% { clip-path: ${clipPath}; opacity: ${opacity}; }
    ${halfPercent + 0.01}% { clip-path: inset(0 0 0 50%); opacity: ${opacity}; }
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
  const progress = circleLeftRotation / 360;
  const actualTime = (1 - progress) * time + extraTime;
  const longPressPercent = Math.round((((1 - progress) * time) / actualTime) * 100);
  const halfPercent = Math.round(longPressPercent / 2);
  const afterEffectPercent = longPressPercent + (100 - longPressPercent) / 4;

  return {
    rules: {
      main: mainInRule(longPressPercent, currentColor, targetColor),
      effect: effectInRule(longPressPercent, afterEffectPercent, effectOpacity, effectScale),
      circleRight: circleRightInRule(halfPercent, circleRightRotation),
      circleLeft: circleLeftInRule(longPressPercent, circleLeftRotation),
      circle: circleInRule(halfPercent, circleClipPath, circleOpacity),
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
  const progress = circleLeftRotation / 360;
  const actualTime = progress * time;
  const rotatedPercent = Math.min(100, progress * 100);
  const halfPercent = rotatedPercent > 50 ? Math.round((1 - 50 / rotatedPercent) * 100) : 0;

  return {
    rules: {
      main: mainOutRule(currentColor, targetColor),
      effect: effectOutRule(effectOpacity, effectScale),
      circleRight: circleRightOutRule(halfPercent, circleRightRotation),
      circleLeft: circleLeftOutRule(circleLeftRotation),
      circle: circleOutRule(halfPercent, circleClipPath, circleOpacity),
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
  const elementsRef = useRef<ReturnType<typeof createElements> | null>(null);
  const ruleIndicesRef = useRef<RuleIndices>(emptyIndices());
  const isStarting = useRef(false);

  // Create / destroy DOM elements
  useEffect(() => {
    const created = createElements();
    created.root.style.color = INDICATOR_COLOR;
    elementsRef.current = created;

    const parent = parentRef.current ?? document.body;
    parent.appendChild(created.root);

    return () => {
      created.root.remove();
      elementsRef.current = null;
    };
  }, [parentRef]);

  const show = useCallback((x: number, y: number) => {
    const el = elementsRef.current;
    if (!el) return;

    isStarting.current = true;

    const mainStyle = getComputedStyle(el.root);
    el.root.style.color = INDICATOR_COLOR;
    el.root.style.top = `${y}px`;
    el.root.style.left = `${x}px`;
    el.root.style.animation = 'none';

    const circleStyle = getComputedStyle(el.circle);
    el.circle.style.clipPath = circleStyle.clipPath;
    el.circle.style.opacity = circleStyle.opacity;
    el.circle.style.animation = 'none';

    const effectStyle = getCurrentTransform(el.effect);
    el.effect.style.opacity = String(effectStyle.opacity);
    el.effect.style.transform = `scale(${effectStyle.scale})`;
    el.effect.style.animation = 'none';

    const leftStyle = getCurrentTransform(el.circleLeft);
    el.circleLeft.style.transform = `rotate(${leftStyle.rotate}deg)`;
    el.circleLeft.style.animation = 'none';

    const rightStyle = getCurrentTransform(el.circleRight);
    el.circleRight.style.transform = `rotate(${rightStyle.rotate}deg)`;
    el.circleRight.style.animation = 'none';

    requestAnimationFrame(() => {
      if (!isStarting.current || !elementsRef.current) return;

      const indices = ruleIndicesRef.current;
      safeRemove(indices, 'circleIn');
      safeRemove(indices, 'circleRightIn');
      safeRemove(indices, 'circleLeftIn');
      safeRemove(indices, 'effectIn');
      safeRemove(indices, 'mainIn');

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

      indices.mainIn = addRule(rules.main);
      indices.effectIn = addRule(rules.effect);
      indices.circleLeftIn = addRule(rules.circleLeft);
      indices.circleRightIn = addRule(rules.circleRight);
      indices.circleIn = addRule(rules.circle);

      el.root.style.animation = names.main;
      el.effect.style.animation = names.effect;
      el.circleLeft.style.animation = names.circleLeft;
      el.circleRight.style.animation = names.circleRight;
      el.circle.style.animation = names.circle;
    });
  }, []);

  const hide = useCallback(() => {
    const el = elementsRef.current;
    if (!el || !isStarting.current) return;

    isStarting.current = false;

    const mainStyle = getComputedStyle(el.root);
    el.root.style.color = mainStyle.color;
    el.root.style.animation = 'none';

    const circleStyle = getComputedStyle(el.circle);
    el.circle.style.clipPath = circleStyle.clipPath;
    el.circle.style.opacity = circleStyle.opacity;
    el.circle.style.animation = 'none';

    const effectStyle = getCurrentTransform(el.effect);
    el.effect.style.opacity = String(effectStyle.opacity);
    el.effect.style.transform = `scale(${effectStyle.scale})`;
    el.effect.style.animation = 'none';

    // Detect if past the 50% mark of the circle animation
    const pastHalf = circleStyle.clipPath.slice(-2, -1) === 'x';

    const leftStyle = getCurrentTransform(el.circleLeft, pastHalf);
    el.circleLeft.style.transform = `rotate(${leftStyle.rotate}deg)`;
    el.circleLeft.style.animation = 'none';

    const rightStyle = getCurrentTransform(el.circleRight);
    el.circleRight.style.transform = `rotate(${rightStyle.rotate}deg)`;
    el.circleRight.style.animation = 'none';

    requestAnimationFrame(() => {
      const indices = ruleIndicesRef.current;
      safeRemove(indices, 'circleOut');
      safeRemove(indices, 'circleRightOut');
      safeRemove(indices, 'circleLeftOut');
      safeRemove(indices, 'effectOut');
      safeRemove(indices, 'mainOut');

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

      indices.mainOut = addRule(rules.main);
      indices.effectOut = addRule(rules.effect);
      indices.circleLeftOut = addRule(rules.circleLeft);
      indices.circleRightOut = addRule(rules.circleRight);
      indices.circleOut = addRule(rules.circle);

      el.root.style.animation = names.main;
      el.effect.style.animation = names.effect;
      el.circleLeft.style.animation = names.circleLeft;
      el.circleRight.style.animation = names.circleRight;
      el.circle.style.animation = names.circle;
    });
  }, []);

  return { show, hide };
};
