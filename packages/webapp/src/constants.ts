/** Height-to-width ratio of the dtour logo (its SVG viewBox is 31.2 × 9.1). */
const LOGO_ASPECT = 9.1 / 31.2;

/** Below this viewport width (Tailwind's `sm`), the landing page uses a compact hero. */
const COMPACT_MAX_WIDTH = 640;

/**
 * Size of the landing hero logo and where the content below it starts, in px.
 *
 * On wide screens the logo spans the content block (max-w-lg = 512px minus the
 * overlay's px-4 padding) and sits right above 30% of the viewport height. On
 * phones that leaves too little room for the content, so the logo shrinks and
 * hugs the top of the screen instead.
 *
 * Uses `innerHeight`, not CSS `vh`: on iOS, `vh` is the height with the browser
 * toolbars hidden, which would push the content below the logo's actual bottom.
 */
export const getHeroLayout = () => {
  const { innerWidth: width, innerHeight: height } = window;
  if (width < COMPACT_MAX_WIDTH) {
    const logoWidth = Math.min(240, width - 32);
    const logoTop = 32;
    const contentTop = logoTop + logoWidth * LOGO_ASPECT + 24;
    return { logoWidth, contentTop };
  }
  return { logoWidth: Math.min(512, width - 32), contentTop: height * 0.3 };
};
