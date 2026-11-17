import { GithubLogoIcon, ScrollIcon } from '@phosphor-icons/react';

const LINK_CLASS =
  'pointer-events-auto hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors';

const Separator = () => <span className="bg-dtour-text-muted/30 w-px h-4 mx-1" />;

/**
 * Credits and links at the bottom of the landing page. Rendered in flow (not
 * absolutely positioned) so it can never overlap the content above it, and
 * wraps centered on narrow screens.
 */
export const LandingFooter = () => (
  <p className="shrink-0 w-full py-6 text-xs text-dtour-text-muted/60 flex flex-wrap items-center justify-center gap-x-1 gap-y-2">
    <span>
      Made by{' '}
      <a
        href="https://lekschas.de"
        target="_blank"
        rel="noopener noreferrer"
        className={LINK_CLASS}
      >
        Fritz
      </a>
      {' and '}
      <a href="https://nvictus.me" target="_blank" rel="noopener noreferrer" className={LINK_CLASS}>
        Nezar
      </a>
      .
    </span>
    <Separator />
    <a
      href="https://github.com/flekschas/dtour"
      target="_blank"
      rel="noopener noreferrer"
      className={`flex items-center gap-1 ${LINK_CLASS}`}
    >
      <GithubLogoIcon size={14} weight="fill" />
      GitHub
    </a>
    <Separator />
    <a
      href="https://arxiv.org/abs/2605.04306"
      target="_blank"
      rel="noopener noreferrer"
      className={`flex items-center gap-1 ${LINK_CLASS}`}
    >
      <ScrollIcon size={14} weight="fill" />
      arXiv
    </a>
  </p>
);
