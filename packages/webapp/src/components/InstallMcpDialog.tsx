import {
  CaretRightIcon,
  CheckIcon,
  CopyIcon,
  DownloadSimpleIcon,
  XIcon,
} from '@phosphor-icons/react';
import { AnimatePresence, motion } from 'motion/react';
import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import { Button } from './ui/button.tsx';

const MCPB_URL = 'https://github.com/flekschas/dtour/releases/latest/download/dtour.mcpb';

const CLAUDE_CODE_COMMAND = '/plugin install dtour --marketplace flekschas/dtour';

const UVX_COMMAND = 'uvx --from "dtour[mcp,umap]" dtour-mcp';

const MCP_CONFIG = `{
  "mcpServers": {
    "dtour": {
      "command": "uvx",
      "args": ["--from", "dtour[mcp,umap]", "dtour-mcp"]
    }
  }
}`;

const EXAMPLE_PROMPT =
  'Using dtour, show a PCA tour of the pixel columns in https://data.dtour.dev/digits.csv next to a 2D UMAP, colored by digit.';

const Snippet = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timeout = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timeout);
  }, [copied]);

  return (
    <div className="relative group">
      <pre className="text-xs leading-relaxed bg-dtour-bg/60 border border-dtour-border rounded-md p-3 pr-10 whitespace-pre-wrap break-words font-mono text-dtour-text">
        {text}
      </pre>
      <button
        type="button"
        aria-label={copied ? 'Copied' : 'Copy'}
        className="absolute top-2 right-2 p-1 rounded text-dtour-text-muted hover:text-dtour-text hover:bg-dtour-surface transition-colors cursor-pointer"
        onClick={() => navigator.clipboard.writeText(text).then(() => setCopied(true))}
      >
        {copied ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
      </button>
    </div>
  );
};

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <section className="flex flex-col gap-2">
    <h3 className="text-sm font-semibold text-dtour-highlight">{title}</h3>
    {children}
  </section>
);

const Collapsible = ({ title, children }: { title: string; children: ReactNode }) => (
  <details className="group">
    <summary className="text-sm font-semibold text-dtour-highlight cursor-pointer select-none list-none flex items-center gap-1.5">
      <CaretRightIcon
        size={12}
        weight="bold"
        className="transition-transform group-open:rotate-90"
      />
      {title}
    </summary>
    <div className="flex flex-col gap-2 mt-2">{children}</div>
  </details>
);

const Link = ({ href, children }: { href: string; children: ReactNode }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="underline underline-offset-2 hover:text-dtour-text"
  >
    {children}
  </a>
);

/** How to install dtour's MCP server, with an example prompt to try. */
export const InstallMcpDialog = ({ open, onClose }: { open: boolean; onClose: () => void }) => {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <div
            role="presentation"
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={onClose}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="install-mcp-title"
            className="relative z-10 w-full max-w-xl mx-4 max-h-[85vh] overflow-y-auto rounded-lg border border-dtour-border bg-dtour-surface p-6 flex flex-col gap-5 text-sm text-dtour-text-muted"
          >
            <button
              type="button"
              aria-label="Close"
              className="absolute top-4 right-4 text-dtour-text-muted hover:text-dtour-text transition-colors cursor-pointer"
              onClick={onClose}
            >
              <XIcon size={20} />
            </button>
            <div className="flex flex-col gap-1.5 pr-8">
              <h2 id="install-mcp-title" className="text-base font-semibold text-dtour-highlight">
                Visually Explore your data with AI
              </h2>
              <p>
                dtour's MCP server lets your favorite MCP-enabled AI friend show interactive tours
                right in the chat. It reads your files and computes the tours on your computer. You
                can also create a classic 2D UMAP this way.
              </p>
            </div>

            <Section title="Claude Desktop">
              <p>Download the MCP server and install it with one click:</p>
              <Button asChild className="self-start">
                <a href={MCPB_URL}>
                  <DownloadSimpleIcon size={16} weight="bold" />
                  Download dtour.mcpb
                </a>
              </Button>
            </Section>

            <div className="flex flex-col gap-3">
              <Collapsible title="ChatGPT">
                <p>
                  In the ChatGPT desktop app, open Settings → MCP servers → Add server, choose
                  STDIO, and enter this command. It needs{' '}
                  <Link href="https://docs.astral.sh/uv/">uv</Link>. ChatGPT on the web only
                  connects to remote MCP servers.
                </p>
                <Snippet text={UVX_COMMAND} />
              </Collapsible>

              <Collapsible title="Claude Code">
                <p>
                  Install the plugin, which also includes the dtour skill. It needs{' '}
                  <Link href="https://docs.astral.sh/uv/">uv</Link>.
                </p>
                <Snippet text={CLAUDE_CODE_COMMAND} />
              </Collapsible>

              <Collapsible title="Other MCP apps">
                <p>
                  Add the server to the app's MCP config. It needs{' '}
                  <Link href="https://docs.astral.sh/uv/">uv</Link>.
                </p>
                <Snippet text={MCP_CONFIG} />
              </Collapsible>
            </div>

            <Section title="Try it">
              <p>Ask your assistant:</p>
              <Snippet text={EXAMPLE_PROMPT} />
              <p>
                Lasso points in either view to select them in both, then ask what's special about
                them.
              </p>
            </Section>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
