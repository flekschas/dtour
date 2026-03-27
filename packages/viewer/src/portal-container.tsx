import { createContext, useContext } from 'react';

/**
 * Context for overriding where Radix portals render their content.
 * When provided, dropdown menus, tooltips, etc. portal into the given
 * element instead of `document.body`.  This is required for Shadow DOM
 * isolation (e.g. the anywidget/Marimo embed) so that portalled content
 * stays inside the shadow root and inherits scoped styles.
 */
export const PortalContainerContext = createContext<HTMLElement | undefined>(undefined);

export function usePortalContainer(): HTMLElement | undefined {
  return useContext(PortalContainerContext);
}
